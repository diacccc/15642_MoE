/*
 * Custom MoE forward pass for the fixed DeepSeek-V3/R1 geometry.
 *
 * This version deliberately avoids cuBLAS. It keeps the complete routed MoE
 * path on custom CUDA kernels:
 *
 *   1. pack_w13_kernel / pack_w2_kernel
 *        prepack FP8 weights into the exact uint32 B-fragment layout consumed
 *        by mma.sync, removing scalar byte gathers from the GEMM inner loops.
 *
 *   2. route_dispatch_kernel
 *        sigmoid -> group top-k -> global top-k -> normalized weights, then
 *        compact local (token, expert, weight) work into per-expert queues
 *        and emit the active (expert, token-tile) work list.
 *
 *   3. stage1_fp8_mma_kernel
 *        custom FP8 Tensor Core GEMM over true FP8 inputs:
 *        hidden[token, H] x packed W13[expert]^T, then fused SwiGLU
 *        into block-scaled FP8 C[expert, slot, I] plus C_scale.
 *
 *   4. stage2_fp8_mma_scatter_kernel
 *        custom FP8 Tensor Core GEMM with fused epilogue:
 *        C x packed W2[expert]^T over true FP8 fragments with block scales,
 *        then weighted atomic accumulation directly into the final token output.
 *
 *   5. bf16_convert_kernel
 *        converts the float accumulator to the public bf16 result.
 *
 * moe_forward() and the two weight packers are exported to Python.
 */

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

// Fixed DeepSeek-V3/R1 geometry.
static constexpr int E_GLOBAL   = 256;
static constexpr int N_GROUP    = 8;
static constexpr int GROUP_SIZE = E_GLOBAL / N_GROUP;  // 32
static constexpr int TOP_K      = 8;
static constexpr int TOPK_GROUP = 4;
static constexpr int H_FIXED    = 7168;
static constexpr int I_FIXED    = 2048;
static constexpr int G1_COLS    = 2 * I_FIXED;
static constexpr int SCALE_BLOCK = 128;
static constexpr int H_BLOCKS    = H_FIXED / SCALE_BLOCK;
static constexpr int I_BLOCKS    = I_FIXED / SCALE_BLOCK;
static constexpr int G1_BLOCKS   = G1_COLS / SCALE_BLOCK;

// Warp-level FP8 MMA tile.
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 32;
static constexpr int WARP_THREADS = 32;
static constexpr int MMA_WARPS_PER_CTA = 4;
static constexpr int MMA_THREADS = MMA_WARPS_PER_CTA * WARP_THREADS;
static constexpr int MMA_M_FRAGS_PER_WARP = 2;
static constexpr int CTA_M = MMA_M * MMA_M_FRAGS_PER_WARP;
static constexpr int CTA_N = MMA_N * MMA_WARPS_PER_CTA;
static constexpr int STAGE1_N_FRAGS = SCALE_BLOCK / CTA_N;
static constexpr int MMA_A_TILE_ELEMS = CTA_M * MMA_K;
static constexpr float FP8_E4M3_MAX = 448.0f;
static constexpr int H_MMA_TILES = H_FIXED / MMA_K;
static constexpr int I_MMA_TILES = I_FIXED / MMA_K;

__device__ __forceinline__ uint8_t fp32_to_e4m3_byte(float x)
{
    return (uint8_t)__nv_fp8_e4m3(x).__x;
}

__device__ __forceinline__ uint16_t fp32_to_bf16_bits(float x)
{
    uint32_t bits = __float_as_uint(x);
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    return (uint16_t)((bits + rounding_bias) >> 16);
}

__device__ __forceinline__ float bf16_bits_to_fp32(uint16_t x)
{
    return __uint_as_float((uint32_t)x << 16);
}

__device__ __forceinline__ uint32_t pack_e4m3x4(
    uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3)
{
    return ((uint32_t)x0) |
           ((uint32_t)x1 << 8) |
           ((uint32_t)x2 << 16) |
           ((uint32_t)x3 << 24);
}

__host__ __device__ __forceinline__ size_t w13_packed_offset(
    int le, int n, int kt, int thread_in_group, int reg)
{
    return (((((size_t)le * G1_COLS + n) * H_MMA_TILES + kt) * 4 +
             thread_in_group) * 2 + reg);
}

__host__ __device__ __forceinline__ size_t w2_packed_offset(
    int le, int h, int it, int thread_in_group, int reg)
{
    return (((((size_t)le * H_FIXED + h) * I_MMA_TILES + it) * 4 +
             thread_in_group) * 2 + reg);
}

__device__ __forceinline__ void mma_m16n8k32_e4m3(float acc[4],
                                                  const uint32_t a[4],
                                                  const uint32_t b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp_async_16(void* smem_ptr,
                                            const void* gmem_ptr,
                                            int src_bytes)
{
    unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_int), "l"(gmem_ptr), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void warp_reduce_max_pair(float& val, int& idx)
{
    unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(mask, val, offset);
        int other_idx = __shfl_down_sync(mask, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Offline FP8 weight packing for the exact B fragments used by mma.m16n8k32.
// Each packed uint32 stores four consecutive K bytes for one lane fragment.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void pack_w13_kernel(
    const uint8_t* __restrict__ W13,
    uint32_t*      __restrict__ W13_packed,
    int E_local)
{
    size_t total = (size_t)E_local * G1_COLS * H_MMA_TILES * 4 * 2;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= total) return;

    int reg = (int)(idx & 1);
    size_t t = idx >> 1;
    int thread_in_group = (int)(t & 3);
    t >>= 2;
    int kt = (int)(t % H_MMA_TILES);
    t /= H_MMA_TILES;
    int n = (int)(t % G1_COLS);
    int le = (int)(t / G1_COLS);

    int row = thread_in_group * 4 + (reg ? 16 : 0);
    const uint8_t* src = W13 + ((size_t)le * G1_COLS + n) * H_FIXED + kt * MMA_K + row;
    W13_packed[idx] = pack_e4m3x4(src[0], src[1], src[2], src[3]);
}

__global__ void pack_w2_kernel(
    const uint8_t* __restrict__ W2,
    uint32_t*      __restrict__ W2_packed,
    int E_local)
{
    size_t total = (size_t)E_local * H_FIXED * I_MMA_TILES * 4 * 2;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= total) return;

    int reg = (int)(idx & 1);
    size_t t = idx >> 1;
    int thread_in_group = (int)(t & 3);
    t >>= 2;
    int it = (int)(t % I_MMA_TILES);
    t /= I_MMA_TILES;
    int h = (int)(t % H_FIXED);
    int le = (int)(t / H_FIXED);

    int row = thread_in_group * 4 + (reg ? 16 : 0);
    const uint8_t* src = W2 + ((size_t)le * H_FIXED + h) * I_FIXED + it * MMA_K + row;
    W2_packed[idx] = pack_e4m3x4(src[0], src[1], src[2], src[3]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Routing + compact local dispatch
// ─────────────────────────────────────────────────────────────────────────────
__global__ void route_dispatch_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int*         __restrict__ dispatch_token,
    float*       __restrict__ dispatch_weight,
    int*         __restrict__ expert_counts,
    int*         __restrict__ tile_expert,
    int*         __restrict__ tile_m0,
    int*         __restrict__ tile_count,
    float rsf,
    int T,
    int E_local,
    int local_offset)
{
    int t = (int)blockIdx.x;
    if (t >= T) return;
    int e = (int)threadIdx.x;
    int warp_id = e / WARP_THREADS;
    int lane = e - warp_id * WARP_THREADS;

    __shared__ float sh_sig[E_GLOBAL];
    __shared__ float sh_group_scores[N_GROUP];
    __shared__ int   sh_group_mask[N_GROUP];
    __shared__ float sh_group_best_val[N_GROUP];
    __shared__ int   sh_group_best_idx[N_GROUP];
    __shared__ int   sh_topk_idx[TOP_K];
    __shared__ float sh_topk_sig[TOP_K];

    float sig = 1.0f / (1.0f + __expf(-logits[(size_t)t * E_GLOBAL + e]));
    float biased = sig + bias[e];
    sh_sig[e] = sig;

    float top1_val = biased;
    int top1_idx = e;
    warp_reduce_max_pair(top1_val, top1_idx);
    top1_val = __shfl_sync(0xffffffffu, top1_val, 0);
    top1_idx = __shfl_sync(0xffffffffu, top1_idx, 0);

    float top2_val = (e == top1_idx) ? -1e30f : biased;
    int top2_idx = e;
    warp_reduce_max_pair(top2_val, top2_idx);

    if (lane == 0) {
        sh_group_scores[warp_id] = top1_val + top2_val;
        sh_group_mask[warp_id] = 0;
    }
    __syncthreads();

    if (e == 0) {
        for (int k = 0; k < TOPK_GROUP; ++k) {
            float bv = -1e30f;
            int bg = -1;
            for (int g = 0; g < N_GROUP; ++g) {
                if (!sh_group_mask[g] && sh_group_scores[g] > bv) {
                    bv = sh_group_scores[g];
                    bg = g;
                }
            }
            if (bg >= 0) sh_group_mask[bg] = 1;
        }
    }
    __syncthreads();

    bool selected = false;
    for (int k = 0; k < TOP_K; ++k) {
        float cand_val = (sh_group_mask[warp_id] && !selected) ? biased : -1e30f;
        int cand_idx = e;
        warp_reduce_max_pair(cand_val, cand_idx);

        if (lane == 0) {
            sh_group_best_val[warp_id] = cand_val;
            sh_group_best_idx[warp_id] = cand_idx;
        }
        __syncthreads();

        if (e == 0) {
            float bv = sh_group_best_val[0];
            int be = sh_group_best_idx[0];
            for (int g = 0; g < N_GROUP; ++g) {
                float gv = sh_group_best_val[g];
                int gi = sh_group_best_idx[g];
                if (gv > bv || (gv == bv && gi < be)) {
                    bv = gv;
                    be = gi;
                }
            }
            sh_topk_idx[k] = be;
            sh_topk_sig[k] = sh_sig[be];
        }
        __syncthreads();

        selected = selected || (e == sh_topk_idx[k]);
    }

    if (e == 0) {
        float wsum = 1e-20f;
        for (int k = 0; k < TOP_K; ++k) wsum += sh_topk_sig[k];

        for (int k = 0; k < TOP_K; ++k) {
            int ge = sh_topk_idx[k];
            int le = ge - local_offset;
            if (0 <= le && le < E_local) {
                int slot = atomicAdd(&expert_counts[le], 1);
                if (slot < T) {
                    if ((slot % CTA_M) == 0) {
                        int tile = atomicAdd(tile_count, 1);
                        tile_expert[tile] = le;
                        tile_m0[tile] = slot;
                    }
                    size_t off = (size_t)le * T + slot;
                    dispatch_token[off]  = t;
                    dispatch_weight[off] = (sh_topk_sig[k] / wsum) * rsf;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1: gathered hidden x W13^T, custom FP8 Tensor Core GEMM.
//
// This consumes true FP8 hidden states and GEMM1 weights. Each CTA owns a full
// 128-wide intermediate block, computes gate and up projections, fuses SwiGLU,
// then stores block-scaled FP8 C plus one scale per token row.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void stage1_fp8_mma_kernel(
    const uint8_t* __restrict__ hidden,
    const float*   __restrict__ hidden_scale,
    const uint32_t* __restrict__ W13_packed,
    const float*   __restrict__ W13_scale,
    const int*   __restrict__ dispatch_token,
    const int*   __restrict__ expert_counts,
    const int*   __restrict__ tile_expert,
    const int*   __restrict__ tile_m0,
    const int*   __restrict__ tile_count,
    uint8_t*     __restrict__ C,
    float*       __restrict__ C_scale,
    int T)
{
    int tile_id = (int)blockIdx.y;
    if (tile_id >= tile_count[0]) return;
    int le = tile_expert[tile_id];
    int m0 = tile_m0[tile_id];
    int warp_id = (int)threadIdx.x / WARP_THREADS;
    int lane = (int)threadIdx.x - warp_id * WARP_THREADS;
    int ib = (int)blockIdx.x;
    int block_n = ib * SCALE_BLOCK;
    int group = lane >> 2;
    int thread_in_group = lane & 3;
    int count = expert_counts[le];
    if (m0 >= count) return;
    __shared__ uint8_t sh_a_pipe[2][MMA_A_TILE_ELEMS];
    __shared__ float sh_c[CTA_M * SCALE_BLOCK];
    __shared__ float sh_c_scale[CTA_M];

    float gate_acc[STAGE1_N_FRAGS][MMA_M_FRAGS_PER_WARP][4] = {};
    float value_acc[STAGE1_N_FRAGS][MMA_M_FRAGS_PER_WARP][4] = {};

    for (int kb = 0; kb < H_BLOCKS; ++kb) {
        float raw_gate[STAGE1_N_FRAGS][MMA_M_FRAGS_PER_WARP][4] = {};
        float raw_value[STAGE1_N_FRAGS][MMA_M_FRAGS_PER_WARP][4] = {};

        constexpr int A_CHUNKS = MMA_A_TILE_ELEMS / 16;
        for (int cid = (int)threadIdx.x; cid < A_CHUNKS; cid += (int)blockDim.x) {
            int row = cid / (MMA_K / 16);
            int chunk = cid - row * (MMA_K / 16);
            int col = chunk * 16;
            int slot = m0 + row;
            bool ok = slot < count;
            int token = ok ? dispatch_token[(size_t)le * T + slot] : 0;
            cp_async_16(sh_a_pipe[0] + row * MMA_K + col,
                        hidden + (size_t)token * H_FIXED + kb * SCALE_BLOCK + col,
                        ok ? 16 : 0);
        }
        cp_async_commit();
        cp_async_wait_group<0>();
        __syncthreads();

        for (int kk = 0; kk < SCALE_BLOCK; kk += MMA_K) {
            int k0 = kb * SCALE_BLOCK + kk;
            int cur_stage = (kk / MMA_K) & 1;
            int next_kk = kk + MMA_K;

            if (next_kk < SCALE_BLOCK) {
                int next_stage = cur_stage ^ 1;
                int next_k0 = kb * SCALE_BLOCK + next_kk;
                for (int cid = (int)threadIdx.x; cid < A_CHUNKS; cid += (int)blockDim.x) {
                    int row = cid / (MMA_K / 16);
                    int chunk = cid - row * (MMA_K / 16);
                    int col = chunk * 16;
                    int slot = m0 + row;
                    bool ok = slot < count;
                    int token = ok ? dispatch_token[(size_t)le * T + slot] : 0;
                    cp_async_16(sh_a_pipe[next_stage] + row * MMA_K + col,
                                hidden + (size_t)token * H_FIXED + next_k0 + col,
                                ok ? 16 : 0);
                }
                cp_async_commit();
            }

#pragma unroll
            for (int nf = 0; nf < STAGE1_N_FRAGS; ++nf) {
                int n0 = block_n + nf * CTA_N + warp_id * MMA_N;
                uint32_t b_gate[2];
                uint32_t b_value[2];

#pragma unroll
                for (int reg = 0; reg < 2; ++reg) {
                    int n = n0 + group;
                    uint32_t v = 0;
                    if (n < I_FIXED) {
                        v = W13_packed[
                            w13_packed_offset(le, n, k0 / MMA_K, thread_in_group, reg)];
                    }
                    b_gate[reg] = v;
                }

                for (int reg = 0; reg < 2; ++reg) {
                    int n = n0 + group;
                    uint32_t v = 0;
                    if (n < I_FIXED) {
                        int value_n = I_FIXED + n;
                        v = W13_packed[
                            w13_packed_offset(le, value_n, k0 / MMA_K, thread_in_group, reg)];
                    }
                    b_value[reg] = v;
                }

#pragma unroll
                for (int mf = 0; mf < MMA_M_FRAGS_PER_WARP; ++mf) {
                    uint32_t a_frag[4];
#pragma unroll
                    for (int reg = 0; reg < 4; ++reg) {
                        uint8_t bytes[4];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            int i = reg * 4 + j;
                            int row = ((i < 4) || (8 <= i && i < 12)) ? group : group + 8;
                            int col = thread_in_group * 4 + (i & 3) + ((i >= 8) ? 16 : 0);
                            bytes[j] = sh_a_pipe[cur_stage][(mf * MMA_M + row) * MMA_K + col];
                        }
                        a_frag[reg] = pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
                    }
                    mma_m16n8k32_e4m3(raw_gate[nf][mf], a_frag, b_gate);
                    mma_m16n8k32_e4m3(raw_value[nf][mf], a_frag, b_value);
                }
            }

            if (next_kk < SCALE_BLOCK) {
                cp_async_wait_group<0>();
                __syncthreads();
            }
        }
        __syncthreads();

#pragma unroll
        for (int nf = 0; nf < STAGE1_N_FRAGS; ++nf) {
            int n0 = block_n + nf * CTA_N + warp_id * MMA_N;
#pragma unroll
            for (int mf = 0; mf < MMA_M_FRAGS_PER_WARP; ++mf) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int row = mf * MMA_M + ((i < 2) ? group : group + 8);
                    int col = thread_in_group * 2 + (i & 1);
                    int slot = m0 + row;
                    int n = n0 + col;
                    if (slot < count && n < I_FIXED) {
                        int token = dispatch_token[(size_t)le * T + slot];
                        int nb = n / SCALE_BLOCK;
                        float as = hidden_scale[(size_t)kb * T + token];
                        float gate_scale =
                            W13_scale[((size_t)le * G1_BLOCKS + nb) * H_BLOCKS + kb];
                        float value_scale =
                            W13_scale[((size_t)le * G1_BLOCKS + (I_FIXED / SCALE_BLOCK + nb)) *
                                      H_BLOCKS + kb];
                        gate_acc[nf][mf][i]  += raw_gate[nf][mf][i] * as * gate_scale;
                        value_acc[nf][mf][i] += raw_value[nf][mf][i] * as * value_scale;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int nf = 0; nf < STAGE1_N_FRAGS; ++nf) {
        int n0 = block_n + nf * CTA_N + warp_id * MMA_N;
#pragma unroll
        for (int mf = 0; mf < MMA_M_FRAGS_PER_WARP; ++mf) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = mf * MMA_M + ((i < 2) ? group : group + 8);
                int col = nf * CTA_N + warp_id * MMA_N + thread_in_group * 2 + (i & 1);
                int slot = m0 + row;
                if (slot < count) {
                    float value = value_acc[nf][mf][i];
                    float c = gate_acc[nf][mf][i] * (value / (1.0f + expf(-value)));
                    sh_c[row * SCALE_BLOCK + col] = c;
                }
            }
        }
    }
    __syncthreads();

    if ((int)threadIdx.x < CTA_M) {
        int row = (int)threadIdx.x;
        int slot = m0 + row;
        if (slot < count) {
            float max_abs = 0.0f;
            for (int col = 0; col < SCALE_BLOCK; ++col) {
                max_abs = fmaxf(max_abs, fabsf(sh_c[row * SCALE_BLOCK + col]));
            }
            float scale = fmaxf(max_abs / FP8_E4M3_MAX, 1.0e-8f);
            sh_c_scale[row] = scale;
            C_scale[((size_t)le * T + slot) * I_BLOCKS + ib] = scale;
        } else {
            sh_c_scale[row] = 1.0f;
        }
    }
    __syncthreads();

    for (int idx = (int)threadIdx.x; idx < CTA_M * SCALE_BLOCK; idx += (int)blockDim.x) {
        int row = idx / SCALE_BLOCK;
        int col = idx - row * SCALE_BLOCK;
        int slot = m0 + row;
        if (slot < count) {
            C[((size_t)le * T + slot) * I_FIXED + block_n + col] =
                fp32_to_e4m3_byte(sh_c[idx] / sh_c_scale[row]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2: custom FP8 Tensor Core GEMM + weighted scatter
// ─────────────────────────────────────────────────────────────────────────────
__global__ void stage2_fp8_mma_scatter_kernel(
    const uint8_t*  __restrict__ C,
    const float*    __restrict__ C_scale,
    const uint32_t* __restrict__ W2_packed,
    const float*    __restrict__ W2_scale,
    const int*   __restrict__ dispatch_token,
    const float* __restrict__ dispatch_weight,
    const int*   __restrict__ expert_counts,
    const int*   __restrict__ tile_expert,
    const int*   __restrict__ tile_m0,
    const int*   __restrict__ tile_count,
    float*       __restrict__ output,
    int T)
{
    int tile_id = (int)blockIdx.y;
    if (tile_id >= tile_count[0]) return;
    int le = tile_expert[tile_id];
    int m0 = tile_m0[tile_id];
    int warp_id = (int)threadIdx.x / WARP_THREADS;
    int lane = (int)threadIdx.x - warp_id * WARP_THREADS;
    int h0 = (int)blockIdx.x * CTA_N + warp_id * MMA_N;
    int group = lane >> 2;
    int thread_in_group = lane & 3;
    int count = expert_counts[le];
    if (m0 >= count) return;
    __shared__ uint8_t sh_c_pipe[2][MMA_A_TILE_ELEMS];

    float acc[MMA_M_FRAGS_PER_WARP][4] = {};

    for (int ib = 0; ib < I_BLOCKS; ++ib) {
        float raw_acc[MMA_M_FRAGS_PER_WARP][4] = {};

        constexpr int C_CHUNKS = MMA_A_TILE_ELEMS / 16;
        for (int cid = (int)threadIdx.x; cid < C_CHUNKS; cid += (int)blockDim.x) {
            int row = cid / (MMA_K / 16);
            int chunk = cid - row * (MMA_K / 16);
            int col = chunk * 16;
            int slot = m0 + row;
            bool ok = slot < count;
            cp_async_16(sh_c_pipe[0] + row * MMA_K + col,
                        C + ((size_t)le * T + (ok ? slot : 0)) * I_FIXED + ib * SCALE_BLOCK + col,
                        ok ? 16 : 0);
        }
        cp_async_commit();
        cp_async_wait_group<0>();
        __syncthreads();

        for (int kk = 0; kk < SCALE_BLOCK; kk += MMA_K) {
            int i0 = ib * SCALE_BLOCK + kk;
            int cur_stage = (kk / MMA_K) & 1;
            int next_kk = kk + MMA_K;
            uint32_t b_frag[2];

            if (next_kk < SCALE_BLOCK) {
                int next_stage = cur_stage ^ 1;
                int next_i0 = ib * SCALE_BLOCK + next_kk;
                for (int cid = (int)threadIdx.x; cid < C_CHUNKS; cid += (int)blockDim.x) {
                    int row = cid / (MMA_K / 16);
                    int chunk = cid - row * (MMA_K / 16);
                    int col = chunk * 16;
                    int slot = m0 + row;
                    bool ok = slot < count;
                    cp_async_16(sh_c_pipe[next_stage] + row * MMA_K + col,
                                C + ((size_t)le * T + (ok ? slot : 0)) * I_FIXED + next_i0 + col,
                                ok ? 16 : 0);
                }
                cp_async_commit();
            }

#pragma unroll
            for (int reg = 0; reg < 2; ++reg) {
                int h = h0 + group;
                uint32_t v = 0;
                if (h < H_FIXED) {
                    v = W2_packed[
                        w2_packed_offset(le, h, i0 / MMA_K, thread_in_group, reg)];
                }
                b_frag[reg] = v;
            }

#pragma unroll
            for (int mf = 0; mf < MMA_M_FRAGS_PER_WARP; ++mf) {
                uint32_t a_frag[4];
#pragma unroll
                for (int reg = 0; reg < 4; ++reg) {
                    uint8_t bytes[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int i = reg * 4 + j;
                        int row = ((i < 4) || (8 <= i && i < 12)) ? group : group + 8;
                        int col = thread_in_group * 4 + (i & 3) + ((i >= 8) ? 16 : 0);
                        bytes[j] = sh_c_pipe[cur_stage][(mf * MMA_M + row) * MMA_K + col];
                    }
                    a_frag[reg] = pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
                }
                mma_m16n8k32_e4m3(raw_acc[mf], a_frag, b_frag);
            }

            if (next_kk < SCALE_BLOCK) {
                cp_async_wait_group<0>();
                __syncthreads();
            }
        }
        __syncthreads();

#pragma unroll
        for (int mf = 0; mf < MMA_M_FRAGS_PER_WARP; ++mf) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int col = thread_in_group * 2 + (i & 1);
                int h = h0 + col;
                int row = mf * MMA_M + ((i < 2) ? group : group + 8);
                int slot = m0 + row;
                if (slot < count && h < H_FIXED) {
                    int hb = h / SCALE_BLOCK;
                    float cs = C_scale[((size_t)le * T + slot) * I_BLOCKS + ib];
                    float ws = W2_scale[((size_t)le * H_BLOCKS + hb) * I_BLOCKS + ib];
                    acc[mf][i] += raw_acc[mf][i] * cs * ws;
                }
            }
        }
    }

#pragma unroll
    for (int mf = 0; mf < MMA_M_FRAGS_PER_WARP; ++mf) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = mf * MMA_M + ((i < 2) ? group : group + 8);
            int col = thread_in_group * 2 + (i & 1);
            int slot = m0 + row;
            int h = h0 + col;
            if (slot < count && h < H_FIXED) {
                size_t qoff = (size_t)le * T + slot;
                int token = dispatch_token[qoff];
                float w = dispatch_weight[qoff];
                atomicAdd(&output[(size_t)token * H_FIXED + h], acc[mf][i] * w);
            }
        }
    }
}

__global__ void bf16_convert_kernel(
    const float* __restrict__ src,
    uint16_t*    __restrict__ dst,
    int n)
{
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) dst[i] = fp32_to_bf16_bits(src[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor pack_w13(torch::Tensor W13)
{
    const at::cuda::CUDAGuard guard(W13.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(W13.dim() == 3, "W13 must be [E_local, 4096, 7168]");
    TORCH_CHECK(W13.element_size() == 1, "W13 must be an FP8 tensor");
    TORCH_CHECK(W13.is_contiguous(), "W13 must be contiguous");
    int E_local = (int)W13.size(0);
    TORCH_CHECK(W13.size(1) == G1_COLS && W13.size(2) == H_FIXED,
                "W13 must be [E_local, 4096, 7168]");

    auto packed = torch::empty({E_local, G1_COLS, H_MMA_TILES, 4, 2},
                               W13.options().dtype(torch::kInt32));
    size_t total = (size_t)E_local * G1_COLS * H_MMA_TILES * 4 * 2;
    pack_w13_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W13.data_ptr()),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        E_local);
    return packed;
}

torch::Tensor pack_w2(torch::Tensor W2)
{
    const at::cuda::CUDAGuard guard(W2.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(W2.dim() == 3, "W2 must be [E_local, 7168, 2048]");
    TORCH_CHECK(W2.element_size() == 1, "W2 must be an FP8 tensor");
    TORCH_CHECK(W2.is_contiguous(), "W2 must be contiguous");
    int E_local = (int)W2.size(0);
    TORCH_CHECK(W2.size(1) == H_FIXED && W2.size(2) == I_FIXED,
                "W2 must be [E_local, 7168, 2048]");

    auto packed = torch::empty({E_local, H_FIXED, I_MMA_TILES, 4, 2},
                               W2.options().dtype(torch::kInt32));
    size_t total = (size_t)E_local * H_FIXED * I_MMA_TILES * 4 * 2;
    pack_w2_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W2.data_ptr()),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        E_local);
    return packed;
}

torch::Tensor moe_forward(
    torch::Tensor hidden,
    torch::Tensor hidden_scale,
    torch::Tensor W13_packed,
    torch::Tensor W13_scale,
    torch::Tensor W2_packed,
    torch::Tensor W2_scale,
    torch::Tensor logits,
    torch::Tensor bias,
    float rsf,
    int local_offset)
{
    const at::cuda::CUDAGuard guard(hidden.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(hidden.dim() == 2, "hidden must be [T, 7168]");
    TORCH_CHECK(hidden_scale.dim() == 2, "hidden_scale must be [56, T]");
    TORCH_CHECK(W13_packed.dim() == 5,
                "W13_packed must be [E_local, 4096, 224, 4, 2]");
    TORCH_CHECK(W13_scale.dim() == 3, "W13_scale must be [E_local, 32, 56]");
    TORCH_CHECK(W2_packed.dim() == 5,
                "W2_packed must be [E_local, 7168, 64, 4, 2]");
    TORCH_CHECK(W2_scale.dim() == 3, "W2_scale must be [E_local, 56, 16]");
    TORCH_CHECK(logits.dim() == 2, "logits must be [T, 256]");
    TORCH_CHECK(hidden.element_size() == 1, "hidden must be an FP8 tensor");
    TORCH_CHECK(hidden_scale.scalar_type() == torch::kFloat32,
                "hidden_scale must be float32");
    TORCH_CHECK(W13_packed.scalar_type() == torch::kInt32,
                "W13_packed must be int32-packed FP8 fragments");
    TORCH_CHECK(W13_scale.scalar_type() == torch::kFloat32,
                "W13_scale must be float32");
    TORCH_CHECK(W2_packed.scalar_type() == torch::kInt32,
                "W2_packed must be int32-packed FP8 fragments");
    TORCH_CHECK(W2_scale.scalar_type() == torch::kFloat32,
                "W2_scale must be float32");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    int T       = (int)hidden.size(0);
    int H_dim   = (int)hidden.size(1);
    int E_local = (int)W13_packed.size(0);

    TORCH_CHECK(H_dim == H_FIXED, "hidden must have hidden dim 7168");
    TORCH_CHECK(hidden_scale.size(0) == H_BLOCKS && hidden_scale.size(1) == T,
                "hidden_scale must be [56, T]");
    TORCH_CHECK(logits.size(1) == E_GLOBAL, "logits must be [T, 256]");
    TORCH_CHECK(bias.numel() == E_GLOBAL, "bias must have 256 elements");
    TORCH_CHECK(W13_packed.size(1) == G1_COLS &&
                W13_packed.size(2) == H_MMA_TILES &&
                W13_packed.size(3) == 4 &&
                W13_packed.size(4) == 2,
                "W13_packed must be [E_local, 4096, 224, 4, 2]");
    TORCH_CHECK(W13_scale.size(0) == E_local &&
                W13_scale.size(1) == G1_BLOCKS &&
                W13_scale.size(2) == H_BLOCKS,
                "W13_scale must be [E_local, 32, 56]");
    TORCH_CHECK(W2_packed.size(0) == E_local &&
                W2_packed.size(1) == H_FIXED &&
                W2_packed.size(2) == I_MMA_TILES &&
                W2_packed.size(3) == 4 &&
                W2_packed.size(4) == 2,
                "W2_packed must be [E_local, 7168, 64, 4, 2]");
    TORCH_CHECK(W2_scale.size(0) == E_local &&
                W2_scale.size(1) == H_BLOCKS &&
                W2_scale.size(2) == I_BLOCKS,
                "W2_scale must be [E_local, 56, 16]");
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(hidden_scale.is_contiguous(), "hidden_scale must be contiguous");
    TORCH_CHECK(W13_packed.is_contiguous(), "W13_packed must be contiguous");
    TORCH_CHECK(W13_scale.is_contiguous(), "W13_scale must be contiguous");
    TORCH_CHECK(W2_packed.is_contiguous(), "W2_packed must be contiguous");
    TORCH_CHECK(W2_scale.is_contiguous(), "W2_scale must be contiguous");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    auto int_opts = torch::TensorOptions().device(hidden.device()).dtype(torch::kInt32);
    auto fp_opts  = torch::TensorOptions().device(hidden.device()).dtype(torch::kFloat32);

    auto expert_counts   = torch::empty({E_local}, int_opts);
    auto dispatch_token  = torch::empty({E_local, T}, int_opts);
    auto dispatch_weight = torch::empty({E_local, T}, fp_opts);
    auto C               = torch::empty({E_local, T, I_FIXED},
                                        hidden.options().dtype(torch::kUInt8));
    auto C_scale         = torch::empty({E_local, T, I_BLOCKS}, fp_opts);
    auto output_fp32     = torch::empty({T, H_FIXED}, fp_opts);
    auto output_bf16     = torch::empty({T, H_FIXED}, hidden.options().dtype(torch::kBFloat16));
    int max_active_tiles = E_local * ((T + CTA_M - 1) / CTA_M);
    auto tile_expert     = torch::empty({max_active_tiles}, int_opts);
    auto tile_m0         = torch::empty({max_active_tiles}, int_opts);
    auto tile_count      = torch::empty({1}, int_opts);

    C10_CUDA_CHECK(cudaMemsetAsync(expert_counts.data_ptr<int>(), 0,
                                   (size_t)E_local * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(output_fp32.data_ptr<float>(), 0,
                                   (size_t)T * H_FIXED * sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(tile_count.data_ptr<int>(), 0,
                                   sizeof(int), stream));

    route_dispatch_kernel<<<T, E_GLOBAL, 0, stream>>>(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        dispatch_token.data_ptr<int>(),
        dispatch_weight.data_ptr<float>(),
        expert_counts.data_ptr<int>(),
        tile_expert.data_ptr<int>(),
        tile_m0.data_ptr<int>(),
        tile_count.data_ptr<int>(),
        rsf,
        T,
        E_local,
        local_offset);

    dim3 block(MMA_THREADS);
    dim3 grid_stage1(I_BLOCKS, max_active_tiles, 1);
    if (max_active_tiles > 0) {
        stage1_fp8_mma_kernel<<<grid_stage1, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(hidden.data_ptr()),
            hidden_scale.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(W13_packed.data_ptr<int32_t>()),
            W13_scale.data_ptr<float>(),
            dispatch_token.data_ptr<int>(),
            expert_counts.data_ptr<int>(),
            tile_expert.data_ptr<int>(),
            tile_m0.data_ptr<int>(),
            tile_count.data_ptr<int>(),
            C.data_ptr<uint8_t>(),
            C_scale.data_ptr<float>(),
            T);
    }

    dim3 grid_stage2((H_FIXED + CTA_N - 1) / CTA_N,
                     max_active_tiles,
                     1);
    if (max_active_tiles > 0) {
        stage2_fp8_mma_scatter_kernel<<<grid_stage2, block, 0, stream>>>(
            C.data_ptr<uint8_t>(),
            C_scale.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(W2_packed.data_ptr<int32_t>()),
            W2_scale.data_ptr<float>(),
            dispatch_token.data_ptr<int>(),
            dispatch_weight.data_ptr<float>(),
            expert_counts.data_ptr<int>(),
            tile_expert.data_ptr<int>(),
            tile_m0.data_ptr<int>(),
            tile_count.data_ptr<int>(),
            output_fp32.data_ptr<float>(),
            T);
    }

    int n = T * H_FIXED;
    bf16_convert_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        output_fp32.data_ptr<float>(),
        reinterpret_cast<uint16_t*>(output_bf16.data_ptr<at::BFloat16>()),
        n);

    return output_bf16;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_w13", &pack_w13,
          "Pack W13 FP8 bytes into mma.m16n8k32 B fragments");
    m.def("pack_w2", &pack_w2,
          "Pack W2 FP8 bytes into mma.m16n8k32 B fragments");
    m.def("moe_forward", &moe_forward,
          "Custom FP8 Tensor Core MoE forward without cuBLAS");
}

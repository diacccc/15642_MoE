/*
 * Custom MoE forward pass for the fixed DeepSeek-V3/R1 geometry.
 *
 * This version deliberately avoids cuBLAS. It keeps the complete routed MoE
 * path on custom CUDA kernels:
 *
 *   1. route_dispatch_kernel
 *        sigmoid -> group top-k -> global top-k -> normalized weights, then
 *        compact local (token, expert, weight) work into per-expert queues.
 *
 *   2. stage1_fp8_mma_kernel
 *        custom FP8 Tensor Core GEMM over true FP8 inputs:
 *        hidden[token, H] x W13[expert, 2I, H]^T, then fused SwiGLU
 *        into a compact bf16 C[expert, slot, I] intermediate.
 *
 *   3. build_active_tiles_kernel
 *        converts per-expert token counts into a compact list of non-empty
 *        (expert, token-tile) work. This avoids launching Tensor Core CTAs for
 *        empty expert tiles when T is large.
 *
 *   4. stage2_fp8_mma_scatter_kernel
 *        custom FP8 Tensor Core GEMM with fused epilogue:
 *        C x W2[expert, H, I]^T over true FP8 W2 bytes with block scales,
 *        then weighted atomic accumulation directly into the final token output.
 *
 *   5. bf16_convert_kernel
 *        converts the float accumulator to the public bf16 result.
 *
 * Only moe_forward() is exported to Python.
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

// ─────────────────────────────────────────────────────────────────────────────
// Routing + compact local dispatch
// ─────────────────────────────────────────────────────────────────────────────
__global__ void route_dispatch_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int*         __restrict__ dispatch_token,
    float*       __restrict__ dispatch_weight,
    int*         __restrict__ expert_counts,
    float rsf,
    int T,
    int E_local,
    int local_offset)
{
    int t = (int)blockIdx.x;
    if (t >= T) return;
    int e = (int)threadIdx.x;

    __shared__ float sh_sig[E_GLOBAL];
    __shared__ float sh_sig_biased[E_GLOBAL];
    __shared__ float sh_group_scores[N_GROUP];
    __shared__ int   sh_group_mask[N_GROUP];
    __shared__ float sh_masked[E_GLOBAL];
    __shared__ bool  sh_selected[E_GLOBAL];
    __shared__ int   sh_topk_idx[TOP_K];
    __shared__ float sh_topk_sig[TOP_K];

    float sig        = 1.0f / (1.0f + expf(-logits[(size_t)t * E_GLOBAL + e]));
    sh_sig[e]        = sig;
    sh_sig_biased[e] = sig + bias[e];
    sh_selected[e]   = false;
    __syncthreads();

    if (e < N_GROUP) {
        int base = e * GROUP_SIZE;
        float g1 = -1e30f;
        float g2 = -1e30f;
        for (int i = base; i < base + GROUP_SIZE; ++i) {
            float v = sh_sig_biased[i];
            if (v > g1) {
                g2 = g1;
                g1 = v;
            } else if (v > g2) {
                g2 = v;
            }
        }
        sh_group_scores[e] = g1 + g2;
        sh_group_mask[e]   = 0;
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

    sh_masked[e] = sh_group_mask[e / GROUP_SIZE] ? sh_sig_biased[e] : -1e30f;
    __syncthreads();

    if (e == 0) {
        for (int k = 0; k < TOP_K; ++k) {
            float bv = -1e30f;
            int be = -1;
            for (int i = 0; i < E_GLOBAL; ++i) {
                if (!sh_selected[i] && sh_masked[i] > bv) {
                    bv = sh_masked[i];
                    be = i;
                }
            }
            sh_topk_idx[k] = be;
            sh_topk_sig[k] = (be >= 0) ? sh_sig[be] : 0.0f;
            if (be >= 0) sh_selected[be] = true;
        }

        float wsum = 1e-20f;
        for (int k = 0; k < TOP_K; ++k) wsum += sh_topk_sig[k];

        for (int k = 0; k < TOP_K; ++k) {
            int ge = sh_topk_idx[k];
            int le = ge - local_offset;
            if (0 <= le && le < E_local) {
                int slot = atomicAdd(&expert_counts[le], 1);
                if (slot < T) {
                    size_t off = (size_t)le * T + slot;
                    dispatch_token[off]  = t;
                    dispatch_weight[off] = (sh_topk_sig[k] / wsum) * rsf;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Build compact list of active (expert, M tile) work.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void build_active_tiles_kernel(
    const int* __restrict__ expert_counts,
    int*       __restrict__ tile_expert,
    int*       __restrict__ tile_m0,
    int*       __restrict__ tile_count,
    int E_local)
{
    int le = (int)blockIdx.x;
    if (le >= E_local) return;

    int count = expert_counts[le];
    int tiles = (count + MMA_M - 1) / MMA_M;
    for (int tile = (int)threadIdx.x; tile < tiles; tile += (int)blockDim.x) {
        int out = atomicAdd(tile_count, 1);
        tile_expert[out] = le;
        tile_m0[out] = tile * MMA_M;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1: gathered hidden x W13^T, custom FP8 Tensor Core GEMM.
//
// This consumes true FP8 hidden states and GEMM1 weights. The raw FP8 dot
// product is accumulated over each 128-wide scale block for gate and up
// projections together, then Stage 1 writes only bf16 SwiGLU output C.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void stage1_fp8_mma_kernel(
    const uint8_t* __restrict__ hidden,
    const float*   __restrict__ hidden_scale,
    const uint8_t* __restrict__ W13,
    const float*   __restrict__ W13_scale,
    const int*   __restrict__ dispatch_token,
    const int*   __restrict__ expert_counts,
    const int*   __restrict__ tile_expert,
    const int*   __restrict__ tile_m0,
    const int*   __restrict__ tile_count,
    uint16_t*    __restrict__ C,
    int T)
{
    int tile_id = (int)blockIdx.y;
    if (tile_id >= tile_count[0]) return;
    int le = tile_expert[tile_id];
    int m0 = tile_m0[tile_id];
    int warp_id = (int)threadIdx.x / WARP_THREADS;
    int lane = (int)threadIdx.x - warp_id * WARP_THREADS;
    int n0 = ((int)blockIdx.x * MMA_WARPS_PER_CTA + warp_id) * MMA_N;
    int group = lane >> 2;
    int thread_in_group = lane & 3;
    int count = expert_counts[le];
    if (m0 >= count || n0 >= I_FIXED) return;

    float gate_acc[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float value_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < H_BLOCKS; ++kb) {
        float raw_gate[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
        float raw_value[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int kk = 0; kk < SCALE_BLOCK; kk += MMA_K) {
            int k0 = kb * SCALE_BLOCK + kk;
            uint32_t a_frag[4];
            uint32_t b_gate[2];
            uint32_t b_value[2];

#pragma unroll
            for (int reg = 0; reg < 4; ++reg) {
                uint8_t bytes[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int i = reg * 4 + j;
                    int row = ((i < 4) || (8 <= i && i < 12)) ? group : group + 8;
                    int col = thread_in_group * 4 + (i & 3) + ((i >= 8) ? 16 : 0);
                    int slot = m0 + row;
                    uint8_t v = 0;
                    if (slot < count) {
                        int token = dispatch_token[(size_t)le * T + slot];
                        v = hidden[(size_t)token * H_FIXED + k0 + col];
                    }
                    bytes[j] = v;
                }
                a_frag[reg] = pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
            }

#pragma unroll
            for (int reg = 0; reg < 2; ++reg) {
                uint8_t bytes[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int i = reg * 4 + j;
                    int row = thread_in_group * 4 + (i & 3) + ((i >= 4) ? 16 : 0);
                    int col = group;
                    int n = n0 + col;
                    uint8_t v = 0;
                    if (n < I_FIXED) {
                        v = W13[((size_t)le * G1_COLS + n) * H_FIXED + k0 + row];
                    }
                    bytes[j] = v;
                }
                b_gate[reg] = pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
            }

            for (int reg = 0; reg < 2; ++reg) {
                uint8_t bytes[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int i = reg * 4 + j;
                    int row = thread_in_group * 4 + (i & 3) + ((i >= 4) ? 16 : 0);
                    int col = group;
                    int n = n0 + col;
                    uint8_t v = 0;
                    if (n < I_FIXED) {
                        int value_n = I_FIXED + n;
                        v = W13[((size_t)le * G1_COLS + value_n) * H_FIXED + k0 + row];
                    }
                    bytes[j] = v;
                }
                b_value[reg] = pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
            }

            mma_m16n8k32_e4m3(raw_gate, a_frag, b_gate);
            mma_m16n8k32_e4m3(raw_value, a_frag, b_value);
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = (i < 2) ? group : group + 8;
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
                gate_acc[i]  += raw_gate[i] * as * gate_scale;
                value_acc[i] += raw_value[i] * as * value_scale;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = (i < 2) ? group : group + 8;
        int col = thread_in_group * 2 + (i & 1);
        int slot = m0 + row;
        int n = n0 + col;
        if (slot < count && n < I_FIXED) {
            float value = value_acc[i];
            float c = gate_acc[i] * (value / (1.0f + expf(-value)));
            C[((size_t)le * T + slot) * I_FIXED + n] = fp32_to_bf16_bits(c);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2: custom FP8 Tensor Core GEMM + weighted scatter
// ─────────────────────────────────────────────────────────────────────────────
__global__ void stage2_fp8_mma_scatter_kernel(
    const uint16_t* __restrict__ C,
    const uint8_t*  __restrict__ W2,
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
    int h0 = ((int)blockIdx.x * MMA_WARPS_PER_CTA + warp_id) * MMA_N;
    int group = lane >> 2;
    int thread_in_group = lane & 3;
    int count = expert_counts[le];
    if (m0 >= count || h0 >= H_FIXED) return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int ib = 0; ib < I_BLOCKS; ++ib) {
        float raw_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int kk = 0; kk < SCALE_BLOCK; kk += MMA_K) {
            int i0 = ib * SCALE_BLOCK + kk;
            uint32_t a_frag[4];
            uint32_t b_frag[2];

#pragma unroll
            for (int reg = 0; reg < 4; ++reg) {
                uint8_t bytes[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int i = reg * 4 + j;
                    int row = ((i < 4) || (8 <= i && i < 12)) ? group : group + 8;
                    int col = thread_in_group * 4 + (i & 3) + ((i >= 8) ? 16 : 0);
                    int slot = m0 + row;
                    int inner = i0 + col;
                    float v = 0.0f;
                    if (slot < count) {
                        v = bf16_bits_to_fp32(C[((size_t)le * T + slot) * I_FIXED + inner]);
                    }
                    bytes[j] = fp32_to_e4m3_byte(v);
                }
                a_frag[reg] = pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
            }

#pragma unroll
            for (int reg = 0; reg < 2; ++reg) {
                uint8_t bytes[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int i = reg * 4 + j;
                    int row = thread_in_group * 4 + (i & 3) + ((i >= 4) ? 16 : 0);
                    int col = group;
                    int h = h0 + col;
                    uint8_t v = 0;
                    if (h < H_FIXED) {
                        v = W2[((size_t)le * H_FIXED + h) * I_FIXED + i0 + row];
                    }
                    bytes[j] = v;
                }
                b_frag[reg] = pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
            }

            mma_m16n8k32_e4m3(raw_acc, a_frag, b_frag);
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int col = thread_in_group * 2 + (i & 1);
            int h = h0 + col;
            if (h < H_FIXED) {
                int hb = h / SCALE_BLOCK;
                float ws = W2_scale[((size_t)le * H_BLOCKS + hb) * I_BLOCKS + ib];
                acc[i] += raw_acc[i] * ws;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = (i < 2) ? group : group + 8;
        int col = thread_in_group * 2 + (i & 1);
        int slot = m0 + row;
        int h = h0 + col;
        if (slot < count && h < H_FIXED) {
            size_t qoff = (size_t)le * T + slot;
            int token = dispatch_token[qoff];
            float w = dispatch_weight[qoff];
            atomicAdd(&output[(size_t)token * H_FIXED + h], acc[i] * w);
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
torch::Tensor moe_forward(
    torch::Tensor hidden,
    torch::Tensor hidden_scale,
    torch::Tensor W13_fp8,
    torch::Tensor W13_scale,
    torch::Tensor W2_fp8,
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
    TORCH_CHECK(W13_fp8.dim() == 3, "W13 must be [E_local, 4096, 7168]");
    TORCH_CHECK(W13_scale.dim() == 3, "W13_scale must be [E_local, 32, 56]");
    TORCH_CHECK(W2_fp8.dim() == 3, "W2 must be [E_local, 7168, 2048]");
    TORCH_CHECK(W2_scale.dim() == 3, "W2_scale must be [E_local, 56, 16]");
    TORCH_CHECK(logits.dim() == 2, "logits must be [T, 256]");
    TORCH_CHECK(hidden.element_size() == 1, "hidden must be an FP8 tensor");
    TORCH_CHECK(hidden_scale.scalar_type() == torch::kFloat32,
                "hidden_scale must be float32");
    TORCH_CHECK(W13_fp8.element_size() == 1, "W13 must be an FP8 tensor");
    TORCH_CHECK(W13_scale.scalar_type() == torch::kFloat32,
                "W13_scale must be float32");
    TORCH_CHECK(W2_fp8.element_size() == 1, "W2 must be an FP8 tensor");
    TORCH_CHECK(W2_scale.scalar_type() == torch::kFloat32,
                "W2_scale must be float32");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    int T       = (int)hidden.size(0);
    int H_dim   = (int)hidden.size(1);
    int E_local = (int)W13_fp8.size(0);

    TORCH_CHECK(H_dim == H_FIXED, "hidden must have hidden dim 7168");
    TORCH_CHECK(hidden_scale.size(0) == H_BLOCKS && hidden_scale.size(1) == T,
                "hidden_scale must be [56, T]");
    TORCH_CHECK(logits.size(1) == E_GLOBAL, "logits must be [T, 256]");
    TORCH_CHECK(bias.numel() == E_GLOBAL, "bias must have 256 elements");
    TORCH_CHECK(W13_fp8.size(1) == G1_COLS && W13_fp8.size(2) == H_FIXED,
                "W13 must be [E_local, 4096, 7168]");
    TORCH_CHECK(W13_scale.size(0) == E_local &&
                W13_scale.size(1) == G1_BLOCKS &&
                W13_scale.size(2) == H_BLOCKS,
                "W13_scale must be [E_local, 32, 56]");
    TORCH_CHECK(W2_fp8.size(0) == E_local &&
                W2_fp8.size(1) == H_FIXED &&
                W2_fp8.size(2) == I_FIXED,
                "W2 must be [E_local, 7168, 2048]");
    TORCH_CHECK(W2_scale.size(0) == E_local &&
                W2_scale.size(1) == H_BLOCKS &&
                W2_scale.size(2) == I_BLOCKS,
                "W2_scale must be [E_local, 56, 16]");
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(hidden_scale.is_contiguous(), "hidden_scale must be contiguous");
    TORCH_CHECK(W13_fp8.is_contiguous(), "W13 must be contiguous");
    TORCH_CHECK(W13_scale.is_contiguous(), "W13_scale must be contiguous");
    TORCH_CHECK(W2_fp8.is_contiguous(), "W2 must be contiguous");
    TORCH_CHECK(W2_scale.is_contiguous(), "W2_scale must be contiguous");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    auto int_opts = torch::TensorOptions().device(hidden.device()).dtype(torch::kInt32);
    auto fp_opts  = torch::TensorOptions().device(hidden.device()).dtype(torch::kFloat32);

    auto expert_counts   = torch::empty({E_local}, int_opts);
    auto dispatch_token  = torch::empty({E_local, T}, int_opts);
    auto dispatch_weight = torch::empty({E_local, T}, fp_opts);
    auto C               = torch::empty({E_local, T, I_FIXED},
                                        hidden.options().dtype(torch::kBFloat16));
    auto output_fp32     = torch::empty({T, H_FIXED}, fp_opts);
    auto output_bf16     = torch::empty({T, H_FIXED}, hidden.options().dtype(torch::kBFloat16));
    int max_active_tiles = E_local * ((T + MMA_M - 1) / MMA_M);
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
        rsf,
        T,
        E_local,
        local_offset);

    build_active_tiles_kernel<<<E_local, 128, 0, stream>>>(
        expert_counts.data_ptr<int>(),
        tile_expert.data_ptr<int>(),
        tile_m0.data_ptr<int>(),
        tile_count.data_ptr<int>(),
        E_local);

    dim3 block(MMA_THREADS);
    dim3 grid_stage1((I_FIXED + MMA_N * MMA_WARPS_PER_CTA - 1) /
                         (MMA_N * MMA_WARPS_PER_CTA),
                     max_active_tiles,
                     1);
    if (max_active_tiles > 0) {
        stage1_fp8_mma_kernel<<<grid_stage1, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(hidden.data_ptr()),
            hidden_scale.data_ptr<float>(),
            reinterpret_cast<const uint8_t*>(W13_fp8.data_ptr()),
            W13_scale.data_ptr<float>(),
            dispatch_token.data_ptr<int>(),
            expert_counts.data_ptr<int>(),
            tile_expert.data_ptr<int>(),
            tile_m0.data_ptr<int>(),
            tile_count.data_ptr<int>(),
            reinterpret_cast<uint16_t*>(C.data_ptr<at::BFloat16>()),
            T);
    }

    dim3 grid_stage2((H_FIXED + MMA_N * MMA_WARPS_PER_CTA - 1) /
                         (MMA_N * MMA_WARPS_PER_CTA),
                     max_active_tiles,
                     1);
    if (max_active_tiles > 0) {
        stage2_fp8_mma_scatter_kernel<<<grid_stage2, block, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(C.data_ptr<at::BFloat16>()),
            reinterpret_cast<const uint8_t*>(W2_fp8.data_ptr()),
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
    m.def("moe_forward", &moe_forward,
          "Custom FP8 Tensor Core MoE forward without cuBLAS");
}

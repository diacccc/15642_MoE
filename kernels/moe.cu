/*
 * Custom MoE kernels for DeepSeek-V3/R1 geometry.
 *
 * Three fused kernels replace the scattered PyTorch ops in the reference:
 *
 *   1. fused_routing_kernel   — sigmoid → group scoring → group top-k →
 *                               global top-k → weight normalisation, all
 *                               in a single kernel launch per batch.
 *
 *   2. swiglu_kernel          — split + SiLU(X2)*X1 in one pass over memory
 *                               (avoids storing the full [N, 2*I] intermediate
 *                               twice).
 *
 *   3. weighted_scatter_accum_kernel — scale each expert-output row by its
 *                               routing weight and atomically accumulate into
 *                               the output buffer, replacing index_add_ calls.
 *
 * Build via torch.utils.cpp_extension.load().
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ── Fixed DeepSeek-V3/R1 routing geometry ────────────────────────────────────
static constexpr int E_GLOBAL   = 256;
static constexpr int N_GROUP    = 8;
static constexpr int GROUP_SIZE = E_GLOBAL / N_GROUP;  // 32
static constexpr int TOP_K      = 8;
static constexpr int TOPK_GROUP = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Fused routing
//
// Grid : (T,) – one block per token
// Block: E_GLOBAL=256 threads – one thread per expert
//
// Steps executed collaboratively:
//   A) All 256 threads: sigmoid(logit) + bias  → sh_sig, sh_sig_biased
//   B) Threads 0..7:    top-2-sum per group    → sh_group_scores
//   C) Thread 0:        greedy top-TOPK_GROUP  → sh_group_mask
//   D) All 256 threads: mask out pruned groups → sh_masked
//   E) Thread 0:        greedy top-TOP_K       → sh_topk_idx, sh_topk_sig
//   F) Thread 0:        normalise weights, write output
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_routing_kernel(
    const float* __restrict__ logits,        // [T, E_GLOBAL]
    const float* __restrict__ bias,          // [E_GLOBAL]
    int*         __restrict__ out_topk_idx,  // [T, TOP_K]
    float*       __restrict__ out_topk_w,    // [T, TOP_K]
    float        routed_scaling_factor,
    int          T)
{
    int t = (int)blockIdx.x;
    if (t >= T) return;
    int e = (int)threadIdx.x;  // 0 .. E_GLOBAL-1

    __shared__ float sh_sig[E_GLOBAL];        // sigmoid(logit)
    __shared__ float sh_sig_biased[E_GLOBAL]; // sigmoid(logit) + bias
    __shared__ float sh_group_scores[N_GROUP];
    __shared__ int   sh_group_mask[N_GROUP];
    __shared__ float sh_masked[E_GLOBAL];     // pruned biased scores
    __shared__ bool  sh_selected[E_GLOBAL];   // top-k selection state
    __shared__ int   sh_topk_idx[TOP_K];
    __shared__ float sh_topk_sig[TOP_K];      // sig (no bias) for weighting

    // ── A: sigmoid ───────────────────────────────────────────────────────────
    float logit = logits[(size_t)t * E_GLOBAL + e];
    float sig   = 1.0f / (1.0f + expf(-logit));
    sh_sig[e]        = sig;
    sh_sig_biased[e] = sig + bias[e];
    sh_selected[e]   = false;
    __syncthreads();

    // ── B: group scores (threads 0..N_GROUP-1, each owns one group) ──────────
    if (e < N_GROUP) {
        int base  = e * GROUP_SIZE;
        float t1  = -1e30f, t2 = -1e30f;
        for (int i = base; i < base + GROUP_SIZE; ++i) {
            float v = sh_sig_biased[i];
            if      (v > t1) { t2 = t1; t1 = v; }
            else if (v > t2) { t2 = v; }
        }
        sh_group_scores[e] = t1 + t2;
        sh_group_mask[e]   = 0;
    }
    __syncthreads();

    // ── C: select top-TOPK_GROUP groups (thread 0, O(N_GROUP*TOPK_GROUP)) ────
    if (e == 0) {
        for (int k = 0; k < TOPK_GROUP; ++k) {
            float best_v = -1e30f;
            int   best_g = -1;
            for (int g = 0; g < N_GROUP; ++g) {
                if (!sh_group_mask[g] && sh_group_scores[g] > best_v) {
                    best_v = sh_group_scores[g];
                    best_g = g;
                }
            }
            if (best_g >= 0) sh_group_mask[best_g] = 1;
        }
    }
    __syncthreads();

    // ── D: mask out non-selected groups ──────────────────────────────────────
    sh_masked[e] = sh_group_mask[e / GROUP_SIZE] ? sh_sig_biased[e] : -1e30f;
    __syncthreads();

    // ── E: global top-k among kept groups (thread 0, O(E_GLOBAL*TOP_K)) ─────
    if (e == 0) {
        for (int k = 0; k < TOP_K; ++k) {
            float best_v = -1e30f;
            int   best_e = -1;
            for (int i = 0; i < E_GLOBAL; ++i) {
                if (!sh_selected[i] && sh_masked[i] > best_v) {
                    best_v = sh_masked[i];
                    best_e = i;
                }
            }
            sh_topk_idx[k] = best_e;
            sh_topk_sig[k] = (best_e >= 0) ? sh_sig[best_e] : 0.0f;
            if (best_e >= 0) sh_selected[best_e] = true;
        }
    }
    __syncthreads();

    // ── F: normalise and write (thread 0) ────────────────────────────────────
    if (e == 0) {
        float wsum = 1e-20f;
        for (int k = 0; k < TOP_K; ++k) wsum += sh_topk_sig[k];

        int*   oi = out_topk_idx + (size_t)t * TOP_K;
        float* ow = out_topk_w   + (size_t)t * TOP_K;
        for (int k = 0; k < TOP_K; ++k) {
            oi[k] = sh_topk_idx[k];
            ow[k] = (sh_topk_sig[k] / wsum) * routed_scaling_factor;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Fused SwiGLU
//
// Grid : (N, ceil(I / BLOCK_COLS))
// Block: (BLOCK_COLS,)
//
// Each thread reads gate=input[row, col] and val=input[row, I+col],
// computes silu(val)*gate, writes output[row, col].
// One read pass over the 2*I input (vs. two passes for split-then-apply).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swiglu_kernel(
    const float* __restrict__ input,   // [N, 2*I]
    float*       __restrict__ output,  // [N, I]
    int N, int I)
{
    int row = (int)blockIdx.x;
    int col = (int)(blockIdx.y * blockDim.x) + (int)threadIdx.x;
    if (row >= N || col >= I) return;

    float gate  = input[(size_t)row * 2 * I + col];
    float value = input[(size_t)row * 2 * I + I + col];
    float silu  = value / (1.0f + expf(-value));
    output[(size_t)row * I + col] = silu * gate;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: Weighted scatter accumulation
//
// Grid : (Tk,) – one block per (dispatched-token, expert) pair
// Block: (BLOCK_H,) threads that stride over H
//
// Replaces the per-expert index_add_ calls in the reference:
// atomicAdd allows all expert outputs to be accumulated concurrently.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void weighted_scatter_accum_kernel(
    const float* __restrict__ src,        // [Tk, H] — expert output rows
    const int*   __restrict__ token_idx,  // [Tk]    — destination token
    const float* __restrict__ weight,     // [Tk]    — scalar weight per row
    float*       __restrict__ dst,        // [T,  H] — output accumulator
    int Tk, int H)
{
    int row = (int)blockIdx.x;
    if (row >= Tk) return;

    int   t = token_idx[row];
    float w = weight[row];

    for (int col = (int)threadIdx.x; col < H; col += (int)blockDim.x) {
        atomicAdd(&dst[(size_t)t * H + col],
                  src[(size_t)row * H + col] * w);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyTorch C++ wrappers
// ─────────────────────────────────────────────────────────────────────────────

/*
 * fused_routing(logits, bias, routed_scaling_factor)
 *   logits : float32 [T, 256]
 *   bias   : float32 [256]
 *   returns: (topk_idx int32 [T,8], topk_weights float32 [T,8])
 */
std::vector<torch::Tensor> fused_routing(
    torch::Tensor logits,
    torch::Tensor bias,
    float         routed_scaling_factor)
{
    TORCH_CHECK(logits.dim() == 2 && logits.size(1) == E_GLOBAL,
                "logits must be [T, 256]");
    TORCH_CHECK(bias.numel() == E_GLOBAL, "bias must have 256 elements");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(bias.scalar_type()   == torch::kFloat32, "bias must be float32");

    const at::cuda::CUDAGuard guard(logits.device());
    int T = (int)logits.size(0);

    auto topk_idx = torch::empty({T, TOP_K},
                        torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
    auto topk_w   = torch::empty({T, TOP_K},
                        torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));

    fused_routing_kernel<<<T, E_GLOBAL, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        topk_idx.data_ptr<int>(),
        topk_w.data_ptr<float>(),
        routed_scaling_factor,
        T);

    return {topk_idx, topk_w};
}

/*
 * swiglu_forward(input)
 *   input  : float32 [N, 2*I]
 *   returns: float32 [N, I]
 */
torch::Tensor swiglu_forward(torch::Tensor input)
{
    TORCH_CHECK(input.dim() == 2 && input.size(1) % 2 == 0,
                "input must be [N, 2*I]");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    const at::cuda::CUDAGuard guard(input.device());
    int N = (int)input.size(0);
    int I = (int)(input.size(1) / 2);

    auto output = torch::empty({N, I},
                      torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    constexpr int BLOCK_COLS = 256;
    dim3 block(BLOCK_COLS);
    dim3 grid(N, (I + BLOCK_COLS - 1) / BLOCK_COLS);

    swiglu_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, I);

    return output;
}

/*
 * weighted_scatter_accum(src, token_idx, weight, T, H)
 *   src       : float32 [Tk, H]
 *   token_idx : int32   [Tk]
 *   weight    : float32 [Tk]
 *   T         : number of output tokens
 *   H         : hidden dimension
 *   returns   : float32 [T, H]  (zero-initialised, then accumulated into)
 */
torch::Tensor weighted_scatter_accum(
    torch::Tensor src,
    torch::Tensor token_idx,
    torch::Tensor weight,
    int T,
    int H)
{
    TORCH_CHECK(src.dim() == 2 && src.size(1) == H, "src must be [Tk, H]");
    TORCH_CHECK(src.scalar_type()       == torch::kFloat32, "src must be float32");
    TORCH_CHECK(token_idx.scalar_type() == torch::kInt32,   "token_idx must be int32");
    TORCH_CHECK(weight.scalar_type()    == torch::kFloat32, "weight must be float32");

    const at::cuda::CUDAGuard guard(src.device());
    int Tk = (int)src.size(0);

    auto dst = torch::zeros({T, H},
                   torch::TensorOptions().dtype(torch::kFloat32).device(src.device()));

    if (Tk > 0) {
        constexpr int BLOCK_H = 256;
        weighted_scatter_accum_kernel<<<Tk, BLOCK_H, 0, at::cuda::getCurrentCUDAStream()>>>(
            src.data_ptr<float>(),
            token_idx.data_ptr<int>(),
            weight.data_ptr<float>(),
            dst.data_ptr<float>(),
            Tk, H);
    }

    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_routing",          &fused_routing,          "Fused sigmoid+group-topk+global-topk+weight-norm");
    m.def("swiglu_forward",         &swiglu_forward,         "Fused SwiGLU (SiLU(X2)*X1)");
    m.def("weighted_scatter_accum", &weighted_scatter_accum, "Weighted scatter-add accumulation");
}

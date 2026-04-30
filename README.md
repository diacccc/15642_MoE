# 15642 MoE — Custom CUDA Kernels for DeepSeek-V3/R1 MoE

Custom CUDA kernels for the Mixture-of-Experts (MoE) forward pass used in DeepSeek-V3/R1, benchmarked against a pure-PyTorch reference implementation. All experiments run on Modal B200 GPUs.

## Overview

The reference implementation (`reference/moe_ref.py`) is a faithful PyTorch translation of the DeepSeek-V3/R1 MoE forward pass. The custom CUDA path (`kernels/moe.cu`) now exposes one `moe_forward()` entry point with no cuBLAS calls: routing compacts local expert work, Stage 1 fuses GEMM1 with SwiGLU into a compact bf16 intermediate, then Stage 2 runs GEMM2 over true FP8 W2 weights and weighted scatter.

| Kernel | Replaces | Key optimization |
|---|---|---|
| `route_dispatch_kernel` | routing + Python-side token filtering | Selects top-k experts and compacts local `(token, expert, weight)` work on GPU |
| `build_active_tiles_kernel` | fixed full expert/token-tile launch grid | Builds a compact list of non-empty expert token tiles to avoid idle Tensor Core CTAs |
| `stage1_fp8_mma_kernel` | GEMM1 cuBLAS call + SwiGLU | Uses true FP8 hidden/weight inputs, computes gate/up together, and stores only bf16 `C = gate * SiLU(up)` |
| `stage2_fp8_mma_scatter_kernel` | GEMM2 cuBLAS + per-expert `index_add_` | Uses true FP8 W2 bytes, applies W2 block scales inside accumulation, then weighted scatter |

### Fixed geometry (DeepSeek-V3/R1)

| Parameter | Value |
|---|---|
| Hidden dim `H` | 7168 |
| Intermediate dim `I` | 2048 |
| Global experts `E` | 256 |
| Local experts per rank | 32 |
| Top-K experts per token | 8 |
| Expert groups | 8 (top-4 selected) |
| Weight scale block | 128 |

## Repository structure

```
├── reference/
│   └── moe_ref.py              # PyTorch reference — ground truth for correctness
├── kernels/
│   └── moe.cu                  # Custom CUDA kernels + pybind11 wrappers
├── tests/
│   └── test_moe.py             # End-to-end moe_forward correctness vs reference
├── bench/
│   └── bench_moe.py            # End-to-end latency: custom vs PyTorch
├── moe_layer.py                # Kernel loader (JIT) and input builder
├── modal_app.py                # Modal B200 entry points
└── requirements.txt
```

## Running on Modal (B200)

All compute runs remotely on Modal B200 GPUs. Install Modal once locally:

```bash
pip install modal
modal setup          # authenticate
```

Then call any entry point directly — no local GPU needed:

```bash
# Correctness: verify moe_forward matches the reference math
modal run modal_app.py::test

# Performance: end-to-end latency vs PyTorch reference
modal run modal_app.py::bench

# Interactive shell on a B200 for exploration / debugging
modal run modal_app.py::shell
```

## Kernel details

### `route_dispatch_kernel`

Implements the DeepSeek no-aux-loss routing in a single kernel. Each thread block handles one token; 256 threads (one per expert) run in parallel.

**Steps:**
1. All 256 threads compute `sigmoid(logit) + bias` in parallel
2. Threads 0–7 compute the top-2 sum score for each of the 8 expert groups
3. Thread 0 selects the top-4 groups (O(64) serial, negligible)
4. All 256 threads apply the group mask
5. Thread 0 selects the global top-8 experts from kept groups
6. Thread 0 normalises weights using `s` (without bias) scaled by `routed_scaling_factor`

Replaces the reference sequence: `sigmoid → +bias → view → topk (group) → scatter → masked_fill → topk (global) → scatter → sum → div`.

### `build_active_tiles_kernel`

Converts per-expert dispatch counts into a compact list of active `(expert, token-tile)` work. This prevents the large-`T` path from launching Tensor Core CTAs for empty expert slots.

### `stage1_fp8_mma_kernel`

Computes the gate and up projections from true FP8 `hidden` and `W13` inputs using `mma.sync.aligned.m16n8k32...e4m3.e4m3...`. It applies the 128-wide block scales, fuses `C = gate * SiLU(up)`, and stores only bf16 `C[expert, slot, 2048]` instead of the full FP32 `[gate, up]` buffer.

### `stage2_fp8_mma_scatter_kernel`

Reads bf16 `C`, packs activation fragments to E4M3, multiplies by true FP8 `W2[expert].T` fragments, applies `gemm2_weights_scale[expert, h_block, i_block]` inside accumulation, and atomically accumulates the routed weighted result into the final `[T, H]` output.

## Correctness

`moe_forward()` is tested end to end against the exact computation in `moe_ref.py` across several token counts. The test compares the final bf16 output with a `1e-2` absolute tolerance and `5%` max relative tolerance.

```bash
modal run modal_app.py::test
```

## Requirements

```
torch >= 2.7.0   # first release with Blackwell (SM100) support
ninja            # for JIT kernel compilation
modal >= 1.0
numpy
```

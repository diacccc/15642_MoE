# 15642 MoE — Custom CUDA Kernels for DeepSeek-V3/R1 MoE

Custom CUDA kernels for the Mixture-of-Experts (MoE) forward pass used in DeepSeek-V3/R1, benchmarked against a pure-PyTorch reference implementation. Experiments currently target Modal H200 GPUs.

## Overview

The reference implementation (`reference/moe_ref.py`) is a faithful PyTorch translation of the DeepSeek-V3/R1 MoE forward pass. The custom CUDA path (`kernels/moe.cu`) uses no cuBLAS calls: Python packs FP8 weights once into MMA B-fragment layout, routing compacts local expert work, Stage 1 fuses GEMM1 with SwiGLU into a block-scaled FP8 intermediate, then Stage 2 runs GEMM2 over true FP8 activations/packed W2 weights and weighted scatter.

| Kernel | Replaces | Key optimization |
|---|---|---|
| `pack_w13_kernel`, `pack_w2_kernel` | per-MMA scalar byte gathers from weight tensors | Prepack FP8 weights into `uint32` fragments consumed directly by `mma.sync` |
| `route_dispatch_kernel` | routing + Python-side token filtering | Selects top-k experts and compacts local `(token, expert, weight)` work on GPU |
| `stage1_fp8_mma_kernel` | GEMM1 cuBLAS call + SwiGLU | Uses true FP8 hidden and packed W13 fragments, computes gate/up together, and stores block-scaled FP8 `C = gate * SiLU(up)` |
| `stage2_fp8_mma_scatter_kernel` | GEMM2 cuBLAS + per-expert `index_add_` | Uses true FP8 C and packed W2 fragments, applies activation and W2 block scales inside accumulation, then weighted scatter |

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
├── modal_app.py                # Modal H200 entry points
└── requirements.txt
```

## Running on Modal (H200)

All compute runs remotely on Modal H200 GPUs. Install Modal once locally:

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

# Interactive shell on an H200 for exploration / debugging
modal run modal_app.py::shell
```

## Kernel details

### `route_dispatch_kernel`

Implements the DeepSeek no-aux-loss routing in a single kernel. Each thread block handles one token; 256 threads (one per expert) run in parallel.

**Steps:**
1. All 256 threads compute `sigmoid(logit) + bias` in parallel using `__expf`
2. Each of the 8 warps owns one 32-expert group and computes its top-2 sum with warp shuffles
3. Thread 0 selects the top-4 groups from the 8 group scores
4. Each warp repeatedly selects its best remaining candidate from selected groups with warp shuffles
5. Thread 0 chooses the global top-8 from the 8 per-group candidates
6. Thread 0 normalises weights using `s` (without bias) scaled by `routed_scaling_factor`
7. Local dispatch appends the active `(expert, token-tile)` entry when a new 32-row tile starts

Replaces the reference sequence: `sigmoid → +bias → view → topk (group) → scatter → masked_fill → topk (global) → scatter → sum → div`.

### Weight pack kernels

`pack_w13_kernel` and `pack_w2_kernel` convert row-major FP8 weight tensors into the exact `uint32` B-fragment layout consumed by `mma.sync.aligned.m16n8k32...e4m3.e4m3...`. `run_custom()` caches the packed tensors by weight storage so repeated benchmark iterations do not repack.

### `stage1_fp8_mma_kernel`

Computes the gate and up projections from true FP8 `hidden` and packed W13 fragments using `mma.sync.aligned.m16n8k32...e4m3.e4m3...`. It applies the 128-wide block scales, fuses `C = gate * SiLU(up)`, computes a dynamic scale per `(expert, slot, 128-wide I block)`, and stores FP8 `C[expert, slot, 2048]` plus `C_scale`.

### `stage2_fp8_mma_scatter_kernel`

Reads FP8 `C`, multiplies by packed W2 fragments, applies `C_scale[expert, slot, i_block] * gemm2_weights_scale[expert, h_block, i_block]` inside accumulation, and atomically accumulates the routed weighted result into the final `[T, H]` output.

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

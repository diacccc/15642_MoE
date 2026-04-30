# 15642 MoE — Custom CUDA Kernels for DeepSeek-V3/R1 MoE

Custom CUDA kernels for the Mixture-of-Experts (MoE) forward pass used in DeepSeek-V3/R1, benchmarked against a pure-PyTorch reference implementation. All experiments run on Modal B200 GPUs.

## Overview

The reference implementation (`reference/moe_ref.py`) is a faithful PyTorch translation of the DeepSeek-V3/R1 MoE forward pass. Three custom CUDA kernels (`kernels/moe.cu`) replace the most memory-bandwidth-bound portions with fused GPU kernels, each producing numerically identical results.

| Kernel | Replaces | Key optimization |
|---|---|---|
| `fused_routing_kernel` | ~10 PyTorch ops (sigmoid, topk, masked\_fill, scatter, normalize) | Single kernel launch; all 256 expert scores computed in parallel per token |
| `swiglu_kernel` | split + `torch.exp` + multiply | One pass over memory instead of materializing intermediate tensors |
| `weighted_scatter_accum_kernel` | per-expert `index_add_` loop | All expert outputs accumulated concurrently via `atomicAdd` |

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
│   └── test_moe.py             # Per-kernel correctness tests vs reference
├── bench/
│   └── bench_moe.py            # Per-kernel latency: custom vs PyTorch
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
# Correctness: verify each kernel matches the reference math
modal run modal_app.py::test

# Performance: per-kernel latency vs PyTorch equivalents
modal run modal_app.py::bench

# Interactive shell on a B200 for exploration / debugging
modal run modal_app.py::shell
```

## Kernel details

### `fused_routing_kernel`

Implements the DeepSeek no-aux-loss routing in a single kernel. Each thread block handles one token; 256 threads (one per expert) run in parallel.

**Steps:**
1. All 256 threads compute `sigmoid(logit) + bias` in parallel
2. Threads 0–7 compute the top-2 sum score for each of the 8 expert groups
3. Thread 0 selects the top-4 groups (O(64) serial, negligible)
4. All 256 threads apply the group mask
5. Thread 0 selects the global top-8 experts from kept groups
6. Thread 0 normalises weights using `s` (without bias) scaled by `routed_scaling_factor`

Replaces the reference sequence: `sigmoid → +bias → view → topk (group) → scatter → masked_fill → topk (global) → scatter → sum → div`.

### `swiglu_kernel`

Computes `silu(X2) * X1` where `X1 = input[:, :I]` and `X2 = input[:, I:]`. Reads each input element once and writes the output in the same pass, avoiding the two-read pattern of the reference split.

### `weighted_scatter_accum_kernel`

Each block handles one row of expert output. Threads stride over the hidden dimension `H=7168` and atomically accumulate `weight * expert_output` into the output buffer. Replacing the sequential per-expert `index_add_` loop allows contributions from all experts to be written concurrently.

## Correctness

Each kernel is tested against the exact computation extracted from `moe_ref.py`:

- **Routing** — selected expert sets must match exactly; weights within `1e-4` absolute error
- **SwiGLU** — element-wise max absolute error `< 1e-5`
- **Scatter accum** — element-wise max absolute error `< 1e-3`

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

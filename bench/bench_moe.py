"""
Latency benchmark: ops.moe_forward() vs reference/moe_ref.py.

Run on Modal:  modal run modal_app.py::bench
Run locally:   python bench/bench_moe.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from moe_layer import build_moe_inputs, run_custom
import reference

DEVICE  = "cuda"
WARMUP  = 10
REPEATS = 100
BENCH_T_VALUES = [1, 4, 32, 64, 256, 2048, 4096, 8192, 16384]

H        = 7168
I_DIM    = 2048
E_GLOBAL = 256
TOP_K    = 8
N_GROUP  = 8
TOPK_GROUP = 4

# Per routed token-expert pair:
#   GEMM1: [1, H] x [H, 2I] = 2 * H * 2I FLOPs
#   GEMM2: [1, I] x [I, H]  = 2 * I * H  FLOPs
FLOPS_PER_ROUTE = 2 * H * (2 * I_DIM) + 2 * I_DIM * H


def _time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPEATS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / REPEATS  # ms


@torch.no_grad()
def _local_route_count(inputs):
    logits = inputs["routing_logits"].to(torch.float32)
    bias = inputs["routing_bias"].to(torch.float32).reshape(-1)
    local_start = int(inputs["local_expert_offset"])
    e_local = int(inputs["gemm1_weights"].shape[0])

    scores = torch.sigmoid(logits)
    scores_with_bias = scores + bias
    group_size = E_GLOBAL // N_GROUP

    grouped = scores_with_bias.view(-1, N_GROUP, group_size)
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand_as(grouped).reshape(-1, E_GLOBAL)

    pruned = scores_with_bias.masked_fill(score_mask == 0, torch.finfo(torch.float32).min)
    _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    local_end = local_start + e_local
    return ((topk_idx >= local_start) & (topk_idx < local_end)).sum().item()


def _fmt_flops(flops):
    if flops >= 1e15:
        return f"{flops / 1e15:.2f}P"
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    return f"{flops / 1e6:.2f}M"


def _tflops(flops, ms):
    return flops / (ms * 1e9)


def run_bench(ops):
    print("moe_forward benchmark — custom vs reference")
    print(f"H={H}, I={I_DIM}, E={E_GLOBAL}, E_local=32, TOP_K={TOP_K}")
    print(f"FLOPs count includes useful local GEMM1+GEMM2 work only: {FLOPS_PER_ROUTE:,} per routed local expert")
    print("=" * 104)
    print(
        f"{'T':>6},{'routes':>8},{'FLOPs':>9},"
        f"{'ref (ms)':>10},{'ref TF/s':>9},"
        f"{'custom (ms)':>12},{'cust TF/s':>10},{'speedup':>8}"
    )

    for T in BENCH_T_VALUES:
        inputs = build_moe_inputs(T=T, device=DEVICE)
        local_routes = _local_route_count(inputs)
        flops = local_routes * FLOPS_PER_ROUTE

        def ref():
            reference.run(**inputs)

        def cust():
            run_custom(ops, **inputs)

        t_ref, t_cust = _time(ref), _time(cust)
        print(
            f"{T:>6},{local_routes:>8},{_fmt_flops(flops):>9},"
            f"{t_ref:>10.4f},{_tflops(flops, t_ref):>9.2f},"
            f"{t_cust:>12.4f},{_tflops(flops, t_cust):>10.2f},"
            f"{t_ref/t_cust:>7.2f}"
        )


if __name__ == "__main__":
    from moe_layer import _load_custom_ops
    run_bench(_load_custom_ops())

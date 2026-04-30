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
BENCH_T_VALUES = [1, 4, 8, 16, 32, 64, 2048, 4096, 8192, 16384]


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


def run_bench(ops):
    print("moe_forward benchmark — custom vs reference")
    print(f"H={7168}, I={2048}, E={256}, E_local={32}, TOP_K={8}")
    print("=" * 60)
    print(f"{'T':>6}  {'ref (ms)':>10}  {'custom (ms)':>12}  {'speedup':>8}")

    for T in BENCH_T_VALUES:
        inputs = build_moe_inputs(T=T, device=DEVICE)

        def ref():
            reference.run(**inputs)

        def cust():
            run_custom(ops, **inputs)

        t_ref, t_cust = _time(ref), _time(cust)
        print(f"{T:>6}  {t_ref:>10.4f}  {t_cust:>12.4f}  {t_ref/t_cust:>7.2f}x")


if __name__ == "__main__":
    from moe_layer import _load_custom_ops
    run_bench(_load_custom_ops())

"""
Per-kernel benchmarks: routing, swiglu, weighted scatter accumulation.

Accepts the pre-compiled extension as an argument so modal_app.py can call
run_bench(ext) directly without spawning a subprocess.

Run locally:
    python bench/bench_moe.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from moe_layer import build_moe_inputs, run_custom
import reference

DEVICE  = "cuda"
H       = 7168
I_DIM   = 2048
E_GLOBAL = 256
TOP_K   = 8
WARMUP  = 10
REPEATS = 100


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


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def bench_routing(ops):
    N_GROUP = 8; TOPK_GROUP = 4; gs = E_GLOBAL // N_GROUP

    print("\n── Routing  [E=256, N_GROUP=8, TOP_K=8] ──")
    print(f"{'T':>6}  {'ref (ms)':>10}  {'custom (ms)':>12}  {'speedup':>8}")

    for T in [1, 8, 32, 64, 128, 256]:
        logits = torch.randn(T, E_GLOBAL, device=DEVICE)
        bias   = torch.zeros(E_GLOBAL,    device=DEVICE)

        def ref():
            s     = torch.sigmoid(logits)
            s_wb  = s + bias
            grp   = s_wb.view(T, N_GROUP, gs)
            t2, _ = torch.topk(grp, k=2, dim=2)
            gsc   = t2.sum(2)
            _, gi = torch.topk(gsc, k=TOPK_GROUP, dim=1)
            gm    = torch.zeros_like(gsc)
            gm.scatter_(1, gi, 1.0)
            sm    = gm.unsqueeze(2).expand(T, N_GROUP, gs).reshape(T, E_GLOBAL)
            pr    = s_wb.masked_fill(sm == 0, torch.finfo(torch.float32).min)
            _, ki = torch.topk(pr, k=TOP_K, dim=1)
            M     = torch.zeros_like(s)
            M.scatter_(1, ki, 1.0)
            w     = s * M
            w / (w.sum(1, keepdim=True) + 1e-20)

        def cust():
            ops.fused_routing(logits, bias, 1.0)

        t_ref, t_cust = _time(ref), _time(cust)
        print(f"{T:>6}  {t_ref:>10.4f}  {t_cust:>12.4f}  {t_ref/t_cust:>7.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU
# ─────────────────────────────────────────────────────────────────────────────

def bench_swiglu(ops):
    print("\n── SwiGLU  [I=2048] ──")
    print(f"{'N':>6}  {'ref (ms)':>10}  {'custom (ms)':>12}  {'speedup':>8}")

    for N in [1, 8, 32, 64, 128, 256]:
        x = torch.randn(N, 2 * I_DIM, device=DEVICE)

        def ref():
            X1, X2 = x[:, :I_DIM], x[:, I_DIM:]
            return (X2 / (1.0 + torch.exp(-X2))) * X1

        def cust():
            ops.swiglu_forward(x)

        t_ref, t_cust = _time(ref), _time(cust)
        print(f"{N:>6}  {t_ref:>10.4f}  {t_cust:>12.4f}  {t_ref/t_cust:>7.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Weighted scatter accumulation
# ─────────────────────────────────────────────────────────────────────────────

def bench_scatter_accum(ops):
    print("\n── Weighted scatter accum  [H=7168] ──")
    print(f"{'Tk':>6}  {'T':>6}  {'ref (ms)':>10}  {'custom (ms)':>12}  {'speedup':>8}")

    for T, Tk in [(128, 32), (128, 128), (256, 256), (512, 512)]:
        src       = torch.randn(Tk, H, device=DEVICE)
        token_idx = torch.randint(0, T, (Tk,), device=DEVICE, dtype=torch.int32)
        weight    = torch.rand(Tk, device=DEVICE)

        def ref():
            out = torch.zeros(T, H, device=DEVICE)
            out.index_add_(0, token_idx.long(), src * weight.unsqueeze(1))

        def cust():
            ops.weighted_scatter_accum(src, token_idx, weight, T, H)

        t_ref, t_cust = _time(ref), _time(cust)
        print(f"{Tk:>6}  {T:>6}  {t_ref:>10.4f}  {t_cust:>12.4f}  {t_ref/t_cust:>7.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Full MoE benchmark (custom vs reference)
# ─────────────────────────────────────────────────────────────────────────────

def bench_moe_e2e(ops):
    print("\n── Full MoE  [H=7168, I=2048, E=256, E_local=32, TOP_K=8] ──")
    print(f"{'T':>6}  {'ref (ms)':>10}  {'custom (ms)':>12}  {'speedup':>8}")

    for T in [1, 4, 8, 16, 32, 64, 128]:
        inputs = build_moe_inputs(T=T, device=DEVICE)

        def ref():
            reference.run(**inputs)

        def cust():
            run_custom(ops, **inputs)

        t_ref, t_cust = _time(ref), _time(cust)
        print(f"{T:>6}  {t_ref:>10.4f}  {t_cust:>12.4f}  {t_ref/t_cust:>7.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# run_bench: called by modal_app.py
# ─────────────────────────────────────────────────────────────────────────────

def run_bench(ops):
    print("Per-kernel benchmarks — custom vs PyTorch reference")
    print("=" * 60)
    bench_routing(ops)
    bench_swiglu(ops)
    bench_scatter_accum(ops)
    bench_moe_e2e(ops)


# ─────────────────────────────────────────────────────────────────────────────
# Local entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from moe_layer import _load_custom_ops
    run_bench(_load_custom_ops())

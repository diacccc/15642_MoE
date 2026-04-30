"""
Correctness tests: custom CUDA kernels vs PyTorch reference.

Each test function accepts the pre-compiled extension as its first argument
so Modal can call run_all(ext) directly without spawning a subprocess.

Run locally via pytest (kernel compiled by moe_layer._load_custom_ops):
    python -m pytest tests/test_moe.py -v
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from moe_layer import build_moe_inputs, run_custom
import reference

DEVICE   = "cuda"
E_GLOBAL = 256
TOP_K    = 8
H        = 7168
I_DIM    = 2048


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check(name, ref, got, atol, max_rel):
    assert ref.shape == got.shape, f"{name}: shape mismatch {ref.shape} vs {got.shape}"
    abs_err = (ref.float() - got.float()).abs()
    max_err = abs_err.max().item()
    rel_err = (abs_err / (ref.float().abs() + 1e-6)).max().item()
    print(f"  [{name}]  max_abs={max_err:.4e}  max_rel={rel_err:.4e}")
    assert max_err <= atol,    f"{name}: max abs error {max_err:.4e} > {atol}"
    assert rel_err <= max_rel, f"{name}: max rel error {rel_err:.4e} > {max_rel}"


# ─────────────────────────────────────────────────────────────────────────────
# Per-kernel tests
# ─────────────────────────────────────────────────────────────────────────────

def test_fused_routing(ops, T):
    torch.manual_seed(42)
    logits = torch.randn(T, E_GLOBAL, device=DEVICE)
    bias   = torch.zeros(E_GLOBAL, device=DEVICE)
    rsf    = 1.0

    topk_idx_cust, topk_w_cust = ops.fused_routing(logits, bias, rsf)
    assert topk_idx_cust.shape == (T, TOP_K)
    assert topk_w_cust.shape   == (T, TOP_K)

    # Reference routing (mirrors moe_ref.py exactly)
    N_GROUP = 8; TOPK_GROUP = 4; gs = E_GLOBAL // N_GROUP
    s             = torch.sigmoid(logits)
    s_wb          = s + bias
    s_grouped     = s_wb.view(T, N_GROUP, gs)
    top2, _       = torch.topk(s_grouped, k=2, dim=2)
    group_scores  = top2.sum(2)
    _, g_idx      = torch.topk(group_scores, k=TOPK_GROUP, dim=1)
    g_mask        = torch.zeros_like(group_scores)
    g_mask.scatter_(1, g_idx, 1.0)
    score_mask    = g_mask.unsqueeze(2).expand(T, N_GROUP, gs).reshape(T, E_GLOBAL)
    neg_inf       = torch.finfo(torch.float32).min
    pruned        = s_wb.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx_r = torch.topk(pruned, k=TOP_K, dim=1)
    M             = torch.zeros_like(s)
    M.scatter_(1, topk_idx_r, 1.0)
    w             = s * M
    w             = (w / (w.sum(1, keepdim=True) + 1e-20)) * rsf

    for t in range(T):
        ref_set  = set(topk_idx_r[t].tolist())
        cust_set = set(topk_idx_cust[t].tolist())
        assert ref_set == cust_set, \
            f"token {t}: ref experts {ref_set} != custom {cust_set}"

    ref_w_vec  = w[torch.arange(T).unsqueeze(1), topk_idx_r].float()
    cust_w_vec = topk_w_cust.float()
    for t in range(T):
        ref_list  = topk_idx_r[t].tolist()
        cust_list = topk_idx_cust[t].tolist()
        order     = [cust_list.index(x) for x in ref_list]
        reordered = cust_w_vec[t][torch.tensor(order, device=DEVICE)]
        max_err   = (ref_w_vec[t] - reordered).abs().max().item()
        assert max_err < 1e-4, f"token {t}: weight mismatch {max_err:.4e}"

    print(f"  [fused_routing  T={T:>3}]  PASSED")


def test_swiglu(ops, N, I):
    torch.manual_seed(7)
    x    = torch.randn(N, 2 * I, device=DEVICE)
    got  = ops.swiglu_forward(x)
    X1, X2 = x[:, :I], x[:, I:]
    ref  = (X2 / (1.0 + torch.exp(-X2))) * X1
    _check(f"swiglu N={N} I={I}", ref, got, atol=1e-5, max_rel=1e-4)
    print(f"  [swiglu         N={N:>3} I={I}]  PASSED")


def test_weighted_scatter_accum(ops, Tk, T):
    torch.manual_seed(3)
    src       = torch.randn(Tk, H, device=DEVICE)
    token_idx = torch.randint(0, T, (Tk,), device=DEVICE, dtype=torch.int32)
    weight    = torch.rand(Tk, device=DEVICE)
    got       = ops.weighted_scatter_accum(src, token_idx, weight, T, H)
    ref       = torch.zeros(T, H, device=DEVICE)
    for i in range(Tk):
        ref[token_idx[i]] += src[i] * weight[i]
    _check(f"scatter_accum Tk={Tk} T={T}", ref, got, atol=1e-3, max_rel=0.01)
    print(f"  [scatter_accum  Tk={Tk:>3} T={T:>3}]  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Full MoE test
# ─────────────────────────────────────────────────────────────────────────────

def test_moe_e2e(ops, T):
    inputs   = build_moe_inputs(T=T, device=DEVICE, seed=T)
    out_ref  = reference.run(**inputs).float()
    out_cust = run_custom(ops, **inputs).float()
    _check(f"moe_e2e T={T}", out_ref, out_cust, atol=1e-2, max_rel=0.05)
    print(f"  [moe_e2e        T={T:>3}]  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# run_all: called by modal_app.py
# ─────────────────────────────────────────────────────────────────────────────

def run_all(ops) -> bool:
    ok = True
    try:
        print("\n── fused_routing ──")
        for T in [1, 8, 64, 128]:
            test_fused_routing(ops, T)

        print("\n── swiglu ──")
        for N, I in [(1, 2048), (16, 2048), (64, 2048)]:
            test_swiglu(ops, N, I)

        print("\n── weighted_scatter_accum ──")
        for Tk, T in [(32, 128), (1, 10), (256, 256)]:
            test_weighted_scatter_accum(ops, Tk, T)

        print("\n── full MoE (custom vs reference) ──")
        for T in [1, 8, 32, 64]:
            test_moe_e2e(ops, T)

        print("\nAll tests PASSED.")
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        ok = False
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# pytest shims (so `pytest tests/test_moe.py` still works locally)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pytest
    from moe_layer import _load_custom_ops

    @pytest.mark.parametrize("T", [1, 8, 64, 128])
    def test_fused_routing_pytest(T):
        test_fused_routing(_load_custom_ops(), T)

    @pytest.mark.parametrize("N,I", [(1, 2048), (16, 2048), (64, 2048)])
    def test_swiglu_pytest(N, I):
        test_swiglu(_load_custom_ops(), N, I)

    @pytest.mark.parametrize("Tk,T", [(32, 128), (1, 10), (256, 256)])
    def test_weighted_scatter_accum_pytest(Tk, T):
        test_weighted_scatter_accum(_load_custom_ops(), Tk, T)

except ImportError:
    pass

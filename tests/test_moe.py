"""
Correctness test: ops.moe_forward() vs reference/moe_ref.py.

Run on Modal:  modal run modal_app.py::test
Run locally:   python -m pytest tests/test_moe.py -v
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from moe_layer import build_moe_inputs, run_custom
import reference

DEVICE = "cuda"
TEST_T_VALUES = [1, 4, 32, 64, 256, 2048, 4096, 8192, 16384]


def _check(T, ref, got, atol=1e-2, rtol=2e-2):
    assert ref.shape == got.shape, f"T={T}: shape mismatch"
    ref_f = ref.float()
    got_f = got.float()
    abs_err = (ref_f - got_f).abs()
    max_err = abs_err.max().item()
    tol = atol + rtol * ref_f.abs()
    max_tol_ratio = (abs_err / tol).max().item()
    rel_mask = ref_f.abs() >= atol
    max_rel = (
        (abs_err[rel_mask] / ref_f[rel_mask].abs()).max().item()
        if rel_mask.any()
        else 0.0
    )
    bad = abs_err > tol
    bad_count = bad.sum().item()
    print(
        f"  [T={T:>3}]  max_abs={max_err:.4e}  "
        f"max_rel(|ref|>=atol)={max_rel:.4e}  "
        f"max_tol_ratio={max_tol_ratio:.4e}  bad={bad_count}"
    )
    assert not bad.any(), (
        f"T={T}: {bad_count} values exceed atol={atol} + rtol={rtol} * abs(ref); "
        f"max abs error {max_err:.4e}, max tolerance ratio {max_tol_ratio:.4e}"
    )


def test_moe(ops, T):
    inputs   = build_moe_inputs(T=T, device=DEVICE, seed=T)
    out_ref  = reference.run(**inputs).float()
    out_cust = run_custom(ops, **inputs).float()
    _check(T, out_ref, out_cust)
    print(f"  [T={T:>3}]  PASSED")


# ── run_all: called by modal_app.py ──────────────────────────────────────────

def run_all(ops) -> bool:
    ok = True
    try:
        print("\n── moe_forward correctness (custom vs reference) ──")
        for T in TEST_T_VALUES:
            test_moe(ops, T)
        print("\nAll tests PASSED.")
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        ok = False
    return ok


# ── pytest shim ───────────────────────────────────────────────────────────────

try:
    import pytest
    from moe_layer import _load_custom_ops

    @pytest.mark.parametrize("T", TEST_T_VALUES)
    def test_moe_pytest(T):
        test_moe(_load_custom_ops(), T)

except ImportError:
    pass

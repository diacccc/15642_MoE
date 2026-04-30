"""
Kernel loader, test-input builder, and full MoE forward pass.

The three kernels in kernels/moe.cu implement the same math as the
corresponding operations in reference/moe_ref.py:

  fused_routing_kernel      ↔  routing section   (sigmoid → group top-k
                                                   → global top-k → weight norm)
  swiglu_kernel             ↔  SwiGLU activation  (silu(X2) * X1)
  weighted_scatter_accum    ↔  output accumulation (weighted index_add_)

run_custom() chains all three kernels with torch.matmul for the GEMMs,
producing numerically identical output to reference/moe_ref.py.
"""

import os
import torch
from torch.utils.cpp_extension import load

# ── Geometry ──────────────────────────────────────────────────────────────────
H        = 7168
I_DIM    = 2048
E_GLOBAL = 256
E_LOCAL  = 32
BLOCK    = 128
TOP_K    = 8

# ── Lazy kernel loader ────────────────────────────────────────────────────────
_custom_ops = None

def _load_custom_ops():
    global _custom_ops
    if _custom_ops is not None:
        return _custom_ops
    src = os.path.join(os.path.dirname(__file__), "kernels", "moe.cu")
    major, minor = torch.cuda.get_device_capability()
    arch_flag = f"-arch=sm_{major}{minor}"
    _custom_ops = load(
        name="moe_kernels",
        sources=[src],
        extra_cuda_cflags=["-O3", "--use_fast_math", arch_flag],
        verbose=True,
    )
    return _custom_ops


# ── Test-input builder ────────────────────────────────────────────────────────
def build_moe_inputs(T: int = 128, device: str = "cuda", seed: int = 0):
    """Random FP8-scale inputs matching the shapes expected by moe_ref.run()."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    num_hidden_blocks    = H // BLOCK
    num_intermediate_blocks = I_DIM // BLOCK
    num_gemm1_out_blocks = (2 * I_DIM) // BLOCK

    def rfp8(*shape):
        t = torch.rand(*shape, generator=gen, device=device) * 2 - 1
        return t.to(torch.float8_e4m3fnuz)

    def rscale(*shape):
        return torch.rand(*shape, generator=gen, device=device) * 0.1 + 0.01

    return dict(
        routing_logits        = torch.randn(T, E_GLOBAL, device=device),
        routing_bias          = torch.zeros(E_GLOBAL,    device=device),
        hidden_states         = rfp8(T, H),
        hidden_states_scale   = rscale(num_hidden_blocks, T),
        gemm1_weights         = rfp8(E_LOCAL, 2 * I_DIM, H),
        gemm1_weights_scale   = rscale(E_LOCAL, num_gemm1_out_blocks, num_hidden_blocks),
        gemm2_weights         = rfp8(E_LOCAL, H, I_DIM),
        gemm2_weights_scale   = rscale(E_LOCAL, num_hidden_blocks, num_intermediate_blocks),
        local_expert_offset   = 0,
        routed_scaling_factor = 1.0,
    )


# ── Shared FP8 dequantisation ─────────────────────────────────────────────────
def _dequant(hidden_states, hidden_states_scale,
             gemm1_weights, gemm1_weights_scale,
             gemm2_weights, gemm2_weights_scale, T):
    A_fp32     = hidden_states.to(torch.float32)
    A_scale_TH = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()
    A = A_fp32 * A_scale_TH.unsqueeze(-1).repeat(1, 1, BLOCK).reshape(T, H)

    W13_fp32 = gemm1_weights.to(torch.float32)
    S13      = gemm1_weights_scale.to(torch.float32)
    W13      = W13_fp32 * torch.repeat_interleave(
                   torch.repeat_interleave(S13, BLOCK, dim=1), BLOCK, dim=2)

    W2_fp32 = gemm2_weights.to(torch.float32)
    S2      = gemm2_weights_scale.to(torch.float32)
    W2      = W2_fp32 * torch.repeat_interleave(
                  torch.repeat_interleave(S2, BLOCK, dim=1), BLOCK, dim=2)

    return A, W13, W2


# ── Full MoE forward with custom kernels ──────────────────────────────────────
@torch.no_grad()
def run_custom(ops,
               routing_logits, routing_bias,
               hidden_states, hidden_states_scale,
               gemm1_weights, gemm1_weights_scale,
               gemm2_weights, gemm2_weights_scale,
               local_expert_offset, routed_scaling_factor):
    """
    Same math as reference/moe_ref.py, using custom CUDA kernels for:
      - routing  (fused_routing_kernel)
      - SwiGLU   (swiglu_kernel)
      - weighted accumulation (weighted_scatter_accum_kernel)
    torch.matmul is used for GEMM1 and GEMM2.
    """
    T      = routing_logits.shape[0]
    device = hidden_states.device

    # 1. FP8 dequant (identical to reference)
    A, W13, W2 = _dequant(
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale, T,
    )

    # 2. Fused routing kernel
    logits_f = routing_logits.to(torch.float32).contiguous()
    bias_f   = routing_bias.to(torch.float32).reshape(-1).contiguous()
    topk_idx, topk_w = ops.fused_routing(logits_f, bias_f, float(routed_scaling_factor))
    # topk_idx: int32 [T, TOP_K],  topk_w: float32 [T, TOP_K]

    # 3. Local expert compute
    output      = torch.zeros(T, H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_LOCAL):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue

        sel_mask  = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

        # GEMM1: [Tk, H] @ [H, 2I] → [Tk, 2I]
        G1 = A.index_select(0, token_idx).matmul(W13[le].t())

        # Fused SwiGLU kernel: [Tk, 2I] → [Tk, I]
        C  = ops.swiglu_forward(G1.contiguous())

        # GEMM2: [Tk, I] @ [I, H] → [Tk, H]
        O  = C.matmul(W2[le].t())

        # Routing weight for this (token, expert) pair
        slot_mask = (topk_idx[token_idx] == ge)
        w_tok     = (topk_w[token_idx] * slot_mask.float()).sum(dim=1)

        # Fused weighted scatter-add kernel
        output += ops.weighted_scatter_accum(
            O.contiguous(),
            token_idx.to(torch.int32),
            w_tok.contiguous(),
            T, H,
        )

    return output.to(torch.bfloat16)

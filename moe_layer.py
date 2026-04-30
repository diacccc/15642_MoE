"""
Kernel loader, test-input builder, and full MoE forward pass.

kernels/moe.cu exposes a single moe_forward() that fuses:
  routing (sigmoid → group top-k → global top-k → weight norm)
  + compact local expert dispatch
  + custom FP8 Tensor Core GEMM1 with fused SwiGLU into block-scaled FP8 C
  + custom FP8 Tensor Core GEMM2 with scatter accumulation

run_custom() passes hidden_states as true FP8 tensors, packs gemm1/gemm2
weights once into the exact mma.m16n8k32 B-fragment layout, and delegates to
ops.moe_forward() without materializing dequantized weights.
"""

import os
import weakref
import torch
from torch.utils.cpp_extension import load

# ── Geometry ──────────────────────────────────────────────────────────────────
H        = 7168
I_DIM    = 2048
E_GLOBAL = 256
E_LOCAL  = 32
BLOCK    = 128

# ── Lazy kernel loader ────────────────────────────────────────────────────────
_custom_ops = None
_packed_weight_cache = {}

def _load_custom_ops():
    global _custom_ops
    if _custom_ops is not None:
        return _custom_ops
    src = os.path.join(os.path.dirname(__file__), "kernels", "moe.cu")
    major, minor = torch.cuda.get_device_capability()
    _custom_ops = load(
        name="moe_fused",
        sources=[src],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-arch=sm_{major}{minor}"],
        verbose=True,
    )
    return _custom_ops


# ── Test-input builder ────────────────────────────────────────────────────────
def build_moe_inputs(T: int = 128, device: str = "cuda", seed: int = 0):
    """Random FP8-scale inputs matching the shapes expected by moe_ref.run()."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    num_hidden_blocks       = H // BLOCK
    num_intermediate_blocks = I_DIM // BLOCK
    num_gemm1_out_blocks    = (2 * I_DIM) // BLOCK

    def rfp8(*shape):
        t = torch.rand(*shape, generator=gen, device=device) * 2 - 1
        return t.to(torch.float8_e4m3fn)

    def rscale(*shape):
        return torch.rand(*shape, generator=gen, device=device) * 0.1 + 0.01

    return dict(
        routing_logits        = torch.randn(T, E_GLOBAL, generator=gen, device=device),
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


def _cached_pack(cache_name, tensor, pack_fn):
    tensor = tensor.contiguous()
    key = (
        cache_name,
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        int(tensor._version),
    )
    entry = _packed_weight_cache.get(key)
    if entry is not None:
        owner_ref, packed = entry
        if owner_ref() is tensor:
            return packed

    packed = pack_fn(tensor)
    _packed_weight_cache[key] = (weakref.ref(tensor), packed)
    return packed


# ── Full MoE forward with custom kernel ──────────────────────────────────────
@torch.no_grad()
def run_custom(ops,
               routing_logits, routing_bias,
               hidden_states, hidden_states_scale,
               gemm1_weights, gemm1_weights_scale,
               gemm2_weights, gemm2_weights_scale,
               local_expert_offset, routed_scaling_factor):
    gemm1_weights_packed = _cached_pack("w13", gemm1_weights, ops.pack_w13)
    gemm2_weights_packed = _cached_pack("w2", gemm2_weights, ops.pack_w2)

    return ops.moe_forward(
        hidden_states.contiguous(),
        hidden_states_scale.to(torch.float32).contiguous(),
        gemm1_weights_packed,
        gemm1_weights_scale.to(torch.float32).contiguous(),
        gemm2_weights_packed,
        gemm2_weights_scale.to(torch.float32).contiguous(),
        routing_logits.to(torch.float32).contiguous(),
        routing_bias.to(torch.float32).reshape(-1).contiguous(),
        float(routed_scaling_factor),
        int(local_expert_offset),
    )

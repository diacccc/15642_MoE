import torch


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """
    DeepSeek-V3/R1 MoE reference implementation.

    FP8 block-scale dequantization → no-aux-loss routing → local expert
    compute (GEMM1 → SwiGLU → GEMM2) → weighted accumulation.

    Geometry (fixed):
        H=7168, I=2048, E_global=256, E_local=32
        TOP_K=8, N_GROUP=8, TOPK_GROUP=4, BLOCK=128
    """
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == 7168
    assert I == 2048
    assert E_global == 256
    assert E_local == 32

    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    num_hidden_blocks   = H // BLOCK           # 56
    num_intermediate_blocks = I // BLOCK       # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK    # 32

    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (num_hidden_blocks, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (E_local, num_gemm1_out_blocks, num_hidden_blocks)
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (E_local, num_hidden_blocks, num_intermediate_blocks)
    assert routing_bias.shape[-1] == E_global

    device = hidden_states.device

    # ── 1. FP8 block-scale dequantisation ────────────────────────────────────
    A_fp32  = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)           # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()          # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)                                  # [T, H/128, 128]
        .reshape(T, H)
    )
    A = A_fp32 * A_scale_expanded                             # [T, H]

    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_exp = torch.repeat_interleave(S13, BLOCK, dim=1)
    S13_exp = torch.repeat_interleave(S13_exp, BLOCK, dim=2)
    W13 = W13_fp32 * S13_exp                                  # [E_local, 2I, H]

    W2_fp32 = gemm2_weights.to(torch.float32)
    S2  = gemm2_weights_scale.to(torch.float32)
    S2_exp = torch.repeat_interleave(S2, BLOCK, dim=1)
    S2_exp = torch.repeat_interleave(S2_exp, BLOCK, dim=2)
    W2 = W2_fp32 * S2_exp                                     # [E_local, H, I]

    # ── 2. No-aux-loss routing ────────────────────────────────────────────────
    logits = routing_logits.to(torch.float32)                 # [T, E_global]
    bias   = routing_bias.to(torch.float32).reshape(-1)       # [E_global]

    s            = torch.sigmoid(logits)                      # [T, E]
    s_with_bias  = s + bias

    group_size   = E_global // N_GROUP                        # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)   # [T, 8, 32]

    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)                       # [T, 8]

    _, group_idx  = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask    = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf       = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx   = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M            = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights      = s * M
    weights_sum  = weights.sum(dim=1, keepdim=True) + 1e-20
    weights      = (weights / weights_sum) * routed_scaling_factor  # [T, E]

    # ── 3. Local expert compute ───────────────────────────────────────────────
    output      = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

        A_e   = A.index_select(0, token_idx)   # [Tk, H]
        W13_e = W13[le]                         # [2I, H]
        W2_e  = W2[le]                          # [H, I]

        G1    = A_e.matmul(W13_e.t())           # [Tk, 2I]
        X1, X2 = G1[:, :I], G1[:, I:]
        C     = (X2 / (1.0 + torch.exp(-X2))) * X1   # SwiGLU → [Tk, I]
        O     = C.matmul(W2_e.t())             # [Tk, H]

        w_tok = weights.index_select(0, token_idx)[:, ge]  # [Tk]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)

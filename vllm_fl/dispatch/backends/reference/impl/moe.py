# Copyright (c) 2026 BAAI. All rights reserved.
"""Portable PyTorch reference implementations for MoE routing."""

import torch


def topk_sigmoid_torch(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts using biased sigmoid scores, but return unbiased weights.

    Repeated argmax ensures ties choose the lowest expert index, matching the
    CUDA topk_sigmoid kernel. All operations stay on-device for graph capture.
    """
    scores = torch.nan_to_num(torch.sigmoid(gating_output.float()), nan=0.0)
    selection = (
        scores + e_score_correction_bias.float()
        if e_score_correction_bias is not None
        else scores
    )
    topk = topk_indices.shape[-1]
    if not 1 <= topk <= gating_output.shape[-1]:
        raise ValueError(
            f"topk must be between 1 and num_experts, got {topk}"
        )
    chosen = []
    for _ in range(topk):
        index = selection.argmax(dim=-1, keepdim=True)
        chosen.append(index)
        selection = selection.scatter(dim=-1, index=index, value=-torch.inf)
    indices = torch.cat(chosen, dim=-1)
    weights = scores.gather(dim=-1, index=indices)

    if renormalize:
        totals = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(totals > 0, totals, torch.ones_like(totals))

    topk_weights.copy_(weights)
    topk_indices.copy_(indices)
    token_count = gating_output.shape[0]
    source_rows = (
        torch.arange(topk, device=indices.device).unsqueeze(0) * token_count
        + torch.arange(token_count, device=indices.device).unsqueeze(1)
    )
    token_expert_indices.copy_(source_rows)
    return topk_weights, topk_indices

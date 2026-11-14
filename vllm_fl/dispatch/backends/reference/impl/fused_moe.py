# Copyright (c) 2026 BAAI. All rights reserved.
"""Device-independent reference implementations for MoE routing."""

import torch


def topk_softmax(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
):
    """Write routing outputs in place; correction bias affects selection only.

    Padding belongs to the caller's execution context, not this reference op.
    Source indices use the slot-major layout expected by the MoE kernels.
    """
    scores = torch.softmax(gating_output.float(), dim=-1)
    selection_scores = scores
    if e_score_correction_bias is not None:
        selection_scores = scores + e_score_correction_bias.float()
    indices = torch.topk(selection_scores, topk_indices.shape[1], dim=-1).indices
    weights = scores.gather(1, indices)
    if renormalize:
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
    topk_weights.copy_(weights)
    topk_indices.copy_(indices)
    rows = torch.arange(gating_output.shape[0], device=gating_output.device)
    slots = torch.arange(topk_indices.shape[1], device=gating_output.device)
    token_expert_indices.copy_(rows[:, None] + slots * gating_output.shape[0])
    return topk_weights, topk_indices

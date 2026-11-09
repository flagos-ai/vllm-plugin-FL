# Copyright (c) 2026 BAAI. All rights reserved.

"""Torch fallback for the grouped-top-k dispatch signature."""

import torch


def grouped_topk_torch(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    if scores.ndim != 2 or bias.numel() != scores.shape[-1]:
        raise ValueError("scores must be 2D and bias must match the expert count")
    num_tokens, num_experts = scores.shape
    if n_group <= 0 or num_experts % n_group:
        raise ValueError("num_experts must be divisible by a positive n_group")
    experts_per_group = num_experts // n_group
    if (
        not 1 <= topk_group <= n_group
        or not 1 <= topk <= topk_group * experts_per_group
    ):
        raise ValueError("topk exceeds the selected expert groups")
    if scoring_func not in (0, 1):
        raise ValueError("scoring_func must be 0 (none) or 1 (sigmoid)")

    processed = scores.float().sigmoid().to(scores.dtype) if scoring_func else scores
    # Bias affects selection only; output weights use the uncorrected scores.
    selection = (processed + bias.to(scores.dtype).reshape(1, -1)).float()
    grouped = selection.reshape(num_tokens, n_group, experts_per_group)
    group_scores = grouped.topk(min(2, experts_per_group), dim=-1).values.sum(-1)
    group_scores = group_scores.to(scores.dtype).float()
    groups = group_scores.argsort(dim=-1, descending=True, stable=True)[:, :topk_group]
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, groups, True)
    expert_mask = (
        group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape_as(selection)
    )
    selected = selection.masked_fill(~expert_mask, -float("inf"))
    ids = selected.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    weights = processed.gather(1, ids).float()
    if renormalize:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * routed_scaling_factor, ids.to(torch.int32)

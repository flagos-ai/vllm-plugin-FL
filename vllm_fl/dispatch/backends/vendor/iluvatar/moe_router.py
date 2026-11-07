# Copyright (c) 2026 BAAI. All rights reserved.
"""EMPTY-vLLM MoE routing adapters for Iluvatar."""

import importlib

import torch


def topk_softmax(topk_weights, topk_indices, token_expert_indices,
                 gating_output, renormalize=False, e_score_correction_bias=None):
    from vllm_fl.dispatch import call_op
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        _get_padding_mask,
    )

    if e_score_correction_bias is None:
        weights, indices = call_op(
            "topk_softmax", topk_weights, topk_indices,
            token_expert_indices, gating_output, renormalize,
        )
        # Some backends return a new normalized tensor rather than mutating it.
        topk_weights.copy_(weights)
        topk_indices.copy_(indices)
    else:
        # Bias affects selection only; the returned scores remain unbiased.
        scores = torch.softmax(gating_output.float(), dim=-1)
        indices = torch.topk(scores + e_score_correction_bias.float(),
                             topk_indices.shape[1], dim=-1).indices
        weights = scores.gather(1, indices)
        if renormalize:
            denominator = weights.sum(dim=-1, keepdim=True)
            weights = weights / torch.where(
                denominator > 0, denominator, torch.ones_like(denominator))
        topk_weights.copy_(weights)
        topk_indices.copy_(indices)
        rows = torch.arange(gating_output.shape[0], device=gating_output.device)
        slots = torch.arange(topk_indices.shape[1], device=gating_output.device)
        token_expert_indices.copy_(rows[:, None] + slots * gating_output.shape[0])
    padding = _get_padding_mask(topk_indices.shape[0])
    if padding is not None:
        topk_indices.masked_fill_(padding[:, None], -1)
    return topk_weights, topk_indices


def patch_moe_router():
    """Replace only Iluvatar's missing CUDA extension entry points."""
    for name in ("fused_topk_router", "fused_topk_bias_router"):
        module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router." + name)
        module.vllm_topk_softmax = topk_softmax

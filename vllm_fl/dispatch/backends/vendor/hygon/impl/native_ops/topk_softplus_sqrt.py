# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon implementation of vLLM's ``topk_softplus_sqrt`` router op."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_softplus_sqrt_hygon(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    correction_bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
) -> None:
    """Preserve Hygon scores and use a guarded fused decode routing tail.

    Bias affects expert selection only; weights always come from the unbiased
    ``sqrt(softplus(logit))`` scores.
    """
    if gating_output.is_cuda and torch.version.hip is not None:
        from .router_fusion import try_router_fusion

        if try_router_fusion(
            topk_weights, topk_indices, gating_output, renormalize,
            routed_scaling_factor, correction_bias, input_ids, tid2eid,
        ):
            # The existing Hygon op intentionally leaves token_expert_indices
            # untouched; it is not consumed by this external-router ABI.
            return
    scores = torch.sqrt(F.softplus(gating_output.float()))
    scores_for_choice = (
        scores + correction_bias.float()
        if correction_bias is not None
        else scores
    )

    if tid2eid is not None:
        if input_ids is None:
            raise ValueError("input_ids is required when tid2eid is provided")
        selected_indices = tid2eid[input_ids.long()]
    else:
        selected_indices = torch.topk(
            scores_for_choice,
            k=topk_weights.shape[-1],
            dim=-1,
        ).indices

    weights = scores.gather(1, selected_indices.long())
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)

    topk_weights.copy_(weights * routed_scaling_factor)
    topk_indices.copy_(selected_indices)


__all__ = ["topk_softplus_sqrt_hygon"]

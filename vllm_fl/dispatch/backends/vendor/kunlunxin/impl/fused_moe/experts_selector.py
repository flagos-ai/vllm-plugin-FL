# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 2026 - Modified by Kunlunxin, Inc. All Rights Reserved.

"""
Kunlunxin expert routing functions: top-k selection, grouped top-k, and
the unified select_experts entry point.
"""

from __future__ import annotations

from typing import Optional

import torch

import xtorch_ops


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, ...]:
    if renormalize:
        xtorch_ops.moe_softmax_topk_norm(
            gating_output, topk_weights, topk_indices, None,
        )
    else:
        xtorch_ops.moe_softmax_topk(
            gating_output, topk_weights, topk_indices, None,
        )
    return topk_weights, topk_indices


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused top-k selection with softmax."""
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"
    M, _ = hidden_states.size()

    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(
        M, topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    topk_weights, topk_ids = vllm_topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
    )
    return topk_weights, topk_ids, token_expert_indices


def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    n_routed_experts = gating_output.shape[-1]
    scores = gating_output.softmax(dim=-1)
    scores_for_choice = scores.view(-1, n_routed_experts) + e_score_correction_bias.unsqueeze(0)
    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped top-k selection."""
    seq_num = gating_output.shape[0]

    if scoring_func == "softmax":
        scores = gating_output.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring_func: {scoring_func}")

    if e_score_correction_bias is not None:
        assert e_score_correction_bias.dtype == torch.float32
        scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores

    topk_weights = torch.empty((seq_num, topk), dtype=torch.float, device=gating_output.device)
    topk_ids = torch.empty((seq_num, topk), dtype=torch.int32, device=gating_output.device)

    xtorch_ops.moe_group_topk(
        scores_for_choice, num_expert_group, topk_group,
        topk_weights, topk_ids, None,
    )

    if e_score_correction_bias is not None:
        topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids

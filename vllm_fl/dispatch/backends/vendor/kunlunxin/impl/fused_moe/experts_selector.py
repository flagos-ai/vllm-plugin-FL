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

from vllm.model_executor.layers.fused_moe.layer import FusedMoE

import xtorch_ops


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, ...]:
    # moe_softmax_topk_norm: fused softmax + topk + renormalize
    # moe_softmax_topk:      fused softmax + topk (no renormalize)
    if renormalize:
        xtorch_ops.moe_softmax_topk_norm(
            gating_output,
            topk_weights,
            topk_indices,
            None,  # block_statistic
        )
    else:
        xtorch_ops.moe_softmax_topk(
            gating_output,
            topk_weights,
            topk_indices,
            None,  # block_statistic
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

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

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
    scores_for_choice = scores.view(
        -1, n_routed_experts) + e_score_correction_bias.unsqueeze(0)
    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1,
                              sorted=False)[1]
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
    """Grouped top-k selection.
    """
    seq_num = gating_output.shape[0]

    # scoring function
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

    # grouped topk
    topk_weights = torch.empty(
        (seq_num, topk), dtype=torch.float, device=gating_output.device
    )
    topk_ids = torch.empty(
        (seq_num, topk), dtype=torch.int32, device=gating_output.device
    )

    xtorch_ops.moe_group_topk(
        scores_for_choice,
        num_expert_group,
        topk_group,
        topk_weights,
        topk_ids,
        None,  #block_statistic
    )

    # Use original unbiased scores for the routing weights
    if e_score_correction_bias is not None:
        topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


def select_experts(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Route the input hidden states to the top-k experts based on the
    router logits.

    Returns:
            (topk_weights, topk_ids, zero_expert_result)
            (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            The weights, expert ids, and zero expert computation result.

        **Compatibility**: When EPLB is not enabled, the returned ids are
        equivalent to global logical ids, so should be compatible with
        plain MoE implementations without redundant experts.
    """

    def valid_grouping() -> bool:
        num_experts = router_logits.shape[-1]
        if num_experts <= layer.num_expert_group:
            return False
        return num_experts % layer.num_expert_group == 0

    indices_type = layer.quant_method.topk_indices_dtype

    if layer.use_grouped_topk and valid_grouping():
        assert layer.topk_group is not None
        assert layer.num_expert_group is not None

        topk_weights, topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=layer.top_k,
            renormalize=layer.renormalize,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            scoring_func=layer.scoring_func,
            routed_scaling_factor=layer.routed_scaling_factor,
            e_score_correction_bias=layer.e_score_correction_bias,
        )
    elif layer.e_score_correction_bias is not None:
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            e_score_correction_bias=layer.e_score_correction_bias.data,
            topk=layer.top_k,
            renormalize=layer.renormalize,
        )
        if layer.routed_scaling_factor != 1.0:
            topk_weights *= layer.routed_scaling_factor
    elif layer.custom_routing_function is None:
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=layer.top_k,
            renormalize=layer.renormalize,
            indices_type=indices_type,
        )
    else:
        topk_weights, topk_ids = layer.custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=layer.top_k,
            renormalize=layer.renormalize,
        )

    # TODO: eplb
    assert layer.enable_eplb == False

    if indices_type is not None and topk_ids.dtype != indices_type:
        topk_ids = topk_ids.to(dtype=indices_type)

    return topk_weights, topk_ids, None

# Copyright (c) 2025 BAAI. All rights reserved.
# FL router subclasses that route ops through call_op dispatch.

import torch
from functools import partial

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    rocm_aiter_grouped_topk,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
    fused_topk_bias
)
from vllm_fl.dispatch import CachedOp

_topk_softmax = CachedOp("topk_softmax")
_grouped_topk = CachedOp("grouped_topk")

def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # topk_weights, topk_ids = vllm_topk_softmax(
    topk_weights, topk_ids = _topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_ids, token_expert_indices

class FusedTopKRouterFL(FusedTopKRouter):
    """FL router that routes topk_softmax through call_op."""

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
        )
        return topk_weights, topk_ids


def _fl_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """grouped_topk that routes ops.grouped_topk through call_op."""
    assert hidden_states.size(0) == gating_output.size(0), (
        "Number of tokens mismatch"
    )

    if e_score_correction_bias is not None:
        if scoring_func == "sigmoid":
            topk_values, topk_indices = _grouped_topk(
                gating_output,
                num_expert_group,
                topk_group,
                topk,
                renormalize,
                routed_scaling_factor,
                e_score_correction_bias,
                1,  # scoring_func=1 for sigmoid
            )
        elif scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
            topk_values, topk_indices = _grouped_topk(
                scores,
                num_expert_group,
                topk_group,
                topk,
                renormalize,
                routed_scaling_factor,
                e_score_correction_bias,
                0,  # scoring_func=0
            )
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")
        return topk_values, topk_indices

    # Fallback: no e_score_correction_bias, use pure-torch path
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(gating_output)
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_experts = scores.size(-1)
    group_size = num_experts // num_expert_group

    scores_grouped = scores.view(-1, num_expert_group, group_size)
    group_scores = scores_grouped.amax(dim=-1)
    _, selected_groups = torch.topk(
        group_scores, k=topk_group, dim=-1, sorted=False
    )
    mask = torch.zeros_like(scores)
    for i in range(topk_group):
        group_idx = selected_groups[:, i]
        start = group_idx * group_size
        for j in range(group_size):
            mask.scatter_(1, (start + j).unsqueeze(1), 1.0)

    scores = scores * mask
    topk_weights, topk_ids = torch.topk(
        scores, k=topk, dim=-1, sorted=False
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class GroupedTopKRouterFL(GroupedTopKRouter):
    """FL router that routes grouped_topk through call_op."""

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirror upstream GroupedTopKRouter._compute_routing, which checks
        # expert/grouping validity via an inner `valid_grouping()` closure
        # rather than a `_valid_grouping` method. The earlier port called a
        # non-existent `self._valid_grouping`, raising AttributeError.
        def valid_grouping() -> bool:
            num_experts = router_logits.shape[-1]
            if num_experts <= self.num_expert_group:
                return False
            return num_experts % self.num_expert_group == 0

        if not valid_grouping():
            if self.e_score_correction_bias is not None:
                topk_weights, topk_ids = fused_topk_bias(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    e_score_correction_bias=self.e_score_correction_bias.data,
                    topk=self.top_k,
                    renormalize=self.renormalize,
                    scoring_func=self.scoring_func,
                    indices_type=indices_type,
                )
                if self.routed_scaling_factor != 1.0:
                    topk_weights *= self.routed_scaling_factor
            else:
                topk_weights, topk_ids, _ = fused_topk(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    topk=self.top_k,
                    renormalize=self.renormalize,
                    indices_type=indices_type,
                )
            return topk_weights, topk_ids

        if rocm_aiter_ops.is_fused_moe_enabled():
            if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
                assert self.num_fused_shared_experts == 0
            grouped_topk_impl = partial(
                rocm_aiter_grouped_topk,
                num_fused_shared_experts=self.num_fused_shared_experts,
            )
        else:
            grouped_topk_impl = _fl_grouped_topk

        topk_weights, topk_ids = grouped_topk_impl(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
        )

        return topk_weights, topk_ids


class FusedTopKBiasRouterFL(FusedTopKBiasRouter):
    """FL router that routes topk_softmax (with bias) through call_op."""

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            e_score_correction_bias=self.e_score_correction_bias.data
            if self.e_score_correction_bias is not None
            else None,
            topk=self.top_k,
            renormalize=self.renormalize,
            scoring_func=self.scoring_func,
            indices_type=indices_type,
        )

        if self.routed_scaling_factor != 1.0:
            topk_weights *= self.routed_scaling_factor

        return topk_weights, topk_ids

def replace_router_with_fl() -> None:
    """Monkey-patch upstream router classes to their FL subclasses (in-place)."""
    FusedTopKRouter._compute_routing = FusedTopKRouterFL._compute_routing
    GroupedTopKRouter._compute_routing = GroupedTopKRouterFL._compute_routing
    FusedTopKBiasRouter._compute_routing = FusedTopKBiasRouterFL._compute_routing

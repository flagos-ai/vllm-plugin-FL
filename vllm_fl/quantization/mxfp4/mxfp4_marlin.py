# Copyright (c) 2026 BAAI. All rights reserved.
"""MXFP4 Marlin experts integrated with FL operator dispatch."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts
from vllm_fl.dispatch import CachedOp

_fused_marlin_moe = CachedOp("fused_marlin_moe")


class MarlinExpertsFL(MarlinExperts):
    """Upstream MarlinExperts with both regular and LoRA calls dispatched."""

    def _dispatch_marlin(
        self, *, output, hidden_states, w1, w2, topk_weights, topk_ids,
        activation, global_num_experts, expert_map, workspace13, workspace2,
        apply_router_weight_on_input, activation_func, moe_sum,
        clamp_limit=None,
    ):
        return _fused_marlin_moe(
            hidden_states=hidden_states, w1=w1, w2=w2,
            bias1=self.w1_bias, bias2=self.w2_bias,
            w1_scale=self.w1_scale, w2_scale=self.w2_scale,
            topk_weights=topk_weights, topk_ids=topk_ids,
            global_scale1=self.g1_alphas, global_scale2=self.g2_alphas,
            quant_type_id=self.quant_type_id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts, activation=activation,
            activation_func=activation_func, moe_sum=moe_sum,
            expert_map=expert_map, output=output,
            intermediate_cache13=workspace2,
            intermediate_cache2=workspace13,
            g_idx1=self.w13_g_idx, g_idx2=self.w2_g_idx,
            sort_indices1=self.w13_g_idx_sort_indices,
            sort_indices2=self.w2_g_idx_sort_indices,
            is_k_full=self.is_k_full, input_dtype=self.input_dtype,
            clamp_limit=clamp_limit,
        )

    def apply(
        self, output: torch.Tensor, hidden_states: torch.Tensor,
        w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor,
        topk_ids: torch.Tensor, activation: MoEActivation,
        global_num_experts: int, expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None, a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor, workspace2: torch.Tensor,
        expert_tokens_meta, apply_router_weight_on_input: bool,
    ):
        assert self.w1_scale is not None and self.w2_scale is not None
        ctx = self._lora_context
        if ctx is None:
            self._dispatch_marlin(
                output=output, hidden_states=hidden_states, w1=w1, w2=w2,
                topk_weights=topk_weights, topk_ids=topk_ids,
                activation=activation, global_num_experts=global_num_experts,
                expert_map=expert_map, workspace13=workspace13,
                workspace2=workspace2,
                apply_router_weight_on_input=apply_router_weight_on_input,
                activation_func=self.activation, moe_sum=self.moe_sum,
            )
            return

        # Kept in sync with upstream MarlinExperts.apply's LoRA callbacks.
        num_tokens = hidden_states.size(0)
        top_k_num = topk_ids.size(1)
        state: dict = {}

        def activation_with_lora(act_enum, act_output, act_input):
            sorted_ids, expert_ids, num_padded, token_mapping = self.apply_w13_lora(
                ctx, y=act_input, x=hidden_states, topk_ids=topk_ids,
                topk_weights=topk_weights, expert_map=expert_map, w1=w1, w2=w2,
                num_tokens=num_tokens, top_k_num=top_k_num,
            )
            state.update(sorted=sorted_ids, eids=expert_ids, npad=num_padded,
                         tlm=token_mapping, cache2=act_output)
            self.activation(act_enum, act_output, act_input)

        def moe_sum_with_lora(moe_out, out):
            self.apply_w2_lora(
                ctx, y=moe_out, x=state["cache2"], topk_weights=topk_weights,
                sorted_token_ids_lora=state["sorted"],
                expert_ids_lora=state["eids"],
                num_tokens_post_padded_lora=state["npad"],
                token_lora_mapping=state["tlm"], num_tokens=num_tokens,
                w1=w1, w2=w2, top_k_num=top_k_num,
            )
            self.moe_sum(moe_out, out)

        return self._dispatch_marlin(
            output=output, hidden_states=hidden_states, w1=w1, w2=w2,
            topk_weights=topk_weights, topk_ids=topk_ids,
            activation=activation, global_num_experts=global_num_experts,
            expert_map=expert_map, workspace13=workspace13,
            workspace2=workspace2,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation_func=activation_with_lora, moe_sum=moe_sum_with_lora,
            clamp_limit=self.gemm1_clamp_limit,
        )


__all__ = ["MarlinExpertsFL"]

# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Ascend-native unquantized MoE method.

The execution order and weight layout are copied from the non-quantized
AllGather path in vLLM-Ascend v0.20.2rc1.  The surrounding vLLM modular MoE
runner is intentionally retained: it already owns shared-expert execution and
the final tensor-parallel reduction.  Keeping those responsibilities there
lets FL reuse the current Ascend routing/GMM implementation without importing
``vllm_ascend`` at runtime.
"""

from __future__ import annotations

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Current vLLM-Ascend unquantized expert compute for the FL runtime."""

    def __init__(self, moe: FusedMoEConfig):
        # The parent creates the ordinary unquantized method state.  Its
        # selected modular kernel is deliberately not initialized below.
        super().__init__(moe)

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # This method directly implements the Ascend AllGather compute path;
        # do not replace it with vLLM's CUDA-oriented modular kernel.
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Skip UnquantizedFusedMoEMethod.process_weights_after_loading: it
        # converts weights for the selected upstream modular backend.  This is
        # the same superclass call used by vLLM-Ascend's current implementation.
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        # vLLM loads [experts, out, in].  Ascend GMM consumes
        # [experts, in, out], so transpose once after loading rather than on
        # every prefill/decode invocation.
        w13 = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        w2 = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts_input

        topk_weights = topk_weights.to(x.dtype)
        expert_map = getattr(layer, "_expert_map", None)
        global_redundant_expert_num = max(
            0,
            int(getattr(layer, "global_num_experts", self.moe.num_experts))
            - int(getattr(layer, "logical_num_experts", self.moe.num_experts)),
        )

        apply_router_weight_on_input = bool(
            getattr(layer, "apply_router_weight_on_input", False)
        )
        if apply_router_weight_on_input:
            if topk_weights.shape[-1] != 1:
                raise ValueError(
                    "Ascend apply_router_weight_on_input requires top_k == 1"
                )
            x = x * topk_weights

        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            valid = expert_map[topk_ids.long()] != -1
            topk_weights = topk_weights * valid.to(topk_weights.dtype)
            first_expert_idx = int(layer.ep_rank) * int(layer.local_num_experts)
            last_expert_idx = first_expert_idx + int(layer.local_num_experts)
        else:
            first_expert_idx = 0
            last_expert_idx = int(layer.local_num_experts)
            global_num_experts = int(layer.local_num_experts)

        num_tokens = x.shape[:-1].numel()
        sorted_x, expanded_row_idx, expert_tokens, _ = (
            torch.ops._C_ascend.npu_moe_init_routing_custom(
                x,
                topk_ids.to(torch.int32),
                active_num=num_tokens * topk_ids.shape[-1],
                expert_num=global_num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[first_expert_idx, last_expert_idx],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)

        gate_up = torch_npu.npu_grouped_matmul(
            x=[sorted_x],
            weight=[layer.w13_weight],
            bias=[layer.w13_bias.to(torch.float32)] if self.moe.has_bias else None,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
        )[0]

        activation = getattr(self.moe.activation, "value", self.moe.activation)
        if activation == "silu":
            gate_up = torch_npu.npu_swiglu(gate_up)
        elif activation == "gelu":
            gate_up = torch_npu.npu_gelu_mul(gate_up)
        else:
            raise ValueError(f"Unsupported Ascend MoE activation: {activation}")

        routed = torch_npu.npu_grouped_matmul(
            x=[gate_up],
            weight=[layer.w2_weight],
            bias=[layer.w2_bias.to(torch.float32)] if self.moe.has_bias else None,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
        )[0]

        return torch_npu.npu_moe_token_unpermute(
            permuted_tokens=routed,
            sorted_indices=torch.abs(expanded_row_idx),
            probs=None
            if apply_router_weight_on_input
            else topk_weights,
        )


__all__ = ["AscendUnquantizedFusedMoEMethod"]

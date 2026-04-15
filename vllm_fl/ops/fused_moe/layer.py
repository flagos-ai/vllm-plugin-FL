# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/layer.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Union
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)

from vllm_fl.ops.fused_moe.fused_moe import fused_experts
from vllm_fl.ops.fused_moe.router import (
    FusedTopKRouterFL,
    FusedTopKBiasRouterFL,
    GroupedTopKRouterFL,
)

# Mapping from upstream router classes to FL router subclasses.
# Order matters: check subclasses before base classes.
_ROUTER_CLASS_MAP = {
    GroupedTopKRouter: GroupedTopKRouterFL,
    FusedTopKBiasRouter: FusedTopKBiasRouterFL,
    FusedTopKRouter: FusedTopKRouterFL,
}


class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    def forward_oot(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            quant_config=self.moe_quant_config,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )


class FusedMoEFL(FusedMoE):
    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(
                hidden_states,
                (0, self.hidden_size - og_hidden_states),
                mode="constant",
                value=0.0,
            )

        if self.shared_experts is None:
            fused_output = torch.ops.vllm.moe_forward(
                hidden_states, router_logits, None, self.layer_name
            )
            return fused_output[..., :og_hidden_states]
        else:
            shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                hidden_states, router_logits, None, self.layer_name
            )
            return (
                shared_output[..., :og_hidden_states],
                fused_output[..., :og_hidden_states],
            )

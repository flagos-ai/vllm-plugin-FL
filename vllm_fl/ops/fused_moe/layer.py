# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/layer.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Literal, Optional, Union
import torch
import torch.nn.functional as F
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.routing_simulator import (
    RoutingSimulator)
from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
from vllm.platforms import current_platform

if current_platform.is_cuda_alike():
    from vllm.model_executor.layers.fused_moe.fused_moe import eplb_map_to_physical_and_record
else:
    def _eplb_map_to_physical_and_record(
            topk_ids: torch.Tensor, expert_load_view: torch.Tensor,
            logical_to_physical_map: torch.Tensor,
            logical_replica_count: torch.Tensor,
            indices_type: Optional[torch.dtype]) -> torch.Tensor:
        # CPU fallback: no EPLB so just return as is
        return topk_ids

    eplb_map_to_physical_and_record = _eplb_map_to_physical_and_record

from vllm.model_executor.layers.fused_moe.fused_moe import (
    zero_experts_compute_triton)


from vllm_fl.ops.fused_moe.fused_moe import fused_experts

class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):

    def forward_oot(self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        zero_expert_num = getattr(layer, 'zero_expert_num', 0)
        zero_expert_type = getattr(layer, 'zero_expert_type', None)
        topk_weights, topk_ids, zero_expert_result = FusedMoEFL.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            global_num_experts=global_num_experts,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type)
        
        result = fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

        if zero_expert_num != 0 and zero_expert_type is not None:
            assert not isinstance(result, tuple), \
                "Shared + zero experts are mutually exclusive not yet supported"
            return result, zero_expert_result
        else:
            return result
    forward_native = forward_oot
        

class FusedMoEFL(FusedMoE):
    def forward_oot(self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(hidden_states,
                                  (0, self.hidden_size - og_hidden_states),
                                  mode='constant',
                                  value=0.0)
            
        if self.shared_experts is None:
            fused_output = torch.ops.vllm.moe_forward(
                hidden_states, router_logits, self.layer_name)
            return fused_output[..., :og_hidden_states]
        else:
            shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                hidden_states, router_logits, self.layer_name)
            return (shared_output[..., :og_hidden_states],
                    fused_output[..., :og_hidden_states])

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        indices_type: Optional[torch.dtype] = None,
        enable_eplb: bool = False,
        expert_map: Optional[torch.Tensor] = None,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
        global_num_experts: Optional[int] = None,
        zero_expert_num: Optional[int] = None,
        zero_expert_type: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        from vllm_fl.ops.fused_moe.fused_moe import fused_topk
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk_bias
        # Check if we should use a routing simulation strategy
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=top_k,
                indices_type=indices_type)

        # DeepSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        elif e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=e_score_correction_bias.data,
                topk=top_k,
                renormalize=renormalize,
            )
            if routed_scaling_factor is not None:
                topk_weights *= routed_scaling_factor
        elif custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None

            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                indices_type=indices_type,
            )

        assert topk_ids.dtype == indices_type or indices_type is None

        # Compute zero expert result if needed
        if (zero_expert_num is not None and zero_expert_num > 0
                and zero_expert_type is not None
                and global_num_experts is not None):
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=hidden_states,
            )
        else:
            zero_expert_result = None
        return topk_weights, topk_ids, zero_expert_result

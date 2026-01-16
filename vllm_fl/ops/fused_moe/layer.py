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

    def forward_oot(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, zero_expert_result = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )
        result = fused_experts(
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

        if layer.zero_expert_num != 0 and layer.zero_expert_type is not None:
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

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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

        if self.enable_eplb:
            if self.quant_method.supports_eplb:
                if self.expert_load_view is None:
                    raise ValueError(
                        "enable_eplb=True requiere expert_load_view != None"
                    )
                if self.logical_to_physical_map is None:
                    raise ValueError(
                        "enable_eplb=True requiere logical_to_physical_map != None"
                    )
                if self.logical_replica_count is None:
                    raise ValueError(
                        "enable_eplb=True requiere logical_replica_count != None"
                    )
            else:
                raise NotImplementedError(
                    f"EPLB is not supported for {self.quant_method.method_name}."
                )

        def valid_grouping() -> bool:
            # Check if num_experts is greater than num_expert_group
            # and is divisible by num_expert_group
            num_experts = router_logits.shape[-1]
            if num_experts <= self.num_expert_group:
                return False
            return num_experts % self.num_expert_group == 0

        indices_type = self.quant_method.topk_indices_dtype

        # Check if we should use a routing simulation strategy
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=self.top_k,
                indices_type=indices_type,
            )

        # DeepSeekv2 uses grouped_top_k
        elif self.use_grouped_topk and valid_grouping():
            assert self.topk_group is not None
            assert self.num_expert_group is not None
            if rocm_aiter_ops.is_fused_moe_enabled():
                if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
                    assert self.num_fused_shared_experts == 0
                grouped_topk_impl = partial(
                    rocm_aiter_grouped_topk,
                    num_fused_shared_experts=self.num_fused_shared_experts,
                )
            else:
                grouped_topk_impl = grouped_topk

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
        elif self.e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=self.e_score_correction_bias.data,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
            if self.routed_scaling_factor != 1.0:
                topk_weights *= self.routed_scaling_factor
        elif self.custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = self.custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )

        if self.enable_eplb:
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None

        # Compute zero expert result if needed
        if (
            self.zero_expert_num is not None
            and self.zero_expert_num > 0
            and self.zero_expert_type is not None
            and self.global_num_experts is not None
        ):
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=self.global_num_experts,
                zero_expert_type=self.zero_expert_type,
                hidden_states=hidden_states,
            )
        else:
            zero_expert_result = None
        return topk_weights, topk_ids, zero_expert_result
    FusedMoE.select_experts = select_experts

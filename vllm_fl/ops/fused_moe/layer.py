# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from vllm/model_executor/layers/fused_moe/layer.py

import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
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
    GroupedTopKRouterFL,
    FusedTopKBiasRouterFL,
)


class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    """OOT replacement for UnquantizedFusedMoEMethod that routes computation through flaggems."""

    def forward_oot(
        self,
        layer: "FusedMoE",
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
    """
    PluggableLayer OOT replacement for FusedMoE that routes both routing and
    computation through the dispatch system (call_op) to use flaggems operators.

    This class follows the PluggableLayer design pattern:
    - Registered as OOT replacement via op_registry_oot
    - When FusedMoE() is instantiated, FusedMoEFL is created instead
    - Router operations use call_op("topk_softmax"/"grouped_topk")
    - Expert computation uses call_op("dispatch_fused_moe_kernel")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace router with FL version that uses call_op for flaggems dispatch
        self._replace_router_with_fl()

    def _replace_router_with_fl(self):
        """Replace the router with FL version that routes through call_op dispatch."""
        router = self.router

        if isinstance(router, GroupedTopKRouter):
            # Create FL router with same parameters
            self.router = GroupedTopKRouterFL(
                top_k=router.top_k,
                global_num_experts=router.global_num_experts,
                eplb_state=router.eplb_state,
                num_expert_group=router.num_expert_group,
                topk_group=router.topk_group,
                renormalize=router.renormalize,
                scoring_func=router.scoring_func,
                routed_scaling_factor=router.routed_scaling_factor,
                e_score_correction_bias=router.e_score_correction_bias,
                num_fused_shared_experts=router.num_fused_shared_experts,
                enable_eplb=router.enable_eplb,
                indices_type_getter=router.indices_type_getter,
            )
        elif isinstance(router, FusedTopKBiasRouter):
            self.router = FusedTopKBiasRouterFL(
                top_k=router.top_k,
                global_num_experts=router.global_num_experts,
                eplb_state=router.eplb_state,
                e_score_correction_bias=router.e_score_correction_bias,
                scoring_func=router.scoring_func,
                renormalize=router.renormalize,
                routed_scaling_factor=router.routed_scaling_factor,
                enable_eplb=router.enable_eplb,
                indices_type_getter=router.indices_type_getter,
            )
        elif isinstance(router, FusedTopKRouter):
            self.router = FusedTopKRouterFL(
                top_k=router.top_k,
                global_num_experts=router.global_num_experts,
                eplb_state=router.eplb_state,
                scoring_func=router.scoring_func,
                renormalize=router.renormalize,
                enable_eplb=router.enable_eplb,
                indices_type_getter=router.indices_type_getter,
            )

        # Re-initialize runner with the new FL router
        self.runner = self._init_runner()


class SharedFusedMoEFL(SharedFusedMoE, FusedMoEFL):
    """OOT replacement for SharedFusedMoE.

    PluggableLayer.__new__ matches by cls.__name__, so SharedFusedMoE
    needs its own registration entry.  The FL router/expert replacement
    logic is inherited from FusedMoEFL via MRO.
    """
    pass


__all__ = ["FusedMoEFL", "SharedFusedMoEFL", "UnquantizedFusedMoEMethodFL"]

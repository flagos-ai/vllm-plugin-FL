# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from vllm/model_executor/layers/fused_moe/layer.py (v0.20.2)

import torch
import inspect

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
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

from vllm_fl.ops.fused_moe.router import (
    FusedTopKRouterFL,
    GroupedTopKRouterFL,
    FusedTopKBiasRouterFL,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)

from .fused_moe_utils import select_unquantized_moe_backend_oot
from .mxfp4_selector import select_fixed_mxfp4_method
from vllm_fl.quantization.mxfp4.mxfp4_marlin import MarlinExpertsFL

class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    """OOT replacement for UnquantizedFusedMoEMethod that routes computation through flaggems."""
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend_oot(
            moe_config=self.moe
        )

    @property
    def is_monolithic(self) -> bool:
        if self.moe_kernel is None:
            if self.experts_cls is None:
                return True
            return self.experts_cls.is_monolithic()
        return self.moe_kernel.is_monolithic


class FusedMoEFL(FusedMoE):
    """
    PluggableLayer OOT replacement for FusedMoE that routes both routing and
    computation through the dispatch system (call_op) to use flaggems operators.

    This class follows the PluggableLayer design pattern:
    - Registered as OOT replacement via op_registry_oot
    - When FusedMoE() is instantiated, FusedMoEFL is created instead
    - Router operations use call_op("topk_softmax"/"grouped_topk")
    - Expert computation uses call_op("invoke_fused_moe_triton_kernel")
    """

    def __init__(self, *args, **kwargs):
        routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
        shared_experts = kwargs.get("shared_experts", None)
        gate = kwargs.get("gate", None)
        routed_input_transform = kwargs.get("routed_input_transform", None)
        routed_output_transform = kwargs.get("routed_output_transform", None)
        apply_routed_scale_to_output = kwargs.get("apply_routed_scale_to_output", False)
        super().__init__(*args, **kwargs)
        self._routed_scaling_factor = routed_scaling_factor
        self._apply_routed_scale_to_output = apply_routed_scale_to_output
        # Replace quant_method with FL version that properly handles OOT backend
        if isinstance(self.quant_method, UnquantizedFusedMoEMethod) and not isinstance(self.quant_method, UnquantizedFusedMoEMethodFL):
            self.quant_method = UnquantizedFusedMoEMethodFL(self.moe_config)
            self.base_quant_method = self.quant_method
        else:
            from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
            from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts

            if isinstance(self.quant_method, Mxfp4MoEMethod):
                self.quant_method = select_fixed_mxfp4_method(
                    self.quant_method, self.moe_config
                )
                self.base_quant_method = self.quant_method
            elif getattr(self.quant_method, "experts_cls", None) is MarlinExperts:
                self.quant_method.experts_cls = MarlinExpertsFL
        # Replace router with FL version that uses call_op for flaggems dispatch.
        # NOTE: shared_experts / gate / routed transforms are passed through to
        # the runner rather than stored as attributes on this layer.  Assigning
        # an nn.Module to `self.<name>` would register it as a child submodule,
        # creating a duplicate path (e.g. `<moe>._shared_experts`) that the
        # runner already owns under `<moe>.runner`.  That duplicate breaks LoRA
        # module replacement, which traverses the wrapped FusedMoE*WithLoRA and
        # finds no `_shared_experts` attribute on the wrapper.
        self._replace_router_with_fl(
            shared_experts=shared_experts,
            gate=gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
        )

    def _replace_router_with_fl(
        self,
        shared_experts=None,
        gate=None,
        routed_input_transform=None,
        routed_output_transform=None,
    ):
        """Replace the router with FL version that routes through call_op dispatch."""
        router = self.router

        if isinstance(router, GroupedTopKRouter):
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
        self.runner: MoERunnerInterface = MoERunner(
            layer_name=self.layer_name,
            moe_config=self.moe_config,
            router=self.router,
            gate=gate,
            shared_experts=shared_experts,
            quant_method=self.quant_method,
            enable_dbo=self.vllm_config.parallel_config.enable_dbo,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            # When apply_routed_scale_to_output is True, we allow
            # the scaling factor to be passed to the runner, otherwise
            # we pass 1.0 so it ends up being a nop.
            routed_scaling_factor=self._routed_scaling_factor
            if self._apply_routed_scale_to_output
            else 1.0,
        )


__all__ = ["FusedMoEFL", "UnquantizedFusedMoEMethodFL"]

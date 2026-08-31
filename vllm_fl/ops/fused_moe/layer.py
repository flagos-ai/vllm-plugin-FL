# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from vllm/model_executor/layers/fused_moe/layer.py (v0.24.0)

import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
# Save the original FusedMoE factory BEFORE any monkey-patching occurs.
# custom_ops.py patches _fused_moe_pkg.FusedMoE = FusedMoEFL at runtime,
# so calling _fused_moe_pkg.FusedMoE() inside FusedMoEFL would recurse
# infinitely.  Capturing it here breaks the cycle.
_OrigFusedMoE = _fused_moe_pkg.FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (  # noqa: E501
    CompressedTensorsW8A8Int8MoEMethod,
)

from vllm_fl.ops.fused_moe.router import replace_router_with_fl
from .fused_moe_utils import (
    TritonExpertsFL,
    select_unquantized_moe_backend_oot,
)


class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    """OOT replacement for UnquantizedFusedMoEMethod that routes computation
    through flaggems operators."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = (
            select_unquantized_moe_backend_oot(moe_config=self.moe)
        )

    @property
    def is_monolithic(self) -> bool:
        if self.moe_kernel is None:
            if self.experts_cls is None:
                return True
            return self.experts_cls.is_monolithic()
        return self.moe_kernel.is_monolithic


class CompressedTensorsW8A8Int8MoEMethodFL(
    CompressedTensorsW8A8Int8MoEMethod
):
    """FL W8A8 INT8 method preserving the upstream weight/scale lifecycle."""

    def __init__(self, weight_quant, input_quant, moe, layer_name=None):
        # Keep the official validation and INT8 backend oracle. In particular,
        # create_weights(), process_weights_after_loading(), quant config
        # construction, and apply() remain owned by the upstream method.
        super().__init__(weight_quant, input_quant, moe, layer_name)

        from vllm.platforms import current_platform

        if str(getattr(current_platform, "vendor_name", "")).lower() == "hygon":
            from vllm_fl.dispatch.backends.vendor.hygon.impl.moe.triton_experts import (
                HygonTritonExpertsFL,
            )

            self.experts_cls = HygonTritonExpertsFL
        elif current_platform.is_out_of_tree():
            from vllm_fl.utils import use_flaggems

            if use_flaggems():
                self.experts_cls = TritonExpertsFL

        # For non-OOT platforms, or when FlagGems is disabled, retain the
        # experts class selected by the official INT8 backend oracle.


class RoutedExpertsFL(RoutedExperts):
    """Select FL MoE methods before RoutedExperts creates any weights."""

    def _get_quant_method(self, prefix, quant_config, moe_config):
        quant_method = super()._get_quant_method(prefix, quant_config, moe_config)

        if isinstance(quant_method, CompressedTensorsW8A8Int8MoEMethod):
            return CompressedTensorsW8A8Int8MoEMethodFL(
                quant_method.weight_quant,
                quant_method.input_quant,
                moe_config,
                layer_name=prefix,
            )

        if isinstance(quant_method, UnquantizedFusedMoEMethod):
            return UnquantizedFusedMoEMethodFL(moe_config)

        # Quantization schemes without a validated FL method retain the exact
        # method selected by the upstream quantization config.
        return quant_method


def FusedMoEFL(*args, **kwargs) -> MoERunner:
    """
    OOT factory replacement for FusedMoE (vllm >= 0.24.0).

    In vllm 0.24.0, FusedMoE changed from a class to a factory function that
    returns a MoERunner instance.  FusedMoEFL mirrors this pattern: it
    delegates to the standard FusedMoE() factory after injecting an FL
    RoutedExperts class, then replaces the router implementation.

    Registration: op_registry_oot maps FusedMoE -> FusedMoEFL so that all
    MoE layers in a model use flaggems operators transparently.
    """
    # 1. Inject the FL RoutedExperts class before the upstream factory creates
    #    the quant method and its weights. The factory has no direct
    #    ``quant_method`` argument; ``routed_experts_cls`` is its supported
    #    construction-time extension point.
    if kwargs.get("routed_experts_cls") is None:
        kwargs["routed_experts_cls"] = RoutedExpertsFL

    # 2. Build the standard MoERunner via the upstream factory.
    #    Use _OrigFusedMoE (captured at import time, before monkey-patching)
    #    to avoid infinite recursion when custom_ops.py has already replaced
    #    _fused_moe_pkg.FusedMoE with FusedMoEFL.
    runner: MoERunner = _OrigFusedMoE(*args, **kwargs)

    # 3. Replace router _compute_routing with FL version via monkey-patch.
    #    replace_router_with_fl() patches the class method so the router
    #    instance built by FusedMoE() above uses FL dispatch without needing
    #    to re-construct the router (which would require re-passing all init
    #    args and risks signature mismatch across vllm versions).
    replace_router_with_fl()

    return runner


__all__ = [
    "CompressedTensorsW8A8Int8MoEMethodFL",
    "FusedMoEFL",
    "RoutedExpertsFL",
    "UnquantizedFusedMoEMethodFL",
]

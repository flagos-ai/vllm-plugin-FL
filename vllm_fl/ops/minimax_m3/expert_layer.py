# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Explicit FL MoE factory using the native constructor extension points.

Model code supplies its router. Expert selection is owned by the local
RoutedExperts subclass, before weights are allocated; no upstream factory or
router is patched and no caller changes a runner's private state.
"""

import inspect

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE as native_fused_moe
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from .experts import TritonExpertsFL
from vllm_fl.ops.fused_moe.fused_moe_utils import select_unquantized_moe_backend_oot


class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend_oot(
            moe_config=self.moe, triton_experts_cls=TritonExpertsFL
        )

    @property
    def is_monolithic(self):
        if self.moe_kernel is None:
            return self.experts_cls is None or self.experts_cls.is_monolithic()
        return self.moe_kernel.is_monolithic


class RoutedExpertsFL(RoutedExperts):
    """Use the existing virtual constructor hook; quantized methods keep their metadata."""

    def _get_quant_method(self, prefix, quant_config, moe_config):
        method = (
            quant_config.get_quant_method(self, prefix)
            if quant_config is not None
            else None
        )
        if method is None or isinstance(method, UnquantizedFusedMoEMethod):
            return UnquantizedFusedMoEMethodFL(moe_config)
        return method


def FusedMoEFL(*args, **kwargs):
    required = {"router", "routed_experts_cls"}
    if not required <= set(inspect.signature(native_fused_moe).parameters):
        raise RuntimeError("Unsupported vLLM MoE constructor")
    if list(inspect.signature(RoutedExperts._get_quant_method).parameters) != [
        "self",
        "prefix",
        "quant_config",
        "moe_config",
    ]:
        raise RuntimeError("Unsupported RoutedExperts constructor hook")
    kwargs.setdefault("routed_experts_cls", RoutedExpertsFL)
    return native_fused_moe(*args, **kwargs)


__all__ = ["FusedMoEFL", "UnquantizedFusedMoEMethodFL", "RoutedExpertsFL"]

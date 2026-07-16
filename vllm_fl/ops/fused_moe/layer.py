# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from vllm/model_executor/layers/fused_moe/layer.py (v0.24.0)

import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
# Save the original FusedMoE factory BEFORE any monkey-patching occurs.
# custom_ops.py patches _fused_moe_pkg.FusedMoE = FusedMoEFL at runtime,
# so calling _fused_moe_pkg.FusedMoE() inside FusedMoEFL would recurse
# infinitely.  Capturing it here breaks the cycle.
_OrigFusedMoE = _fused_moe_pkg.FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm_fl.ops.fused_moe.router import replace_router_with_fl
from .fused_moe_utils import select_unquantized_moe_backend_oot


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


def FusedMoEFL(*args, **kwargs) -> MoERunner:
    """
    OOT factory replacement for FusedMoE (vllm >= 0.24.0).

    In vllm 0.24.0, FusedMoE changed from a class to a factory function that
    returns a MoERunner instance.  FusedMoEFL mirrors this pattern: it
    delegates to the standard FusedMoE() factory and then replaces the router
    and quant_method on the returned MoERunner with FL-customised versions.

    Registration: op_registry_oot maps FusedMoE -> FusedMoEFL so that all
    MoE layers in a model use flaggems operators transparently.
    """
    # 1. Build the standard MoERunner via the upstream factory.
    #    Use _OrigFusedMoE (captured at import time, before monkey-patching)
    #    to avoid infinite recursion when custom_ops.py has already replaced
    #    _fused_moe_pkg.FusedMoE with FusedMoEFL.
    runner: MoERunner = _OrigFusedMoE(*args, **kwargs)

    # 2. Replace quant_method with FL version.
    fl_quant_method = UnquantizedFusedMoEMethodFL(runner.moe_config)
    runner._replace_quant_method(fl_quant_method)

    # 3. Replace router _compute_routing with FL version via monkey-patch.
    #    replace_router_with_fl() patches the class method so the router
    #    instance built by FusedMoE() above uses FL dispatch without needing
    #    to re-construct the router (which would require re-passing all init
    #    args and risks signature mismatch across vllm versions).
    replace_router_with_fl()

    return runner


__all__ = ["FusedMoEFL", "UnquantizedFusedMoEMethodFL"]

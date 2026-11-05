# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from vllm/model_executor/layers/fused_moe/layer.py (v0.24.0)

import vllm.model_executor.layers.fused_moe as _fused_moe_pkg

# Save the original FusedMoE factory BEFORE any monkey-patching occurs.
# custom_ops.py patches _fused_moe_pkg.FusedMoE = FusedMoEFL at runtime,
# so calling _fused_moe_pkg.FusedMoE() inside FusedMoEFL would recurse
# infinitely.  Capturing it here breaks the cycle.
_OrigFusedMoE = _fused_moe_pkg.FusedMoE
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from .fused_moe_utils import select_unquantized_moe_backend_oot
from vllm_fl.ops.fused_moe.router import replace_router_with_fl

logger = init_logger(__name__)


class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    """OOT replacement for UnquantizedFusedMoEMethod that routes computation
    through flaggems operators."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = (
            select_unquantized_moe_backend_oot(
                moe_config=self.moe,
                prefer_flaggems_experts=True,
            )
        )

    def _setup_kernel(self, layer, w13, w2):
        # Keep upstream parameter replacement, layout conversion and gate-config
        # propagation, including in-place updates of captured parameter storage.
        first_setup = self.moe_kernel is None
        super()._setup_kernel(layer, w13, w2)
        if not first_setup:
            return
        from vllm.platforms import current_platform

        if getattr(current_platform, "vendor_name", None) != "ascend":
            return
        parallel = self.moe.moe_parallel_config
        if (
            not parallel.use_ep
            or parallel.ep_size <= 1
            or not 1024 <= self.moe.hidden_dim <= 8192
        ):
            return
        from vllm_fl.distributed.device_communicators.npu_moe_runtime import (
            is_a3_device,
        )

        if not is_a3_device():
            return
        from .ascend_bf16_experts import AscendBF16DispatchedExperts

        # Explicit model SP keeps its existing AgRs prepare/finalize contract.
        if not AscendBF16DispatchedExperts._supports_parallel_config(parallel):
            return
        if not AscendBF16DispatchedExperts._supports_activation(self.moe.activation):
            raise NotImplementedError("Unsupported Ascend BF16 dispatched activation")
        from vllm.config import get_current_vllm_config
        from vllm.distributed import get_ep_group
        from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel

        from .ascend_prepare_finalize import NpuMoEPrepareAndFinalize

        experts = AscendBF16DispatchedExperts(self.moe, self.moe_quant_config)
        runtime = get_ep_group().device_communicator.get_moe_runtime(
            get_current_vllm_config(),
            device=layer.w13_weight.device,
        )
        self.moe_kernel = FusedMoEKernel(
            NpuMoEPrepareAndFinalize(runtime, self.moe.num_experts),
            experts,
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
    delegates to the standard FusedMoE() factory, replaces the router, and
    substitutes the FL experts only for unquantized MoE.

    Registration: op_registry_oot maps FusedMoE -> FusedMoEFL so that all
    MoE layers in a model use the FL router transparently.
    """
    # 1. Build the standard MoERunner via the upstream factory.
    #    Use _OrigFusedMoE (captured at import time, before monkey-patching)
    #    to avoid infinite recursion when custom_ops.py has already replaced
    #    _fused_moe_pkg.FusedMoE with FusedMoEFL.
    runner: MoERunner = _OrigFusedMoE(*args, **kwargs)

    # 2. Replace only an upstream unquantized method with the FL version.
    # Quantized methods own their weight/activation scaling metadata and must
    # remain attached to the runner.
    if isinstance(runner._quant_method, UnquantizedFusedMoEMethod):
        fl_quant_method = UnquantizedFusedMoEMethodFL(runner.moe_config)
        runner._replace_quant_method(fl_quant_method)
    else:
        logger.info_once(
            "Preserving upstream quantized MoE method %s in FusedMoEFL.",
            type(runner._quant_method).__name__,
        )

    # 3. Replace router _compute_routing with FL version via monkey-patch.
    #    replace_router_with_fl() patches the class method so the router
    #    instance built by FusedMoE() above uses FL dispatch without needing
    #    to re-construct the router (which would require re-passing all init
    #    args and risks signature mismatch across vllm versions).
    replace_router_with_fl()

    return runner


__all__ = ["FusedMoEFL", "UnquantizedFusedMoEMethodFL"]

# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from vllm/model_executor/layers/fused_moe/layer.py (v0.28.0)

import vllm.model_executor.layers.fused_moe as _fused_moe_pkg

# Save the target-version factory before custom_ops installs the FL wrapper.
_OrigFusedMoEFactory = _fused_moe_pkg.FusedMoEFactory
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.logger import init_logger

from .fused_moe_utils import select_unquantized_moe_backend_oot
from vllm_fl.ops.fused_moe.router import replace_router_with_fl


logger = init_logger(__name__)


class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    """OOT replacement for UnquantizedFusedMoEMethod that routes computation
    through flaggems operators."""

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


def FusedMoEFactoryFL(*args, **kwargs) -> MoERunner:
    """
    OOT replacement for the vLLM v0.28.0 ``FusedMoEFactory``.

    The upstream factory remains responsible for constructing the router,
    expert mapping, quantization method and runner. FL only replaces the
    unquantized method and routing hooks it owns.
    """
    runner: MoERunner = _OrigFusedMoEFactory(*args, **kwargs)

    # Quantized models must keep the quantization method chosen upstream.
    if isinstance(
        runner.routed_experts.quant_method,
        UnquantizedFusedMoEMethod,
    ):
        fl_quant_method = UnquantizedFusedMoEMethodFL(runner.moe_config)
        runner._replace_quant_method(fl_quant_method)

    # 3. Replace router _compute_routing with FL version via monkey-patch.
    #    replace_router_with_fl() patches the class method so the router
    #    instance built by FusedMoE() above uses FL dispatch without needing
    #    to re-construct the router (which would require re-passing all init
    #    args and risks signature mismatch across vllm versions).
    replace_router_with_fl()

    return runner


# Compatibility alias for downstream code that used the old FL name.
FusedMoEFL = FusedMoEFactoryFL


__all__ = [
    "FusedMoEFactoryFL",
    "FusedMoEFL",
    "UnquantizedFusedMoEMethodFL",
]

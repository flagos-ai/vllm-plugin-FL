# Copyright (c) 2025 BAAI. All rights reserved.

import torch

from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.platforms import current_platform

from vllm_fl.dispatch import CachedOp

_silu_and_mul = CachedOp("silu_and_mul")
_gelu_and_mul = CachedOp("gelu_and_mul")


class SiluAndMulFL(SiluAndMul):
    def __init__(self, *, compile_native: bool = True):
        super().__init__(compile_native=compile_native)
        self._use_amd_graph_ops = (
            getattr(current_platform, "vendor_name", None) == "amd"
        )

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_amd_graph_ops and torch.compiler.is_compiling():
            from vllm_fl.dispatch.backends.vendor.amd.graph_ops import (
                silu_and_mul_custom_op,
            )

            return silu_and_mul_custom_op(x)
        return _silu_and_mul(self, x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_oot(x)


class GeluAndMulFL(GeluAndMul):
    def __init__(self, approximate: str = "none"):
        super().__init__(approximate=approximate)
        self._use_amd_graph_ops = (
            getattr(current_platform, "vendor_name", None) == "amd"
        )

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_amd_graph_ops and torch.compiler.is_compiling():
            from vllm_fl.dispatch.backends.vendor.amd.graph_ops import (
                gelu_and_mul_custom_op,
            )

            return gelu_and_mul_custom_op(x, self.approximate)
        return _gelu_and_mul(self, x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_oot(x)


__all__ = ["SiluAndMulFL", "GeluAndMulFL"]

# Copyright (c) 2025 BAAI. All rights reserved.

import torch
from vllm.model_executor.layers.activation import SiluAndMul, GeluAndMul
from vllm_fl.dispatch import CachedOp

_silu_and_mul = CachedOp("silu_and_mul")
_gelu_and_mul = CachedOp("gelu_and_mul")


class SiluAndMulFL(SiluAndMul):
    def __init__(self):
        super().__init__()

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return _silu_and_mul(self, x)


class GeluAndMulFL(GeluAndMul):
    def __init__(self, approximate: str = "none"):
        super().__init__(approximate=approximate)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return _gelu_and_mul(self, x)


__all__ = ["SiluAndMulFL", "GeluAndMulFL"]

# Copyright (c) 2025 BAAI. All rights reserved.

import torch
from vllm.model_executor.layers.activation import SiluAndMul, GeluAndMul, SiluAndMulWithClamp
from vllm_fl.dispatch import CachedOp

_silu_and_mul = CachedOp("silu_and_mul")
_gelu_and_mul = CachedOp("gelu_and_mul")
_silu_and_mul_with_clamp = CachedOp("silu_and_mul_with_clamp")


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


class SiluAndMulWithClampFL(SiluAndMulWithClamp):
    def __init__(self, swiglu_limit: float):
        super().__init__(swiglu_limit)
        self.register_buffer(
              "_swiglu_limit_tensor",
              torch.tensor(swiglu_limit, dtype=torch.bfloat16),
              persistent=False,
          )

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return _silu_and_mul_with_clamp(
            x, self.swiglu_limit, self._swiglu_limit_tensor
        )


__all__ = ["SiluAndMulFL", "GeluAndMulFL", "SiluAndMulWithClampFL"]

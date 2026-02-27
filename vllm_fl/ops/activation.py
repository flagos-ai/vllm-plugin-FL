# Copyright (c) 2025 BAAI. All rights reserved.

import torch
from vllm.model_executor.layers.activation import SiluAndMul
from vllm_fl.dispatch import call_op


class SiluAndMulFL(SiluAndMul):
    def __init__(self):
        super().__init__()

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return call_op("silu_and_mul", self, x)

__all__ = ["SiluAndMulFL"]

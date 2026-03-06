# Copyright (c) 2025 BAAI. All rights reserved.

from vllm.model_executor.layers.activation import SiluAndMul
from vllm_fl.dispatch.method_dispatch import dispatch_method


class SiluAndMulFL(SiluAndMul):
    def __init__(self):
        super().__init__()

    forward_oot = dispatch_method("silu_and_mul")

__all__ = ["SiluAndMulFL"]

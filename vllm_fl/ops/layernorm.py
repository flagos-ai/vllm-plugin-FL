# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm_fl.dispatch.method_dispatch import dispatch_method


class RMSNormFL(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)

    forward_oot = dispatch_method("rms_norm")


__all__ = ["RMSNormFL"]

# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional, Union
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm_fl.dispatch import CachedOp

_rms_norm = CachedOp("rms_norm")


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

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if torch.compiler.is_compiling():
            from vllm_fl.dispatch.backends.vendor.amd.graph_ops import (
                rms_norm_custom_op,
            )
            # rms_norm_custom_op always returns (output, residual_or_None).
            # Caller expects a tuple when residual is passed, bare tensor otherwise.
            out, out_residual = rms_norm_custom_op(
                x, residual, self.weight, self.variance_epsilon
            )
            if residual is not None:
                return out, out_residual
            return out
        return _rms_norm(self, x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_oot(x, residual)


__all__ = ["RMSNormFL"]

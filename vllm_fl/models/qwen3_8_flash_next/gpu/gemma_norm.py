# SPDX-License-Identifier: Apache-2.0
"""Model-scoped MetaX selection of the existing FGV Gemma RMSNorm."""

import torch

from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from .runtime import is_metax


class Qwen38GemmaRMSNorm(GemmaRMSNorm):
    """Keep checkpoint names and native fallback; fuse validated BF16 QSA widths."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, *, use_fused: bool = False):
        super().__init__(hidden_size, eps)
        self.fused_gemma_enabled = bool(use_fused and is_metax())

    def forward_native(self, x, residual=None):
        if (
            self.fused_gemma_enabled
            and residual is None
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.ndim >= 1
            and x.shape[-1] in (128, 256)
            and self.weight.dtype == x.dtype
            and self.weight.device == x.device
        ):
            from flaggems_vllm import gemma_rms_norm

            # Q/K projections can expose strided views. The provider's public
            # contract is contiguous; preserve all rows with an explicit copy.
            return gemma_rms_norm(
                x.contiguous(), self.weight.contiguous(), self.variance_epsilon
            )
        return super().forward_native(x, residual)

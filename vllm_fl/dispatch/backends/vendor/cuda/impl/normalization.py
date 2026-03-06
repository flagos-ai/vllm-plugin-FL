# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_cuda(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using CUDA.

    Uses vLLM's optimized CUDA kernel when available.

    Args:
        self: The calling instance (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor to add before normalization

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    from vllm._custom_ops import rms_norm as vllm_rms_norm
    from vllm._custom_ops import fused_add_rms_norm as vllm_fused_add_rms_norm

    # Get weight and epsilon from self
    weight = self.weight
    epsilon = self.variance_epsilon

    if residual is not None:
        vllm_fused_add_rms_norm(x, residual, weight, epsilon)
        return x, residual
    else:
        out = torch.empty_like(x)
        vllm_rms_norm(out, x, weight, epsilon)
        return out

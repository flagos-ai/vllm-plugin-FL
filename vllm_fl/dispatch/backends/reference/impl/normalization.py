# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference normalization operator implementations using PyTorch.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rmsnorm_torch(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    epsilon: float,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using PyTorch.

    Args:
        x: Input tensor
        residual: Optional residual tensor
        weight: Normalization weight
        epsilon: Small constant for numerical stability

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    if residual is not None:
        x = x + residual
        residual = x

    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    output = weight * x

    if residual is not None:
        return output, residual
    return output

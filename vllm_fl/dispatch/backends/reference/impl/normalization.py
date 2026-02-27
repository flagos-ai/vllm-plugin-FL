# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference normalization operator implementations using PyTorch.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_torch(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using PyTorch.

    Args:
        obj: The calling obj (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    # Get weight and epsilon from obj
    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        x = x + residual
        residual = x

    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    output = weight * x

    if residual is not None:
        return output, residual
    return output

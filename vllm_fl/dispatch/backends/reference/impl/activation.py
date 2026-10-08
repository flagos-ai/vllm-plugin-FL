# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference activation operator implementations using PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul_torch(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using PyTorch.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return F.silu(x1) * x2


def gelu_and_mul_torch(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using PyTorch.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return F.gelu(x1, approximate=approximate) * x2


def silu_and_mul_with_clamp(x, limit, alpha=1.0, beta=0.0):
    """Apply the bounded SwiGLU capability using PyTorch operations."""
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)

# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA activation operator implementations.
"""

from __future__ import annotations

import torch


def silu_and_mul_cuda(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using CUDA.

    Uses vLLM's optimized CUDA kernel when available.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


def gelu_and_mul_cuda(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using CUDA.

    Uses vLLM's optimized CUDA kernel.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    if approximate == "tanh":
        torch.ops._C.gelu_tanh_and_mul(out, x)
    else:
        torch.ops._C.gelu_and_mul(out, x)
    return out


def silu_and_mul_with_clamp_cuda(x: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """
    SiLU activation with clamping followed by element-wise multiplication.

    Computes:
        gate = clamp(x[..., :d], max=swiglu_limit)
        up   = clamp(x[..., d:], min=-swiglu_limit, max=swiglu_limit)
        out  = silu(gate) * up

    Args:
        x: Input tensor of shape [..., 2*d]
        swiglu_limit: Clamping threshold

    Returns:
        Output tensor of shape [..., d]
    """
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul_with_clamp(out, x, swiglu_limit)
    return out

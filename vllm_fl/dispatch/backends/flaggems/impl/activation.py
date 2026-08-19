# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems activation operator implementations.
"""

from __future__ import annotations

import torch

import vllm_fl.envs as fl_envs


def silu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        from flaggems_vllm.ops.silu_and_mul import silu_and_mul
    else:
        from flag_gems import silu_and_mul

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return silu_and_mul(x1, x2)


def gelu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        from flaggems_vllm.ops.gelu_and_mul import gelu_and_mul
    else:
        from flag_gems import gelu_and_mul

    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gelu_and_mul(x1, x2, approximate)


def silu_and_mul_with_clamp_flaggems(x: torch.Tensor, swiglu_limit: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation with clamping followed by element-wise multiplication using FlagGems.

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
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        from flaggems_vllm.ops.silu_and_mul_with_clamp import silu_and_mul_with_clamp_kernel
    else:
        from flag_gems.fused.silu_and_mul_with_clamp import silu_and_mul_with_clamp_kernel

    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    return silu_and_mul_with_clamp_kernel(gate, up, swiglu_limit)

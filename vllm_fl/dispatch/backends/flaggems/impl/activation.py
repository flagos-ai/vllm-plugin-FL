# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems activation operator implementations.
"""

from __future__ import annotations

import math

import torch
from flag_gems.fused.silu_and_mul_with_clamp import silu_and_mul_with_clamp_kernel


@torch.library.custom_op(
    "vllm_fl::flag_gems_silu_and_mul_with_clamp", mutates_args=()
)
def _flag_gems_silu_and_mul_with_clamp(
    gate: torch.Tensor, up: torch.Tensor, limit: torch.Tensor
) -> torch.Tensor:
    return silu_and_mul_with_clamp_kernel(gate, up, limit)


@_flag_gems_silu_and_mul_with_clamp.register_fake
def _flag_gems_silu_and_mul_with_clamp_fake(
    gate: torch.Tensor, up: torch.Tensor, limit: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(gate)


_LIMIT_TENSORS: dict[tuple[torch.device, torch.dtype, float], torch.Tensor] = {}


def silu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from flag_gems.modules.activation import gems_silu_and_mul

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gems_silu_and_mul(x1, x2)


def gelu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from flag_gems.fused import gelu_and_mul

    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gelu_and_mul(x1, x2, approximate)


def silu_and_mul_with_clamp(x, limit, alpha=1.0, beta=0.0):
    """Apply the graph-safe bounded SwiGLU capability provided by FlagGems."""
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.shape[-1] % 2
        or alpha != 1.0
        or beta != 0.0
        or not math.isfinite(float(limit))
        or limit <= 0
    ):
        raise ValueError("Unsupported bounded activation inputs")
    key = (x.device, x.dtype, float(limit))
    scalar = _LIMIT_TENSORS.get(key)
    if scalar is None:
        if torch.compiler.is_compiling():
            scalar = torch.tensor(float(limit), device=x.device, dtype=x.dtype)
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Bounded SwiGLU needs eager warmup before graph capture"
                )
            scalar = torch.tensor(float(limit), device=x.device, dtype=x.dtype)
            _LIMIT_TENSORS[key] = scalar
    gate, up = x.chunk(2, dim=-1)
    return _flag_gems_silu_and_mul_with_clamp(gate, up, scalar)

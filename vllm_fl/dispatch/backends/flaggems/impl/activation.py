# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems activation operator implementations.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform

if current_platform.device_type == "musa":
    from .musa_ops import silu_and_mul as musa_silu_and_mul


@torch.library.custom_op("vllm_fl::iluvatar_gems_silu_and_mul", mutates_args=())
def _iluvatar_gems_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    # Keep FlagGems Python dtype inference outside Dynamo tracing.
    from flaggems_vllm import silu_and_mul

    d = x.shape[-1] // 2
    return silu_and_mul(x[..., :d], x[..., d:])


@_iluvatar_gems_silu_and_mul.register_fake
def _iluvatar_gems_silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def silu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    if getattr(current_platform, "vendor_name", None) == "iluvatar":
        return _iluvatar_gems_silu_and_mul(x)
    if (
        current_platform.device_type == "musa"
        and x.device.type == "musa"
        and torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
    ):
        return musa_silu_and_mul(x)

    from flaggems_vllm import silu_and_mul as gems_silu_and_mul

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
    from flaggems_vllm import gelu_and_mul

    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gelu_and_mul(x1, x2, approximate)

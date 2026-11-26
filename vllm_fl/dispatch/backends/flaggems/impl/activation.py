# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems activation operator implementations.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform

if current_platform.device_type == "musa":
    from .musa_ops import silu_and_mul as musa_silu_and_mul


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
        from vllm_fl.dispatch.backends.vendor.iluvatar.activation import (
            _iluvatar_gems_silu_and_mul,
        )

        return _iluvatar_gems_silu_and_mul(x)
    if (
        current_platform.device_type == "musa"
        and x.device.type == "musa"
        and torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
    ):
        return musa_silu_and_mul(x)

    # FlagGems ships an enflame-specialized silu_and_mul (flat kernel) while
    # flaggems_vllm only has the generic pointwise version, which corrupts the
    # tail rows of mixed text+image batches on GCU.  Prefer the specialized
    # FlagGems implementation here; it is also faster on the flat path.
    from flag_gems import silu_and_mul as gems_silu_and_mul

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

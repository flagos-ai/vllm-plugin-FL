# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems activation operator implementations.
"""

from __future__ import annotations

import flag_gems
import torch

from flag_gems.fused import gelu_and_mul as _eager_gelu_and_mul
from flag_gems.modules.activation import gems_silu_and_mul as _eager_silu_and_mul

# Resolve and materialize compiler-visible FlagGems families during backend
# registration, before Dynamo traces a frozen CachedOp.  No alternate
# mathematical kernel is introduced: eager and compiled execution retain the
# original generated PointwiseDynamic JITFunctions.
from flag_gems.pt2.pointwise_dynamic import (
    ACTIVATION_POINTWISE_FAMILIES,
    gelu_and_mul_pointwise,
    materialize_pointwise_family_plans,
    silu_and_mul_pointwise,
    silu_and_mul_with_clamp_pointwise,
)


# The four common sources captured by the transparent adapter are the NVIDIA
# PointwiseDynamic families.  Other FlagGems vendors may replace these exports
# with architecture-specific implementations, so they retain the original
# eager path until that vendor explicitly supplies equivalent PT2 family specs.
_USE_TRANSPARENT_POINTWISE = flag_gems.vendor_name == "nvidia"

if _USE_TRANSPARENT_POINTWISE:
    # Freeze structural plans while the FlagGems backend is registered, before
    # any Dynamo capture.  Four generated families (SiLU,
    # GELU-none, GELU-tanh, clamped SiLU) each expose six dtype/layout plans.
    # The binary families retain the native contiguous->rank-1 and split->rank-2
    # materializations; clamped SiLU retains rank 2 because of its scalar
    # broadcast.  No Tensor or token-count value is retained.
    materialize_pointwise_family_plans(
        ACTIVATION_POINTWISE_FAMILIES,
        ranks=(2,),
        dtypes=(torch.float16, torch.bfloat16, torch.float32),
        layout_classes=("contiguous_c", "split_last_dim_c"),
    )


def silu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    if _USE_TRANSPARENT_POINTWISE:
        return silu_and_mul_pointwise(x1, x2)
    return _eager_silu_and_mul(x1, x2)


def gelu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    if _USE_TRANSPARENT_POINTWISE:
        return gelu_and_mul_pointwise(x1, x2, approximate)
    return _eager_gelu_and_mul(x1, x2, approximate)


def silu_and_mul_with_clamp_flaggems(
    x: torch.Tensor,
    swiglu_limit: float,
    swiglu_limit_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    SiLU activation with clamping followed by element-wise multiplication using FlagGems.

    Computes:
        gate = clamp(x[..., :d], max=swiglu_limit)
        up   = clamp(x[..., d:], min=-swiglu_limit, max=swiglu_limit)
        out  = silu(gate) * up

    Args:
        x: Input tensor of shape [..., 2*d]
        swiglu_limit: Python-side threshold retained for vLLM API parity.
        swiglu_limit_tensor: Device tensor carrying the same threshold into the
            original FlagGems Triton kernel.

    Returns:
        Output tensor of shape [..., d]
    """
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    if _USE_TRANSPARENT_POINTWISE:
        return silu_and_mul_with_clamp_pointwise(gate, up, swiglu_limit_tensor)
    return flag_gems.silu_and_mul_with_clamp(gate, up, swiglu_limit)

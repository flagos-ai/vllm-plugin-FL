# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

import flag_gems
from flag_gems.config import use_c_extension
from flag_gems.modules.normalization import gems_rms_forward
from flag_gems.pt2.rms_norm import rms_norm as _pt2_rms_norm


# The Python adapter captures the common/NVIDIA Triton objects.  Other vendors
# may replace RMSNorm, while a C++ installation deliberately selects its
# torch.ops implementation.  Preserve both choices instead of bypassing them.
_USE_TRANSPARENT_RMS_NORM = (
    flag_gems.vendor_name == "nvidia" and not use_c_extension
)


def rms_norm_flaggems(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using FlagGems.

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

    if _USE_TRANSPARENT_RMS_NORM:
        variance_size_override = getattr(obj, "variance_size_override", None)
        if variance_size_override is not None:
            raise RuntimeError(
                "FlagGems transparent RMSNorm does not support vLLM "
                "variance_size_override; select a compatible backend"
            )
        return _pt2_rms_norm(x, residual, weight, epsilon)

    return gems_rms_forward(x, residual, weight, epsilon)

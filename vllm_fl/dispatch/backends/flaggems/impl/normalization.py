# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm.platforms import current_platform

if current_platform.device_type == "musa":
    from .musa_ops import rms_norm as musa_rms_norm


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
    if (
        current_platform.device_type == "musa"
        and x.device.type == "musa"
        and torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
    ):
        from flag_gems.modules.normalization import use_c_extension

        if not use_c_extension:
            return musa_rms_norm(x, residual, obj.weight, obj.variance_epsilon)

    from flag_gems.modules.normalization import gems_rms_forward

    # Get weight and epsilon from obj
    weight = obj.weight
    epsilon = obj.variance_epsilon

    return gems_rms_forward(x, residual, weight, epsilon)

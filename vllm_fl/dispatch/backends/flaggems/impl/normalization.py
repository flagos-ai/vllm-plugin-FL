# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


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
    from vllm.platforms import current_platform

    weight = obj.weight
    epsilon = obj.variance_epsilon
    if getattr(current_platform, "vendor_name", "") == "metax":
        from flag_gems.fused.metax_fast_ops import fast_rms_forward

        return fast_rms_forward(x, residual, weight, epsilon)
    from flag_gems.modules.normalization import gems_rms_forward

    return gems_rms_forward(x, residual, weight, epsilon)

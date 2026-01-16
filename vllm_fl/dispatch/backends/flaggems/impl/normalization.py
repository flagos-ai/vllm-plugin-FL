# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rmsnorm_flaggems(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    epsilon: float,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using FlagGems.

    Args:
        x: Input tensor
        residual: Optional residual tensor
        weight: Normalization weight
        epsilon: Small constant for numerical stability

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    from flag_gems.modules.normalization import gems_rms_forward

    return gems_rms_forward(x, residual, weight, epsilon)

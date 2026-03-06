# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_flaggems(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using FlagGems.

    Args:
        self: The calling instance (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    from flag_gems.modules.normalization import gems_rms_forward

    # Get weight and epsilon from self
    weight = self.weight
    epsilon = self.variance_epsilon

    return gems_rms_forward(x, residual, weight, epsilon)

# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_ascend(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using Ascend NPU.

    Args:
        self: The calling instance (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor to add before normalization

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    import torch_npu

    # Get weight and epsilon from self
    weight = self.weight
    epsilon = self.variance_epsilon

    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, weight, epsilon)
        return x, residual

    x, _ = torch_npu.npu_rms_norm(x, weight, epsilon)
    return x

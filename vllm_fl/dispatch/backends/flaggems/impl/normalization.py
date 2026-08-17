# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems normalization operator implementations.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

import vllm_fl.envs as fl_envs

logger = logging.getLogger(__name__)


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
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        logger.warning_once(
            "rms_norm_forward is not available in flaggems_vllm, "
            "falling back to flag_gems.rms_norm_forward"
        )

    from flag_gems import rms_norm_forward as gems_rms_forward

    # Get weight and epsilon from obj
    weight = obj.weight
    epsilon = obj.variance_epsilon

    return gems_rms_forward(x, residual, weight, epsilon)

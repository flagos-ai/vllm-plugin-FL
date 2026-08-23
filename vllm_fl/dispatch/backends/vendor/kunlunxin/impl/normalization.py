# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""
Kunlunxin normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_kunlunxin(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization.

    Args:
        obj: The calling obj (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor to add before normalization

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    import xtorch_ops

    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        # fused_add_rms_norm: output = rmsnorm(x + residual), residual_output = x + residual
        xtorch_ops.add_rmsnorm(
            x=x,
            y=residual,
            weight=weight,
            output=x,
            eps=epsilon,
            residual_output=residual,
            store_output_before_norm=True,
        )
        return x, residual

    # rms_norm_with_stride can handle strided 2D inputs
    if x.dim() == 2 and x.stride(-2) != x.size(-1):
        input_maybe_contiguous = x
    else:
        input_maybe_contiguous = x.contiguous()

    out = torch.empty_like(x)
    xtorch_ops.rmsnorm(input_maybe_contiguous, weight, out, epsilon)
    return out

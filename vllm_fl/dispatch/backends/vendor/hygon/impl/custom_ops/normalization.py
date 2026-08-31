# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon normalization implementations backed by LightOp."""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_hygon(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """RMS normalization using the LightOp APIs selected by vllm_hcu."""
    try:
        from lightop.norm import fused_add_rms_norm, rmsnorm_forward_autograd
    except ImportError:
        from lightop import fused_add_rms_norm
        from lightop.op import rmsnorm_forward_autograd

    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        fused_add_rms_norm(x, residual, weight, epsilon)
        return x, residual

    return rmsnorm_forward_autograd(x, weight, epsilon, obj.training)

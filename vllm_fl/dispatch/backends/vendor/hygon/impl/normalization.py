# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon normalization operator fallbacks backed by vLLM native ops."""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_hygon(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """RMS normalization using vLLM custom ops."""
    from vllm._custom_ops import fused_add_rms_norm as vllm_fused_add_rms_norm
    from vllm._custom_ops import rms_norm as vllm_rms_norm

    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        vllm_fused_add_rms_norm(x, residual, weight, epsilon)
        return x, residual

    out = torch.empty_like(x)
    vllm_rms_norm(out, x, weight, epsilon)
    return out

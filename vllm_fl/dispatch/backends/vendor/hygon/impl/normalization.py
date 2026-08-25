"""Hygon normalization fallbacks backed by vLLM native ops."""

from __future__ import annotations

import torch


def rms_norm_hygon(obj, x: torch.Tensor, residual=None):
    from vllm._custom_ops import fused_add_rms_norm, rms_norm

    if residual is not None:
        fused_add_rms_norm(x, residual, obj.weight, obj.variance_epsilon)
        return x, residual

    out = torch.empty_like(x)
    rms_norm(out, x, obj.weight, obj.variance_epsilon)
    return out

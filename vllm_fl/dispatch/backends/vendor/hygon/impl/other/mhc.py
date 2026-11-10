# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon implementations for DeepSeek-V4 MHC.

Hygon vLLM does not use vLLM's generic TileLang prenorm reduction on ROCm.
That kernel assumes a 32-lane warp, while Hygon executes a 64-lane wavefront.
Prefer the same wave64 AITER operators as Hygon vLLM, with vLLM's PyTorch MHC
math as the fallback when those optional vendor operators are unavailable.
"""

from __future__ import annotations

from functools import cache
import importlib
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


@cache
def _get_aiter_mhc_ops() -> tuple[Any, Any] | None:
    """Return the two Hygon vLLM AITER MHC leaves when installed."""
    try:
        aiter_tilelang = importlib.import_module("aiter.ops.tilelang")
    except (ImportError, AttributeError):
        logger.info("Hygon AITER MHC ops unavailable; using PyTorch fallback")
        return None

    pre = getattr(aiter_tilelang, "mhc_pre_big_fuse", None)
    post = getattr(aiter_tilelang, "mhc_post_fwd", None)
    if not callable(pre) or not callable(post):
        logger.info("Installed AITER lacks Hygon MHC ops; using PyTorch fallback")
        return None
    logger.info("Using Hygon AITER MHC pre/post ops")
    return pre, post


def get_mhc_backend_name() -> str:
    """Expose the selected backend for diagnostics and operator tests."""
    return "aiter" if _get_aiter_mhc_ops() is not None else "torch"


def _apply_optional_rms_norm(
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    """Match the optional norm fused into vLLM's TileLang MHC pre output."""
    if norm_weight is None:
        return layer_input

    layer_input_fp32 = layer_input.float()
    normalized = layer_input_fp32 * torch.rsqrt(
        layer_input_fp32.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    return (normalized * norm_weight.float()).to(layer_input.dtype)


def mhc_pre_hygon(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Hygon vLLM's AITER MHC-pre op or its ROCm reference fallback."""
    del n_splits
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)

    aiter_ops = _get_aiter_mhc_ops()
    if aiter_ops is not None:
        pre_op, _ = aiter_ops
        post_mix, comb_mix, layer_input = pre_op(
            residual=residual_flat,
            fn=fn,
            mhc_scale=hc_scale,
            mhc_base=hc_base,
            rms_eps=rms_eps,
            mhc_pre_eps=hc_pre_eps,
            mhc_sinkhorn_eps=hc_sinkhorn_eps,
            mhc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        post_mix = post_mix.view(*outer_shape, hc_mult, 1)
        comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
        layer_input = layer_input.view(*outer_shape, hidden_size)
    else:
        from vllm.model_executor.kernels import mhc as mhc_kernels

        post_mix, comb_mix, layer_input = mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    layer_input = _apply_optional_rms_norm(layer_input, norm_weight, norm_eps)
    return post_mix, comb_mix, layer_input


def mhc_post_hygon(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Run Hygon vLLM's AITER MHC-post op or its ROCm reference fallback."""
    aiter_ops = _get_aiter_mhc_ops()
    if aiter_ops is not None:
        _, post_op = aiter_ops
        hc_mult = residual.shape[-2]
        hidden_size = residual.shape[-1]
        outer_shape = residual.shape[:-2]
        result = post_op(
            x.reshape(-1, hidden_size).contiguous(),
            residual.reshape(-1, hc_mult, hidden_size),
            post_layer_mix.reshape(-1, hc_mult).contiguous(),
            comb_res_mix.reshape(-1, hc_mult, hc_mult).contiguous(),
        )
        return result.view(*outer_shape, hc_mult, hidden_size)

    from vllm.model_executor.kernels import mhc as mhc_kernels

    return mhc_kernels.mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)


def mhc_fused_post_pre_hygon(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose fused post+pre using the validated Hygon ROCm math."""
    del tile_n
    residual_cur = mhc_post_hygon(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )
    post_mix, comb_mix, layer_input = mhc_pre_hygon(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        norm_weight,
        norm_eps,
    )
    return residual_cur, post_mix, comb_mix, layer_input

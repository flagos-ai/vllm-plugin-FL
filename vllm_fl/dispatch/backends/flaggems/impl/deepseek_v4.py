# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Opt-in DS adapters for pinned FG/FGV APIs, including opaque graph boundaries.

This module does not contain Triton kernels, TP communication or startup hooks.
MHC pre uses the pinned FGV #755 MetaX entry, with its original shape fallback.
Its private API dependency is intentional until that library exposes this contract.
"""
import torch
from vllm import _custom_ops
from flag_gems import hc_head_fused_kernel as _native_hc_head
from flag_gems.ops.dsv4_masked_fill import masked_fill_ as _mask_fast
from flaggems_vllm.ops.dsv4_int8_quant import quant as _quant
from flaggems_vllm import moe_align_block_size_no_tle as selected_align
from flaggems_vllm.ops.silu_and_mul_with_clamp import silu_and_mul_with_clamp_kernel
from flaggems_vllm.runtime.backend._metax.ops.mhc_pre import run as _run_pr755
from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre as _legacy_mhc_pre
from flaggems_vllm import mhc_post as _fgv_mhc_post
from flaggems_vllm import combine_topk_swa_indices as _fgv_combine_topk_swa_indices
from flaggems_vllm import (
    dequantize_and_gather_k_cache as _fgv_dequantize_and_gather_k_cache,
)
from flaggems_vllm import fused_q_kv_rmsnorm as _fgv_fused_q_kv_rmsnorm

_LIB = torch.library.Library("dsv4_fg_explicit", "FRAGMENT")


def _mask_impl(x, mask):
    eligible = (
        x.ndim == 2
        and x.shape[1] == 4096
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float32)
        and mask.dtype == torch.bool
        and mask.ndim == 2
        and mask.shape == (x.shape[0], 1)
        and mask.is_contiguous()
    )
    if eligible:
        _mask_fast(x, mask, 0)
    else:
        x.masked_fill_(mask, 0)


def _quant_impl(x):
    q, scale, _ = _quant(x)
    return q, scale


def _silu_impl(x, limit):
    d = x.shape[-1] // 2
    return silu_and_mul_with_clamp_kernel(x[..., :d], x[..., d:], limit)


_LIB.define("masked_fill_zero_(Tensor(a!) x, Tensor mask) -> ()")
_LIB.impl("masked_fill_zero_", _mask_impl, "CUDA")
_LIB.define("dynamic_int8(Tensor x) -> (Tensor, Tensor)")
_LIB.impl("dynamic_int8", _quant_impl, "CUDA")
_LIB.define("silu_clamp(Tensor x, Tensor limit) -> Tensor")
_LIB.impl("silu_clamp", _silu_impl, "CUDA")


@torch.library.register_fake("dsv4_fg_explicit::masked_fill_zero_")
def _mask_fake(x, mask):
    return None


@torch.library.register_fake("dsv4_fg_explicit::dynamic_int8")
def _quant_fake(x):
    return torch.empty_like(x, dtype=torch.int8), torch.empty(
        (x.shape[0], 1), device=x.device, dtype=torch.float32
    )


@torch.library.register_fake("dsv4_fg_explicit::silu_clamp")
def _silu_fake(x, limit):
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device
    )


def selected_mask(x, mask, value=0):
    if value != 0:
        raise ValueError("This embedding operator only supports zero fill")
    torch.ops.dsv4_fg_explicit.masked_fill_zero_(x, mask)
    return x


def scaled_int8_quant(x, scale=None, azp=None, symmetric=True):
    if (
        scale is None
        and azp is None
        and symmetric is True
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[1] <= 16384
    ):
        q, s = torch.ops.dsv4_fg_explicit.dynamic_int8(x)
        return q, s, None
    return _custom_ops.scaled_int8_quant(x, scale, azp, symmetric)


def silu_clamp(x, limit):
    return torch.ops.dsv4_fg_explicit.silu_clamp(x, limit)


@torch.library.custom_op("vllm_fl::dsv4_hc_head", mutates_args=("out",))
def hc_head(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    _native_hc_head(
        hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
    )


@hc_head.register_fake
def _hc_head_fake(
    hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
):
    return None


@torch.library.custom_op("vllm_fl::dsv4_mhc_pre", mutates_args=())
def _mhc_pre_metax(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eligible = (
        residual.ndim >= 3
        and residual.shape[-2:] == (4, 4096)
        and residual.dtype == torch.bfloat16
        and residual.is_cuda
        and residual.is_contiguous()
        and fn.shape == (24, 16384)
        and fn.dtype == torch.float32
        and fn.is_contiguous()
        and hc_scale.shape == (3,)
        and hc_base.shape == (24,)
        and hc_scale.dtype == hc_base.dtype == torch.float32
        and hc_scale.is_contiguous()
        and hc_base.is_contiguous()
        and fn.device == hc_scale.device == hc_base.device == residual.device
        and sinkhorn_repeat >= 1
    )
    if not eligible:
        return _legacy_mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )
    outer = residual.shape[:-2]
    flat = residual.reshape(-1, 4, 4096)
    if flat.shape[0] == 0:
        return (
            torch.empty((*outer, 4, 1), device=residual.device, dtype=torch.float32),
            torch.empty((*outer, 4, 4), device=residual.device, dtype=torch.float32),
            torch.empty((*outer, 4096), device=residual.device, dtype=torch.bfloat16),
        )
    post, comb, layer = _run_pr755(
        flat,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return post.view(*outer, 4, 1), comb.view(*outer, 4, 4), layer.view(*outer, 4096)


@_mhc_pre_metax.register_fake
def _mhc_pre_metax_fake(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits,
):
    outer, hc, h = residual.shape[:-2], residual.shape[-2], residual.shape[-1]
    return (
        torch.empty((*outer, hc, 1), device=residual.device, dtype=torch.float32),
        torch.empty((*outer, hc, hc), device=residual.device, dtype=torch.float32),
        torch.empty((*outer, h), device=residual.device, dtype=torch.bfloat16),
    )


def mhc_pre(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _mhc_pre_metax(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )


def combine_topk_swa_indices_flaggems(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> None:
    """FlagGems implementation of combine_topk_swa_indices."""
    return _fgv_combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )


def dequantize_and_gather_k_cache_flaggems(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """FlagGems implementation of dequantize_and_gather_k_cache."""
    _fgv_dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )


def fused_q_kv_rmsnorm_flaggems(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlagGems implementation of fused_q_kv_rmsnorm."""
    return _fgv_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)


def silu_and_mul_with_clamp(x, swiglu_limit, swiglu_limit_tensor):
    return silu_clamp(x, swiglu_limit_tensor)


def mhc_post(x, residual, post, comb):
    return _fgv_mhc_post(x, residual, post, comb)


def operator_impls():
    """Only the capabilities whose accepted DS provider differs from generic FG."""
    return {
        "silu_and_mul_with_clamp": silu_and_mul_with_clamp,
        "combine_topk_swa_indices": combine_topk_swa_indices_flaggems,
        "dequantize_and_gather_k_cache": dequantize_and_gather_k_cache_flaggems,
        "fused_q_kv_rmsnorm": fused_q_kv_rmsnorm_flaggems,
        "mhc_pre": mhc_pre,
        "mhc_post": mhc_post,
        "hc_head_fused_kernel": hc_head,
    }

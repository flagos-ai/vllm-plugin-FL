# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems implementation of DeepSeek-V4-specific operators."""

from __future__ import annotations

import importlib
import weakref

import torch
from flag_gems.runtime import torch_device_fn

from vllm_fl.ops.deepseek_v4_int8_woa import (
    fused_inv_rope_quant_int8_triton,
)


def deepseek_v4_inv_rope_quant_int8_flaggems(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused backend-neutral Triton kernel under FlagGems' device guard."""
    with torch_device_fn.device(o.device):
        return fused_inv_rope_quant_int8_triton(
            o,
            positions,
            cos_sin_cache,
            n_groups,
            heads_per_group,
            nope_dim,
            rope_dim,
        )


def deepseek_v4_inv_rope_quant_fp8_flaggems(*args):
    from vllm_fl.dispatch.backends.vendor.cuda.impl.deepseek_v4 import (
        deepseek_v4_inv_rope_quant_fp8_cuda,
    )

    with torch_device_fn.device(args[0].device):
        return deepseek_v4_inv_rope_quant_fp8_cuda(*args)


def deepseek_v4_int8_scaled_mm_flaggems(
    x, weight, scale_a, scale_b, out_dtype, bias=None
):
    from flag_gems import scaled_mm

    with torch_device_fn.device(x.device):
        return scaled_mm(x, weight, scale_a, scale_b, bias=bias, out_dtype=out_dtype)


def _reference(name, *args):
    from vllm_fl.dispatch.backends.reference.impl import deepseek_v4

    with torch_device_fn.device(args[0].device):
        return getattr(deepseek_v4, name)(*args)


_FN_BF16_CACHE_BY_ID = {}


def _get_fn_bf16_cached_by_id(fn):
    if fn.requires_grad or torch.is_grad_enabled():
        return fn.to(dtype=torch.bfloat16)
    key = id(fn)
    version = fn._version
    cached = _FN_BF16_CACHE_BY_ID.get(key)
    if cached is not None and cached[0]() is fn and cached[1] == version:
        return cached[2]
    fn_bf16 = fn.to(dtype=torch.bfloat16)

    def remove(ref, cache_key=key):
        current = _FN_BF16_CACHE_BY_ID.get(cache_key)
        if current is not None and current[0] is ref:
            _FN_BF16_CACHE_BY_ID.pop(cache_key, None)

    _FN_BF16_CACHE_BY_ID[key] = (weakref.ref(fn, remove), version, fn_bf16)
    return fn_bf16


def _flaggems_mhc_pre_impl(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=1e-6,
):
    """Run FlagGems' fused MHC pre kernel and the optional layer RMSNorm."""
    from flag_gems import rms_norm

    mhc_pre_module = importlib.import_module("flag_gems.fused.mhc.mhc_pre")
    mhc_pre_module._get_fn_bf16_cached = _get_fn_bf16_cached_by_id
    from flag_gems.fused import mhc_pre

    with torch_device_fn.device(residual.device):
        post_mix, comb_mix, layer_input = mhc_pre(
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
        if norm_weight is not None:
            layer_input = rms_norm(
                layer_input,
                tuple(norm_weight.shape),
                norm_weight,
                norm_eps,
            )
    return post_mix, comb_mix, layer_input


def _flaggems_mhc_post_impl(*args):
    from flag_gems.fused import mhc_post

    with torch_device_fn.device(args[0].device):
        return mhc_post(*args)


def _flaggems_mhc_fused_post_pre_impl(
    x,
    residual,
    post_mix,
    comb_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=1e-6,
):
    del tile_n
    residual_cur = _flaggems_mhc_post_impl(x, residual, post_mix, comb_mix)
    post_cur, comb_cur, layer_input = _flaggems_mhc_pre_impl(
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
    return residual_cur, post_cur, comb_cur, layer_input


@torch.library.custom_op(
    "vllm_fl::flaggems_deepseek_v4_mhc_pre",
    mutates_args=(),
    device_types="cuda",
)
def _flaggems_mhc_pre_op(
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
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _flaggems_mhc_pre_impl(
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
        norm_weight,
        norm_eps,
    )


@torch.library.custom_op(
    "vllm_fl::flaggems_deepseek_v4_mhc_post",
    mutates_args=(),
    device_types="cuda",
)
def _flaggems_mhc_post_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    return _flaggems_mhc_post_impl(x, residual, post_mix, comb_mix)


@torch.library.custom_op(
    "vllm_fl::flaggems_deepseek_v4_mhc_fused_post_pre",
    mutates_args=(),
    device_types="cuda",
)
def _flaggems_mhc_fused_post_pre_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    tile_n: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _flaggems_mhc_fused_post_pre_impl(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
        norm_weight,
        norm_eps,
    )


def _fake_mhc_pre_outputs(residual: torch.Tensor):
    outer = residual.shape[:-2]
    hc, hidden = residual.shape[-2:]
    return (
        residual.new_empty((*outer, hc, 1), dtype=torch.float32),
        residual.new_empty((*outer, hc, hc), dtype=torch.float32),
        residual.new_empty((*outer, hidden)),
    )


@_flaggems_mhc_pre_op.register_fake
def _flaggems_mhc_pre_fake(
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
    norm_weight,
    norm_eps,
):
    del (
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
    return _fake_mhc_pre_outputs(residual)


@_flaggems_mhc_post_op.register_fake
def _flaggems_mhc_post_fake(x, residual, post_mix, comb_mix):
    del x, post_mix, comb_mix
    return torch.empty_like(residual)


@_flaggems_mhc_fused_post_pre_op.register_fake
def _flaggems_mhc_fused_post_pre_fake(
    x,
    residual,
    post_mix,
    comb_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits,
    tile_n,
    norm_weight,
    norm_eps,
):
    del (
        x,
        post_mix,
        comb_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
        norm_weight,
        norm_eps,
    )
    post_cur, comb_cur, layer_input = _fake_mhc_pre_outputs(residual)
    return torch.empty_like(residual), post_cur, comb_cur, layer_input


def deepseek_v4_mhc_pre_flaggems(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=1e-6,
):
    return _flaggems_mhc_pre_op(
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
        norm_weight,
        norm_eps,
    )


def deepseek_v4_mhc_post_flaggems(x, residual, post_mix, comb_mix):
    return _flaggems_mhc_post_op(x, residual, post_mix, comb_mix)


def deepseek_v4_mhc_fused_post_pre_flaggems(
    x,
    residual,
    post_mix,
    comb_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=1e-6,
):
    return _flaggems_mhc_fused_post_pre_op(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
        norm_weight,
        norm_eps,
    )


def deepseek_v4_hc_head_flaggems(*args):
    return _reference("deepseek_v4_hc_head_torch", *args)


def _reference_kwargs(name, *args, **kwargs):
    from vllm_fl.dispatch.backends.reference.impl import deepseek_v4

    tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if tensor is None:
        tensor = next(
            (value for value in kwargs.values() if isinstance(value, torch.Tensor)),
            None,
        )
    if tensor is None:
        return getattr(deepseek_v4, name)(*args, **kwargs)
    with torch_device_fn.device(tensor.device):
        return getattr(deepseek_v4, name)(*args, **kwargs)


def deepseek_v4_fused_q_kv_rmsnorm_flaggems(*args, **kwargs):
    return _reference_kwargs("deepseek_v4_fused_q_kv_rmsnorm_torch", *args, **kwargs)


def deepseek_v4_fused_q_kv_rmsnorm_int8_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_fused_q_kv_rmsnorm_int8_torch", *args, **kwargs
    )


def deepseek_v4_qnorm_rope_kv_quant_insert_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_qnorm_rope_kv_quant_insert_torch", *args, **kwargs
    )


def deepseek_v4_qnorm_rope_kv_bf16_insert_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_qnorm_rope_kv_bf16_insert_torch", *args, **kwargs
    )


def deepseek_v4_qnorm_rope_kv_fp8_insert_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_qnorm_rope_kv_fp8_insert_torch", *args, **kwargs
    )


def deepseek_v4_compute_global_topk_indices_and_lens_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_compute_global_topk_indices_and_lens_torch", *args, **kwargs
    )


def deepseek_v4_flash_mla_with_kvcache_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_flash_mla_with_kvcache_torch", *args, **kwargs
    )


def deepseek_v4_dequantize_and_gather_k_cache_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_dequantize_and_gather_k_cache_torch", *args, **kwargs
    )


def deepseek_v4_combine_topk_swa_indices_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_combine_topk_swa_indices_torch", *args, **kwargs
    )


def deepseek_v4_flash_mla_sparse_fwd_flaggems(*args, **kwargs):
    return _reference_kwargs("deepseek_v4_flash_mla_sparse_fwd_torch", *args, **kwargs)


def deepseek_v4_fused_indexer_q_rope_quant_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_fused_indexer_q_rope_quant_torch", *args, **kwargs
    )


def deepseek_v4_fused_indexer_q_rope_quant_int8_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_fused_indexer_q_rope_quant_int8_torch", *args, **kwargs
    )


def deepseek_v4_compress_int8_indexer_k_cache_flaggems(*args, **kwargs):
    return _reference_kwargs(
        "deepseek_v4_compress_int8_indexer_k_cache_torch", *args, **kwargs
    )


def deepseek_v4_int8_mqa_logits_flaggems(*args, **kwargs):
    return _reference_kwargs("deepseek_v4_int8_mqa_logits_torch", *args, **kwargs)


def deepseek_v4_int8_paged_mqa_logits_flaggems(*args, **kwargs):
    return _reference_kwargs("deepseek_v4_int8_paged_mqa_logits_torch", *args, **kwargs)

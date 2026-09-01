# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems implementation of DeepSeek-V4-specific operators."""

from __future__ import annotations

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


def deepseek_v4_mhc_pre_flaggems(*args):
    return _reference("deepseek_v4_mhc_pre_torch", *args)


def deepseek_v4_mhc_fused_post_pre_flaggems(*args):
    return _reference("deepseek_v4_mhc_fused_post_pre_torch", *args)


def deepseek_v4_mhc_post_flaggems(*args):
    return _reference("deepseek_v4_mhc_post_torch", *args)


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
    return _reference_kwargs(
        "deepseek_v4_int8_paged_mqa_logits_torch", *args, **kwargs
    )

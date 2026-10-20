# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems sparse MLA entry points for DeepSeek-V4.1 on Hopper.

These functions implement the selected EMPTY backend's contracts. They do
not claim that vLLM's native FlashMLA extension or DeepGEMM is installed.
"""

import torch


def get_mla_metadata():
    from flag_gems.fused.flash_mla_with_kvcache import get_mla_metadata as gems_metadata

    return gems_metadata()


def build_tile_scheduler(builder, num_decode_tokens):
    result = dict.fromkeys(("swaonly", "c4a", "c128a", "c1a", "c2a"))
    if num_decode_tokens:
        for layer_type in builder._layer_types:
            result[layer_type] = get_mla_metadata()[0]
    return result


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None
):
    from flag_gems.fused.flashmla_sparse import flash_mla_sparse_fwd as gems_prefill

    if q.shape[-1] != 512 or d_v != 512:
        raise NotImplementedError("Hopper DSV4.1 MLA supports head dimension 512")
    result, max_logits, lse = gems_prefill(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
    )
    if out is not None:
        if (
            out.shape != result.shape
            or out.dtype != result.dtype
            or out.device != result.device
        ):
            raise ValueError("Sparse MLA output buffer must match the result")
        out.copy_(result)
        result = out
    return result, max_logits, lse


def flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata,
    num_splits=None,
    softmax_scale=None,
    causal=False,
    is_fp8_kvcache=False,
    indices=None,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    topk_length=None,
    extra_topk_length=None,
    out=None,
):
    from flag_gems.fused.flash_mla_with_kvcache import (
        flash_mla_with_kvcache as gems_decode,
    )

    if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != 512 or head_dim_v != 512:
        raise NotImplementedError(
            "Hopper DSV4.1 MLA requires one query with head dimension 512"
        )
    if (
        not is_fp8_kvcache
        or indices is None
        or causal
        or block_table is not None
        or cache_seqlens is not None
    ):
        raise NotImplementedError(
            "Hopper DSV4.1 MLA supports sparse indexed FP8 cache decode"
        )
    for cache in (k_cache, extra_k_cache):
        if cache is not None and (
            cache.dtype != torch.uint8 or cache.ndim != 4 or cache.shape[2:] != (1, 584)
        ):
            raise NotImplementedError(
                "Hopper DSV4.1 MLA requires 584-byte packed KV records"
            )
    return gems_decode(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        causal,
        is_fp8_kvcache,
        indices,
        attn_sink,
        extra_k_cache,
        extra_indices_in_kvcache,
        topk_length,
        extra_topk_length,
        out,
    )

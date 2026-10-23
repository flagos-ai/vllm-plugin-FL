# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""DS vLLM attention ABI: scheduler metadata, TP layout and caller output buffers.

Registered by the MetaX provider; no kernel arithmetic is implemented here.
The local device kernels and launch policy live in vendor/metax/impl/dsv4.
"""
from vllm_fl import dsv4_config as _dsv4
from vllm_fl.dispatch.backends.vendor.metax.impl.dsv4.attention.decode import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
)
from .headpack import sparse_prefill as _framework_sparse_prefill


def decode_attention(
    q,
    k_cache,
    block_table,
    head_dim_v,
    tile_scheduler_metadata,
    cache_seqlens,
    is_fp8_kvcache,
    indices,
    topk_length,
    softmax_scale,
    attn_sink,
    extra_k_cache,
    extra_indices_in_kvcache,
    extra_topk_length,
    out,
):
    original_meta = tile_scheduler_metadata
    if isinstance(original_meta, FlashMLASchedMeta):
        flaggems_meta = original_meta
    else:
        flaggems_meta = FlashMLASchedMeta(
            have_initialized=original_meta.have_initialized,
            config=original_meta.config,
            tile_scheduler_metadata=original_meta.tile_scheduler_metadata,
            num_splits=original_meta.num_splits,
        )
    result = flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=flaggems_meta,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length,
        out=out,
        real_num_heads=_dsv4.current.real_query_heads,
    )
    if flaggems_meta is not original_meta:
        original_meta.have_initialized = flaggems_meta.have_initialized
        original_meta.config = flaggems_meta.config
        original_meta.tile_scheduler_metadata = flaggems_meta.tile_scheduler_metadata
        original_meta.num_splits = flaggems_meta.num_splits
    return result


def prefill_attention(q, kv, indices, sm_scale, attn_sink, topk_length, out):
    (out_tmp, max_logits, lse) = _framework_sparse_prefill(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    out.copy_(out_tmp)
    return (out, max_logits, lse)

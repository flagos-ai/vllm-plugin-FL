# Copyright (c) 2026 BAAI. All rights reserved.

"""HY V4's 576-wide, sink-aware sparse MLA on real FlagGems kernels."""

import torch

from vllm.models.hy_v4.nvidia.flashmla_sparse import (
    HYV4FlashMLASparseBackend,
    HYV4FlashMLASparseImpl,
)


class NvidiaHYV4FlashMLASparseImpl(HYV4FlashMLASparseImpl):
    # HY's sink applies to every token, so upstream already forces sparse MQA.
    # Do not instantiate an unused dense FlashAttention backend in EMPTY.
    supports_dense_mha_prefill = False


class NvidiaHYV4FlashMLASparseBackend(HYV4FlashMLASparseBackend):
    @staticmethod
    def get_impl_cls():
        return NvidiaHYV4FlashMLASparseImpl


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None
):
    from flag_gems.fused.flashmla_sparse import flash_mla_sparse_fwd as gems_prefill

    if q.shape[-1] != 576 or kv.shape[-1] != 576 or d_v != 512:
        raise NotImplementedError(
            "HY V4 sparse MLA requires Q/K dimension 576 and V 512"
        )
    result, max_logits, lse = gems_prefill(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )
    if out is not None:
        if (
            out.shape != result.shape
            or out.dtype != result.dtype
            or out.device != result.device
        ):
            raise ValueError("HY V4 MLA output buffer must match the result")
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

    if (
        q.ndim != 4
        or q.shape[1] != 1
        or q.shape[-1] != 576
        or head_dim_v != 512
        or k_cache.ndim != 4
        or k_cache.shape[2:] != (1, 656)
        or k_cache.dtype != torch.uint8
    ):
        raise NotImplementedError(
            "HY V4 sparse FP8 decode requires Q 576 and 656-byte KV records"
        )
    if not is_fp8_kvcache or indices is None or causal:
        raise NotImplementedError("HY V4 MLA requires sparse, indexed FP8 decode")
    if any(
        x is not None
        for x in (
            extra_k_cache,
            extra_indices_in_kvcache,
            topk_length,
            extra_topk_length,
        )
    ):
        raise NotImplementedError(
            "HY V4 656-byte decode does not use extra caches or dynamic top-k"
        )
    # Upstream supplies dummy paging arguments for sparse decode. The actual
    # physical cache slots are in indices; Gems, like FlashMLA, ignores these.
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
        out=out,
    )

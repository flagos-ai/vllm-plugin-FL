"""Stride-safe MLA fallback; H12 TP8 dispatch stays in the attention backend."""

from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.mla_dense_fallback import dense_decode_fallback
from typing import Optional, Tuple
import torch
from flaggems_vllm.ops.flash_mla_with_kvcache import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache as flagos_decode,
)


def try_dense_decode(
    q, kv_cache, block_table, cache_seqlens, out, lse, *, softmax_scale, causal
):
    """Preserve the stride-safe dense fallback for unsupported TP8 layouts."""
    if kv_cache.stride(-1) != 1 or block_table.stride(-1) != 1:
        raise NotImplementedError("Dense MLA requires contiguous entry/table columns")
    b, sq, heads, width = q.shape
    dense_decode_fallback(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        out,
        lse,
        b,
        sq,
        heads,
        width,
        out.shape[-1],
        kv_cache.shape[1],
        softmax_scale,
        causal,
    )
    return True


def decode_with_backend(*args, **kwargs):
    return flagos_decode(*args, dense_decode_backend=try_dense_decode, **kwargs)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # The TP8 fast path is selected by K3MLAImpl before this fallback entry.
    return decode_with_backend(
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

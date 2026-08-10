# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems implementations for DeepseekV4 attention operators.
"""
from typing import Optional, Tuple

import torch


def deepseek_v4_fp8_einsum_flaggems(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    """
    FlagGems native implementation of deepseek_v4_fp8_einsum.

    Performs FP8 einsum: out = einsum(equation, (a, a_scale), (b, b_scale))
    Mutates `out` in-place.
    """
    import flag_gems

    if len(recipe) < 2:
        raise ValueError(f"Invalid FP8 einsum recipe: {recipe}")
    result = flag_gems.fp8_einsum(
        equation,
        a,
        a_scale,
        b,
        b_scale,
        block_size=list(recipe[-2:]),
        output_dtype=out.dtype,
    )
    out.copy_(result)


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_flaggems(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    """
    FlagGems native implementation of fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.

    Horizontally fused:
      Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
      KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert

    Mutates q, swa_kv_cache_2d in-place.
    """
    import flag_gems

    flag_gems.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, swa_kv_cache_2d, slot_mapping, positions, cos_sin_cache,
        eps, block_size,
    )


# ==================== Sparse Attention Indexer Ops ====================


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
    import flag_gems

    return flag_gems.combine_topk_swa_indices(
        topk_indices, query_start_loc, seq_lens,
        gather_lens, window_size, compress_ratio,
        topk, M, N
    )


def compute_global_topk_indices_and_lens_flaggems(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlagGems implementation of compute_global_topk_indices_and_lens."""
    import flag_gems

    return flag_gems.compute_global_topk_indices_and_lens(
        topk_indices, token_to_req_indices, block_table, block_size, is_valid_token
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
    import flag_gems

    flag_gems.dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )


def fused_indexer_q_rope_quant_flaggems(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
):
    """FlagGems implementation of fused_indexer_q_rope_quant."""
    import flag_gems

    return flag_gems.fused_indexer_q_rope_quant(
        positions, index_q, index_q_cos_sin_cache,
        index_weights, index_weights_softmax_scale,
        index_weights_head_scale, use_fp4,
    )


def fused_inv_rope_fp8_quant_flaggems(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False
):
    """FlagGems implementation of fused_inv_rope_fp8_quant."""
    import flag_gems

    return flag_gems.fused_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, nope_dim, rope_dim,
        quant_group_size, tma_aligned_scales,
    )


def fused_q_kv_rmsnorm_flaggems(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlagGems implementation of fused_q_kv_rmsnorm."""
    import flag_gems

    return flag_gems.fused_q_kv_rmsnorm(
        qr, kv, q_weight, kv_weight, eps,
    )


def indexer_k_quant_and_cache_flaggems(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str,
) -> None:
    """FlagGems implementation of indexer_k_quant_and_cache."""
    import flag_gems

    flag_gems.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt,
    )


def cp_gather_indexer_k_quant_cache_flaggems(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """FlagGems implementation of cp_gather_indexer_k_quant_cache."""
    import flag_gems

    flag_gems.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens,
    )


def top_k_per_row_prefill_flaggems(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """FlagGems implementation of top_k_per_row_prefill."""
    import flag_gems

    flag_gems.top_k_per_row_prefill(
        logits, cu_seqlen_ks, cu_seqlen_ke, raw_topk_indices,
        num_rows, stride0, stride1, topk_tokens,
    )


def pack_seq_triton_flaggems(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
) -> torch.Tensor:
    """FlagGems implementation of pack_seq_triton."""
    import flag_gems

    return flag_gems.pack_seq_triton(x, lengths, pad_value)


def top_k_per_row_decode_flaggems(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """FlagGems implementation of top_k_per_row_decode."""
    import flag_gems

    flag_gems.top_k_per_row_decode(
        logits, next_n, seq_lens, raw_topk_indices,
        num_rows, stride0, stride1, topk_tokens,
    )


def unpack_seq_triton_flaggems(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """FlagGems implementation of unpack_seq_triton."""
    import flag_gems

    return flag_gems.unpack_seq_triton(packed_tensor, lengths)

def flash_mla_with_kvcache_flaggems(
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
        from flag_gems.fused.flash_mla_with_kvcache import (
            FlashMLASchedMeta,
            flash_mla_with_kvcache,
        )

        # vLLM and FlagGems expose the same scheduler fields through separate
        # dataclass types. Adapt at the backend boundary and synchronize the
        # graph-capture buffers back into the vLLM-owned metadata object.
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
        )
        if flaggems_meta is not original_meta:
            original_meta.have_initialized = flaggems_meta.have_initialized
            original_meta.config = flaggems_meta.config
            original_meta.tile_scheduler_metadata = (
                flaggems_meta.tile_scheduler_metadata
            )
            original_meta.num_splits = flaggems_meta.num_splits
        return result

def flash_mla_sparse_fwd_flaggems(
        q,
        kv,
        indices,
        sm_scale,
        attn_sink,
        topk_length,
        out,
):
    import flag_gems
    ### NOTE(lms): check output tensor
    out_tmp, max_logits, lse = flag_gems.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        topk_length=topk_length)
    out.copy_(out_tmp)
    
    return out, max_logits, lse

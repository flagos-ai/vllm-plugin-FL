# Copyright (c) 2026 BAAI. All rights reserved.

"""BF16 Lightning Indexer kernels for Hygon gfx936.

The implementations in this module are adapted from the validated
``vllm_hcu`` DeepSeek-V4 path backed up with this project.  In particular,
``bf16_paged_mqa_logits`` is the Triton kernel that the Hygon implementation
documents as being adapted from FlagGems.  Keeping the implementation in the
FL backend avoids a runtime dependency on the separate ``vllm_hcu`` plugin.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.triton_utils import tl, triton


_FALSE_VALUES = {"0", "false", "no", "off"}
_LOGITS_BUFFER_ELEMENTS = 16384 * 16384
_logits_buffers: dict[torch.device, torch.Tensor] = {}


def use_bf16_indexer_cache() -> bool:
    """Return whether the Hygon Lightning Indexer should use BF16 cache.

    The FL-specific name is preferred.  The legacy HCU variable is accepted
    so an existing Hygon deployment keeps the same startup configuration.
    BF16 defaults to enabled on gfx936, matching the validated Hygon setup.
    """
    raw = os.getenv("VLLM_FL_HYGON_USE_BF16_INDEXER_CACHE")
    if raw is None:
        raw = os.getenv("VLLM_HCU_USE_BF16_INDEXER_CACHE", "1")
    return raw.strip().lower() not in _FALSE_VALUES


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    half_rot_dim: tl.constexpr,
):
    offsets = tl.arange(0, half_rot_dim)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + offsets)
    sin = tl.load(
        cos_sin_cache_ptr + pos * cos_sin_cache_stride + offsets + half_rot_dim
    )
    return cos.to(tl.float32), sin.to(tl.float32)


@triton.jit
def _fused_indexer_q_rope_bf16_kernel(
    positions_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    half_rot_dim: tl.constexpr,
    q_bf16_ptr,
    q_bf16_stride0,
    q_bf16_stride1,
    head_dim: tl.constexpr,
    weights_ptr,
    weights_stride,
    weights_softmax_scale,
    weights_head_scale,
    weights_out_ptr,
    weights_out_stride,
):
    """Apply interleaved GPT-J RoPE and retain Q in BF16."""
    rot_dim: tl.constexpr = 2 * half_rot_dim
    nope_dim: tl.constexpr = head_dim - rot_dim
    tl.static_assert(nope_dim >= 0)

    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    position = tl.load(positions_ptr + token_idx)
    cos, sin = _get_cos_sin(
        cos_sin_cache_ptr,
        cos_sin_cache_stride,
        position,
        half_rot_dim,
    )

    half_offsets = tl.arange(0, half_rot_dim)
    q_base = index_q_ptr + token_idx * index_q_stride0 + head_idx * index_q_stride1
    rot_base = q_base + nope_dim
    even = tl.load(rot_base + half_offsets * 2).to(tl.float32)
    odd = tl.load(rot_base + half_offsets * 2 + 1).to(tl.float32)
    rotated_even = even * cos - odd * sin
    rotated_odd = odd * cos + even * sin

    out_base = q_bf16_ptr + token_idx * q_bf16_stride0 + head_idx * q_bf16_stride1
    if nope_dim > 0:
        nope_offsets = tl.arange(0, nope_dim)
        nope = tl.load(q_base + nope_offsets)
        tl.store(out_base + nope_offsets, nope.to(tl.bfloat16))
    out_rot_base = out_base + nope_dim
    tl.store(out_rot_base + half_offsets * 2, rotated_even.to(tl.bfloat16))
    tl.store(out_rot_base + half_offsets * 2 + 1, rotated_odd.to(tl.bfloat16))

    weight = tl.load(weights_ptr + token_idx * weights_stride + head_idx)
    weight = weight.to(tl.float32)
    weight *= weights_softmax_scale
    weight *= weights_head_scale
    tl.store(
        weights_out_ptr + token_idx * weights_out_stride + head_idx,
        weight,
    )


def fused_indexer_q_rope_bf16(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    weights_softmax_scale: float,
    weights_head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hygon reference ABI for BF16 Indexer Q and folded weights."""
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert cos_sin_cache.ndim == 2

    num_tokens, num_heads, head_dim = index_q.shape
    q_bf16 = torch.empty_like(index_q, dtype=torch.bfloat16)
    weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    _fused_indexer_q_rope_bf16_kernel[(num_tokens, num_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        cos_sin_cache.shape[-1] // 2,
        q_bf16,
        q_bf16.stride(0),
        q_bf16.stride(1),
        head_dim,
        index_weights,
        index_weights.stride(0),
        weights_softmax_scale,
        weights_head_scale,
        weights_out,
        weights_out.stride(0),
        num_warps=1,
    )
    return q_bf16, weights_out


@triton.jit
def _compress_norm_rope_store_bf16_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    head_dim: tl.constexpr,
    triton_block_size: tl.constexpr,
    state_width: tl.constexpr,
    compress_ratio: tl.constexpr,
    overlap: tl.constexpr,
    rope_head_dim: tl.constexpr,
    token_stride: tl.constexpr,
    kv_block_stride: tl.constexpr,
):
    """compress -> RMSNorm -> RoPE -> direct BF16 Indexer cache store."""
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % compress_ratio != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    start = position - (1 + overlap) * compress_ratio + 1
    tokens = tl.arange(0, (1 + overlap) * compress_ratio)
    state_positions = start + tokens
    valid_positions = state_positions >= 0
    block_indices = state_positions // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid_positions,
        other=0,
    )
    block_offsets = state_positions % block_size
    head_offset = (tokens >= compress_ratio).to(tl.int32) * head_dim

    offsets = tl.arange(0, triton_block_size)
    valid_head = offsets < head_dim
    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )
    mask = valid_positions[:, None] & valid_head[None, :]
    score = tl.load(
        row_base[:, None] + state_width + offsets[None, :],
        mask=mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(
        row_base[:, None] + offsets[None, :],
        mask=mask,
        other=0.0,
    )
    compressed_kv = tl.sum(kv * score, axis=0)

    rms_weight = tl.load(
        rms_norm_weight_ptr + offsets,
        mask=valid_head,
        other=0.0,
    )
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / head_dim
    normed = compressed_kv * tl.rsqrt(variance + rms_norm_eps) * rms_weight

    kv_slot = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot < 0:
        return
    kv_block = kv_slot // kv_cache_block_size
    kv_offset = kv_slot % kv_cache_block_size
    out_ptr = (
        k_cache_ptr + kv_block.to(tl.int64) * kv_block_stride + kv_offset * token_stride
    )

    nope_head_dim: tl.constexpr = head_dim - rope_head_dim
    half_rope: tl.constexpr = rope_head_dim // 2
    num_pairs: tl.constexpr = triton_block_size // 2
    nope_pairs: tl.constexpr = nope_head_dim // 2
    normed_pairs = tl.reshape(normed, (num_pairs, 2))
    even, odd = tl.split(normed_pairs)
    pair_idx = tl.arange(0, num_pairs)
    rope_pair = pair_idx - nope_pairs
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    compressed_position = (position // compress_ratio) * compress_ratio
    cs_base = cos_sin_cache_ptr + compressed_position * cos_sin_stride
    cos = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0)
    sin = tl.load(cs_base + half_rope + cs_idx, mask=is_rope, other=0.0)
    result = tl.interleave(even * cos - odd * sin, odd * cos + even * sin)
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=valid_head)


def compress_norm_rope_store_bf16(
    *,
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Launch the validated Hygon BF16 Indexer cache writer."""
    del quant_block, scale_dim
    if use_fp4_cache:
        raise AssertionError("BF16 and MXFP4 Indexer caches are mutually exclusive")
    if head_dim != 128 or kv_cache.dtype != torch.bfloat16:
        raise AssertionError(
            "Hygon BF16 Indexer writer requires head_dim=128 and BF16 cache"
        )
    _compress_norm_rope_store_bf16_kernel[(num_actual,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],
        head_dim=head_dim,
        triton_block_size=triton.next_power_of_2(head_dim),
        state_width=state_width,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rope_head_dim=rope_head_dim,
        token_stride=token_stride,
        kv_block_stride=kv_cache.stride(0),
        num_warps=1,
        **pdl_kwargs,
    )


@triton.jit
def _gather_bf16_indexer_cache_kernel(
    kv_cache_ptr,
    output_ptr,
    block_table_ptr,
    cu_seq_lens_ptr,
    block_size: tl.constexpr,
    batch_size: tl.constexpr,
    blocks_per_seq: tl.constexpr,
    cache_stride: tl.constexpr,
    head_dim: tl.constexpr,
    num_tokens: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return
    head_offsets = tl.arange(0, head_dim)
    batch_idx = tl.full((), -1, dtype=tl.int32)
    for batch in tl.static_range(batch_size):
        seq_start = tl.load(cu_seq_lens_ptr + batch)
        seq_end = tl.load(cu_seq_lens_ptr + batch + 1)
        in_batch = (token_idx >= seq_start) & (token_idx < seq_end)
        batch_idx = tl.where(in_batch, batch, batch_idx)
    if batch_idx < 0:
        return

    seq_start = tl.load(cu_seq_lens_ptr + batch_idx)
    seq_offset = token_idx - seq_start
    table_offset = seq_offset // block_size
    if table_offset >= blocks_per_seq:
        return
    block_id = tl.load(block_table_ptr + batch_idx * blocks_per_seq + table_offset)
    block_offset = seq_offset % block_size
    src = (
        kv_cache_ptr
        + block_id.to(tl.int64) * cache_stride
        + block_offset * head_dim
        + head_offsets
    )
    dst = output_ptr + token_idx * head_dim + head_offsets
    tl.store(dst, tl.load(src))


def gather_bf16_indexer_cache(
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """Gather paged BF16 Indexer keys for prefill MQA."""
    if kv_cache.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise AssertionError("BF16 Indexer gather requires BF16 input/output")
    num_tokens, head_dim = output.shape
    cache_2d = kv_cache.view(kv_cache.shape[0], -1)
    _gather_bf16_indexer_cache_kernel[(num_tokens,)](
        cache_2d,
        output,
        block_table,
        cu_seq_lens,
        kv_cache.shape[1],
        block_table.shape[0],
        block_table.shape[1],
        cache_2d.stride(0),
        head_dim,
        num_tokens,
    )


@triton.jit
def _bf16_paged_mqa_logits_kernel(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    next_n,
    max_context_len,
    block_table_stride,
):
    row = tl.program_id(0)
    logical_block = tl.program_id(1)
    context_len = tl.load(context_lens_ptr + row)
    kv_position = logical_block * 64
    if kv_position >= context_len:
        return

    heads = tl.arange(0, 64)
    dims = tl.arange(0, 128)
    positions = tl.arange(0, 64)
    q_offsets = row * (64 * 128) + heads[:, None] * 128 + dims[None, :]
    q = tl.load(q_ptr + q_offsets, eviction_policy="evict_last")
    weights = tl.load(
        weights_ptr + row * 64 + heads,
        eviction_policy="evict_last",
    )
    batch_idx = row // next_n
    physical_block = tl.load(
        block_table_ptr + batch_idx * block_table_stride + logical_block
    )
    k_offsets = physical_block * (64 * 128) + positions[:, None] * 128 + dims[None, :]
    k = tl.load(kv_cache_ptr + k_offsets, eviction_policy="evict_first")
    scores = tl.maximum(tl.dot(k, tl.trans(q)), 0.0)
    values = tl.sum(scores * weights[None, :], axis=1)
    output_offsets = row * max_context_len + kv_position + positions
    mask = positions < (context_len - kv_position)
    tl.store(output_offsets + logits_ptr, values, mask=mask)


def bf16_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
) -> torch.Tensor:
    """BF16 decode MQA logits, adapted from the Hygon/FlagGems kernel."""
    batch_size, next_n, heads, dims = q.shape
    if heads != 64 or dims != 128:
        raise AssertionError("BF16 Indexer MQA requires H=64 and D=128")
    if (
        kv_cache.ndim != 3
        or kv_cache.dtype != torch.bfloat16
        or kv_cache.shape[1:] != (64, 128)
    ):
        raise AssertionError(
            "Hygon/FlagGems BF16 paged MQA requires cache pages shaped "
            "[num_blocks, 64, 128]"
        )
    total_tokens = batch_size * next_n
    logits = torch.empty(
        (total_tokens, max_context_len),
        dtype=torch.float32,
        device=q.device,
    )
    if total_tokens == 0 or max_context_len == 0:
        return logits
    _bf16_paged_mqa_logits_kernel[(total_tokens, (max_context_len + 63) // 64)](
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        logits,
        next_n,
        max_context_len,
        block_table.stride(0),
        num_warps=8,
        num_stages=1,
    )
    return logits


def _get_logits_buffer(device: torch.device) -> torch.Tensor:
    buffer = _logits_buffers.get(device)
    if buffer is None or buffer.numel() < _LOGITS_BUFFER_ELEMENTS:
        buffer = torch.empty(
            _LOGITS_BUFFER_ELEMENTS,
            dtype=torch.float32,
            device=device,
        )
        _logits_buffers[device] = buffer
    return buffer


def _prefill_bf16_mqa_chunked(
    chunk: Any,
    q_bf16: torch.Tensor,
    k_bf16: torch.Tensor,
    weights: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Hygon LightOp BF16 prefill MQA with bounded logits workspace."""
    from .sparse_attn_indexer import (
        _get_lightop_attention,
        top_k_per_row_prefill_hygon_out,
    )

    attention = _get_lightop_attention()
    if attention is None:
        raise RuntimeError("Hygon BF16 Indexer prefill requires LightOp attention")

    q_all = q_bf16[chunk.token_start : chunk.token_end]
    weights_all = weights[chunk.token_start : chunk.token_end]
    num_queries = q_all.shape[0]
    num_keys = k_bf16.shape[0]
    aligned_keys = ((num_keys + 127) // 128) * 128
    logits_buffer = _get_logits_buffer(q_bf16.device)
    max_queries = max(1, logits_buffer.numel() // max(1, aligned_keys))
    max_queries = max(1, (max_queries // 128) * 128)

    for query_start in range(0, num_queries, max_queries):
        query_end = min(query_start + max_queries, num_queries)
        query_count = query_end - query_start
        aligned_queries = ((query_count + 127) // 128) * 128
        logits_storage = logits_buffer[: aligned_queries * aligned_keys].view(
            aligned_queries, aligned_keys
        )
        q_slice = q_all[query_start:query_end]
        weights_slice = weights_all[query_start:query_end].to(torch.float32)
        starts = chunk.cu_seqlen_ks[query_start:query_end]
        ends = chunk.cu_seqlen_ke[query_start:query_end]
        attention.mqa_logits(
            q_slice,
            k_bf16,
            weights_slice,
            starts,
            ends,
            None,
            True,
            logits_storage,
        )
        logits = logits_storage[:query_count, :num_keys]
        output = topk_indices_buffer[
            chunk.token_start + query_start : chunk.token_start + query_end,
            :topk_tokens,
        ]
        top_k_per_row_prefill_hygon_out(
            logits,
            starts,
            ends,
            output,
            query_count,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )


def sparse_attn_indexer_bf16_hygon(
    hidden_states: torch.Tensor,
    k_cache_prefix: Any,
    kv_cache: torch.Tensor,
    q_bf16: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    """Execute the Hygon BF16 Lightning Indexer prefill/decode flow."""
    from vllm.forward_context import get_forward_context
    from vllm.utils.torch_utils import _resolve_layer_name
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
    from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

    del total_seq_lens
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    layer_name = _resolve_layer_name(k_cache_prefix)
    metadata = attn_metadata[layer_name]
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        raise AssertionError("Unexpected DeepSeek-V4 Indexer metadata type")
    if kv_cache.dtype != torch.bfloat16 or q_bf16.dtype != torch.bfloat16:
        raise AssertionError("BF16 Indexer path requires BF16 Q and K cache")

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if metadata.num_prefills > 0:
        prefill = metadata.prefill
        assert prefill is not None
        for chunk in prefill.chunks:
            gathered_k = torch.empty(
                (chunk.total_seq_lens, head_dim),
                dtype=torch.bfloat16,
                device=q_bf16.device,
            )
            gather_bf16_indexer_cache(
                kv_cache,
                gathered_k,
                chunk.block_table,
                chunk.cu_seq_lens,
            )
            _prefill_bf16_mqa_chunked(
                chunk,
                q_bf16,
                gathered_k,
                weights,
                topk_indices_buffer,
                topk_tokens,
            )

    if metadata.num_decodes > 0:
        from .sparse_attn_indexer import top_k_per_row_decode_hygon_out

        decode = metadata.decode
        assert decode is not None
        num_decode_tokens = metadata.num_decode_tokens
        decode_lens = decode.decode_lens
        if decode.requires_padding:
            padded_q = pack_seq_triton(q_bf16[:num_decode_tokens], decode_lens)
        else:
            padded_q = q_bf16[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_bf16.shape[1:]
            )
        batch_size, next_n = padded_q.shape[:2]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode.seq_lens[:batch_size]
        logits = bf16_paged_mqa_logits(
            padded_q,
            kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            decode.block_table,
            max_model_len,
        )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        top_k_per_row_decode_hygon_out(
            logits,
            next_n,
            seq_lens,
            topk_indices,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
        if decode.requires_padding:
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_tokens),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


__all__ = [
    "compress_norm_rope_store_bf16",
    "fused_indexer_q_rope_bf16",
    "sparse_attn_indexer_bf16_hygon",
    "use_bf16_indexer_cache",
]

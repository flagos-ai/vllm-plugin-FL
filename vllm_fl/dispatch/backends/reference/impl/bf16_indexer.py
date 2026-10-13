# SPDX-License-Identifier: Apache-2.0
"""Portable eager reference operators for BF16 indexer attention."""

from __future__ import annotations

import torch


def bf16_indexer_cache_write_torch(
    keys: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write valid token keys into a flattened paged cache.

    This reference path intentionally favors clarity and portability.  Vendor
    backends should provide a graph-safe implementation when graph replay is a
    platform requirement.
    """
    if keys.ndim != 2 or cache.ndim != 3 or slot_mapping.ndim != 1:
        raise ValueError(
            "BF16 indexer cache write expects keys [T,D], cache [B,S,D], "
            "and slot_mapping [T]"
        )
    if keys.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("BF16 indexer cache write requires BF16 keys and cache")
    if keys.shape[1] != cache.shape[2]:
        raise ValueError("BF16 indexer key/cache head dimensions must match")
    num_tokens = slot_mapping.shape[0]
    if keys.shape[0] < num_tokens:
        raise ValueError("slot_mapping cannot contain more tokens than keys")
    valid = slot_mapping >= 0
    if not bool(valid.any()):
        return
    flat_cache = cache.reshape(-1, cache.shape[-1])
    flat_cache.index_copy_(
        0,
        slot_mapping[valid].to(torch.long),
        keys[:num_tokens][valid].to(cache.dtype),
    )


def _flatten_decode_seq_lens(
    seq_lens: torch.Tensor, batch_size: int, next_n: int
) -> torch.Tensor:
    flat_lens = seq_lens.reshape(-1)
    if flat_lens.numel() == batch_size and next_n > 1:
        offsets = torch.arange(next_n, device=seq_lens.device, dtype=seq_lens.dtype)
        offsets = offsets - (next_n - 1)
        flat_lens = (flat_lens[:, None] + offsets[None, :]).clamp_min(0).reshape(-1)
    return flat_lens


def _bf16_paged_mqa_logits_torch(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    *,
    max_context_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Compute BF16 paged-MQA logits with ordinary PyTorch operations.

    The implementation uses host-visible sequence lengths and is therefore an
    eager correctness fallback, not a graph backend.
    """
    del schedule_metadata, clean_logits
    if q.ndim != 4:
        raise ValueError("BF16 paged-MQA logits expects q [B,N,H,D]")
    if cache.ndim == 4 and cache.shape[-2] == 1:
        cache = cache.squeeze(-2)
    if cache.ndim != 3:
        raise ValueError("BF16 paged-MQA logits expects cache [blocks,S,D]")
    if q.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("BF16 paged-MQA logits requires BF16 query and cache")
    if weights.dtype != torch.float32:
        raise ValueError("BF16 paged-MQA logits requires FP32 per-head weights")

    batch_size, next_n, num_heads, head_dim = q.shape
    if cache.shape[-1] != head_dim:
        raise ValueError("BF16 paged-MQA query/cache head dimensions must match")
    flat_q = q.reshape(-1, num_heads, head_dim)
    flat_weights = weights.reshape(-1, num_heads)
    flat_lens = _flatten_decode_seq_lens(seq_lens, batch_size, next_n)
    if flat_lens.numel() != flat_q.shape[0]:
        raise ValueError(
            "BF16 paged-MQA sequence lengths must contain one value per request or "
            "per padded query token"
        )
    output = q.new_full((flat_q.shape[0], max_context_len), float("-inf"))
    block_size = cache.shape[1]

    for token_id in range(flat_q.shape[0]):
        request_id = token_id // next_n
        seq_len = int(flat_lens[token_id].item())
        if seq_len <= 0:
            continue
        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = block_table[request_id, :num_blocks].to(torch.long)
        keys = cache.index_select(0, blocks).reshape(-1, head_dim)[:seq_len]
        scores = torch.matmul(keys, flat_q[token_id].transpose(0, 1))
        logits = (
            torch.relu(scores) * flat_weights[token_id].to(scores.dtype).unsqueeze(0)
        ).sum(dim=-1)
        output[token_id, :seq_len] = logits
    return output


def bf16_indexer_decode_torch(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    indices: torch.Tensor,
    *,
    next_n: int,
    max_context_len: int,
    clean_logits: bool = False,
) -> None:
    """Portable eager fallback for the complete BF16 Indexer decode stage."""
    logits = _bf16_paged_mqa_logits_torch(
        q,
        cache,
        weights,
        seq_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_context_len,
        clean_logits=clean_logits,
    )
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("BF16 Indexer decode expects INT32 indices [rows, top_k]")
    if indices.shape[0] != logits.shape[0]:
        raise ValueError("logits and indices must have the same number of rows")

    batch_size = q.shape[0]
    flat_lens = _flatten_decode_seq_lens(seq_lens, batch_size, next_n)
    if flat_lens.numel() != logits.shape[0]:
        raise ValueError("sequence lengths must resolve to one value per logit row")
    indices.fill_(-1)
    for row in range(logits.shape[0]):
        valid_len = min(int(flat_lens[row].item()), logits.shape[1])
        count = min(indices.shape[1], valid_len)
        if count:
            indices[row, :count].copy_(
                torch.topk(logits[row, :valid_len], count, sorted=True).indices
            )


__all__ = [
    "bf16_indexer_cache_write_torch",
    "bf16_indexer_decode_torch",
]

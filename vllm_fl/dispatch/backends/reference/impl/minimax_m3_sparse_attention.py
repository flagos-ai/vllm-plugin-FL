# Copyright (c) 2026 BAAI. All rights reserved.

"""PyTorch reference for MiniMax-M3 paged, block-sparse GQA."""

from __future__ import annotations

import torch

_BLOCK_SIZE = 128
_QUERY_TILE = 16
_BLOCK_TILE = 8


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
) -> None:
    """Write causal attention over each query and KV head's selected blocks."""
    if isinstance(kv_cache, (tuple, list)) and len(kv_cache) >= 2:
        key, value = kv_cache[:2]
    elif isinstance(kv_cache, torch.Tensor) and kv_cache.ndim == 5:
        if kv_cache.shape[0] == 2:
            key, value = kv_cache.unbind(0)
        elif kv_cache.shape[1] == 2:
            key, value = kv_cache.unbind(1)
        else:
            raise ValueError("Expected a K/V axis of length two")
    else:
        raise ValueError("Expected a pair of paged caches or a stacked K/V cache")

    rows, heads, dim = q.shape
    batch = seq_lens.numel()
    if num_kv_heads <= 0 or heads % num_kv_heads:
        raise ValueError("Query heads must be divisible by the number of KV heads")
    if (
        key.ndim != 4
        or key.shape != value.shape
        or key.shape[1:] != (_BLOCK_SIZE, num_kv_heads, dim)
    ):
        raise ValueError("Expected K and V in [pages, 128, KV heads, head dim] layout")
    if (
        topk_idx.ndim != 3
        or topk_idx.shape[:2] != (num_kv_heads, rows)
        or output.shape != q.shape
        or block_table.ndim != 2
        or block_table.shape[0] != batch
        or seq_lens.shape != (batch,)
        or prefix_lens.shape != (batch,)
        or cu_seqlens_q.shape != (batch + 1,)
        or max_query_len < 0
    ):
        raise ValueError("Incompatible MiniMax-M3 sparse attention metadata")

    pages = key.shape[0]
    table_width = block_table.shape[1]
    topk = topk_idx.shape[-1]
    if not (batch and pages and table_width and topk and max_query_len):
        output.zero_()
        return

    groups = heads // num_kv_heads
    kv_heads = torch.arange(num_kv_heads, device=q.device)[None, :, None]
    offsets = torch.arange(_BLOCK_SIZE, device=q.device)
    # Index (page, KV head) together, preserving either cache's physical strides.
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    for start in range(0, rows, _QUERY_TILE):
        end = min(start + _QUERY_TILE, rows)
        row = torch.arange(start, end, device=q.device)
        request = (row[:, None] >= cu_seqlens_q[None, 1:]).sum(dim=1)
        request = request.clamp(max=batch - 1)
        local_row = row - cu_seqlens_q[request]
        valid_row = (
            (row < cu_seqlens_q[request + 1])
            & (local_row >= 0)
            & (local_row < max_query_len)
        )
        # A chunk may follow an already cached prefix and end before seq_len.
        last_key = torch.minimum(
            prefix_lens[request] + local_row, seq_lens[request] - 1
        )
        queries = q[start:end].reshape(end - start, num_kv_heads, groups, dim).float()
        total = torch.zeros_like(queries)
        denom = torch.zeros_like(queries[..., :1])
        peak = torch.full_like(denom, -torch.inf)
        for first in range(0, topk, _BLOCK_TILE):
            blocks = topk_idx[:, start:end, first : first + _BLOCK_TILE].transpose(0, 1)
            blocks = blocks.long()
            physical = block_table[
                request[:, None, None], blocks.clamp(0, table_width - 1)
            ]
            valid = (
                valid_row[:, None, None]
                & (blocks >= 0)
                & (blocks < table_width)
                & (physical >= 0)
                & (physical < pages)
            )
            visible = valid[..., None] & (
                blocks[..., None] * _BLOCK_SIZE + offsets
                <= last_key[:, None, None, None]
            )
            physical = physical.long().clamp(0, pages - 1)
            keys = key[physical, kv_heads].flatten(2, 3).to(q.dtype).float()
            values = value[physical, kv_heads].flatten(2, 3).to(q.dtype).float()
            mask = visible.flatten(2, 3)
            scores = (queries @ keys.transpose(-1, -2)) * sm_scale
            scores = scores.masked_fill(~mask[:, :, None, :], -torch.inf)
            next_peak = torch.maximum(peak, scores.amax(dim=-1, keepdim=True))
            # Use a finite shift for an empty row, so every weight is exactly zero.
            shift = torch.where(next_peak == -torch.inf, 0.0, next_peak)
            rescale = (peak - shift).exp()
            weights = (scores - shift).exp()
            # Invalid entries can address uninitialized cache data, including NaNs.
            values = values.masked_fill(~mask[..., None], 0.0)
            total = total * rescale + weights @ values
            denom = denom * rescale + weights.sum(dim=-1, keepdim=True)
            peak = next_peak
        output[start:end].copy_(
            (total / denom.clamp_min(1)).reshape(end - start, heads, dim)
        )

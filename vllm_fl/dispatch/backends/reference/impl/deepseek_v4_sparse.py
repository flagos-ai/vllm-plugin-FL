# SPDX-License-Identifier: Apache-2.0
"""PyTorch precision references for DeepSeek V4 logical paged attention."""

import torch

_FLOAT = (torch.float16, torch.bfloat16, torch.float32)
_INTEGER = (torch.int32, torch.int64)


def _check_requests(query, requests, positions):
    if query.ndim != 3 or min(query.shape[1:]) < 1:
        raise ValueError(
            "Expected queries [tokens, heads, dim] with positive head size"
        )
    for tensor in (requests, positions):
        if tensor.shape != query.shape[:1] or tensor.dtype not in _INTEGER:
            raise ValueError("Expected one integer request ID and position per query")
        if tensor.device != query.device:
            raise ValueError("Query and metadata must share a device")


def _lengths(lengths, requests):
    if not lengths.numel():
        return torch.zeros_like(requests)
    valid = (requests >= 0) & (requests < lengths.numel())
    return torch.where(valid, lengths[requests.long().clamp(0, lengths.numel() - 1)], 0)


def _completed(positions, ratio):
    # Count completed groups without overflowing the largest INT64 position.
    return positions // ratio + (positions % ratio == ratio - 1)


def _paged_read(cache, table, requests, logical, limit):
    """Gather logical rows, zeroing invalid physical and causal addresses."""
    if not table.numel() or not cache.shape[0]:
        return (
            cache.new_zeros((*logical.shape, cache.shape[-1])),
            torch.zeros_like(logical, dtype=torch.bool),
        )
    block = logical.long() // cache.shape[1]
    pages = table[
        requests.long().clamp(0, table.shape[0] - 1)[:, None],
        block.clamp(0, table.shape[1] - 1),
    ].long()
    valid = (
        (requests[:, None] >= 0)
        & (requests[:, None] < table.shape[0])
        & (logical >= 0)
        & (logical < limit[:, None])
        & (block < table.shape[1])
        & (pages >= 0)
        & (pages < cache.shape[0])
    )
    values = cache[pages.clamp(0, cache.shape[0] - 1), logical.long() % cache.shape[1]]
    return torch.where(valid[..., None], values, 0), valid


def deepseek_v4_indexer(
    query,
    key_cache,
    weights,
    block_table,
    request_ids,
    positions,
    key_lengths,
    topk,
    compress_ratio=4,
    query_scale=None,
    key_scale_cache=None,
):
    """Rank causal C4 keys by weighted ReLU dot product, then ascending ID."""
    _check_requests(query, request_ids, positions)
    tokens, heads, dim = query.shape
    if compress_ratio != 4 or topk < 1:
        raise ValueError("The compressed indexer requires C4 and positive topk")
    if key_cache.ndim != 3 or key_cache.shape[2] != dim or not key_cache.shape[1]:
        raise ValueError("Expected paged keys [pages, block, query dim]")
    if weights.shape != (tokens, heads) or weights.dtype not in _FLOAT:
        raise ValueError("Expected floating index weights [tokens, heads]")
    if block_table.ndim != 2 or key_lengths.ndim != 1:
        raise ValueError("Expected a request block table and key lengths vector")
    if block_table.dtype not in _INTEGER or key_lengths.dtype not in _INTEGER:
        raise ValueError("Indexer metadata must be INT32 or INT64")
    if any(x.dtype not in (*_FLOAT, torch.int8) for x in (query, key_cache)):
        raise ValueError("Expected INT8 or floating point query/key tensors")
    if query.dtype == torch.int8 and query_scale is None:
        raise ValueError("INT8 queries require per-token/head scales")
    if key_cache.dtype == torch.int8 and key_scale_cache is None:
        raise ValueError("INT8 keys require per-row scales")
    if query_scale is not None and query_scale.shape not in (
        (tokens * heads,),
        (tokens, heads),
        (tokens, heads, 1),
    ):
        raise ValueError("Expected per-token/head query scales")
    if key_scale_cache is not None and (
        key_scale_cache.ndim != 3
        or key_scale_cache.shape[1:] != (key_cache.shape[1], 1)
    ):
        raise ValueError("Expected paged per-row key scales")
    scales = [x for x in (query_scale, key_scale_cache) if x is not None]
    if any(x.dtype not in _FLOAT for x in scales):
        raise ValueError("Indexer scales must be FP16, BF16 or FP32")
    if any(
        x.device != query.device
        for x in (key_cache, weights, block_table, key_lengths, *scales)
    ):
        raise ValueError("All indexer inputs must share a device")
    capacity = block_table.shape[1] * key_cache.shape[1]
    ranks = min(topk, capacity)
    if capacity > 2**31 - 1 or ranks > 512:
        raise ValueError("Expected INT32 candidate IDs and at most 512 selected keys")
    out = query.new_full((tokens, topk), -1, dtype=torch.int32)
    if not ranks:
        return out
    if query_scale is not None:
        query_scale = query_scale.reshape(tokens, heads, 1)
    # Tile both axes so long prefills never gather the full candidate cache.
    for start in range(0, tokens, 8):
        batch = slice(start, start + 8)
        req = request_ids[batch]
        pos = positions[batch].long()
        limit = torch.minimum(_completed(pos, 4), _lengths(key_lengths, req))
        best = query.new_empty((req.numel(), 0), dtype=torch.float32)
        ids = best.long()
        for offset in range(0, capacity, 1024):
            logical = torch.arange(
                offset, min(offset + 1024, capacity), device=query.device
            )
            logical = logical.expand(req.numel(), -1)
            keys, valid = _paged_read(key_cache, block_table, req, logical, limit)
            q = query[batch].float()
            k = keys.float().transpose(1, 2)
            if query.dtype == key_cache.dtype == torch.int8 and dim > 1024:
                # Each partial INT8 dot fits exactly in FP32. Combine in INT32
                # before applying the per-head and per-row floating scales.
                parts = zip(q.split(1024, dim=2), k.split(1024, dim=1))
                dots = sum(torch.bmm(qi, ki).int() for qi, ki in parts).float()
            else:
                dots = torch.bmm(q, k)
            if query_scale is not None:
                dots = dots * query_scale[batch].float()
            if key_scale_cache is not None:
                key_scales, live = _paged_read(
                    key_scale_cache, block_table, req, logical, limit
                )
                dots = dots * key_scales.transpose(1, 2).float()
                valid = valid & live
            scores = (dots.relu() * weights[batch, :, None].float()).sum(1)
            scores = scores.masked_fill(~(valid & torch.isfinite(scores)), -torch.inf)
            best = torch.cat((best, scores), dim=1)
            ids = torch.cat((ids, logical), dim=1)
            # Earlier chunks contain smaller IDs. Stable sorting preserves that
            # tie order both within a chunk and when merging previous winners.
            order = torch.argsort(best, dim=1, descending=True, stable=True)[:, :ranks]
            best = best.gather(1, order)
            ids = ids.gather(1, order)
        out[batch, :ranks] = torch.where(torch.isfinite(best), ids, -1).int()
    return out


def deepseek_v4_sparse_attention(
    query,
    swa_cache,
    compact_cache,
    swa_block_table,
    compact_block_table,
    request_ids,
    positions,
    swa_lengths,
    compact_lengths,
    compact_indices,
    window_size,
    compress_ratio=4,
    scale=None,
    attn_sink=None,
    value_dim=None,
):
    """Joint causal SWA/compact softmax, including the zero-value sink."""
    _check_requests(query, request_ids, positions)
    tokens, heads, dim = query.shape
    value_dim = dim if value_dim is None else value_dim
    if (
        not isinstance(window_size, int)
        or window_size < 1
        or compress_ratio not in (0, 4, 128)
    ):
        raise ValueError("Expected positive window and compression ratio 0, 4 or 128")
    if not 0 < value_dim <= dim <= 512:
        raise ValueError("Expected positive value width within shared KV width <=512")
    if query.dtype not in _FLOAT:
        raise TypeError("Expected BF16, FP16 or FP32 queries")
    compact_metadata = (compact_block_table, compact_lengths, compact_indices)
    sources = [(swa_cache, swa_block_table, swa_lengths, None, window_size)]
    if compact_cache is not None:
        if not compress_ratio or any(x is None for x in compact_metadata):
            raise ValueError(
                "Compact attention requires a ratio, table, lengths and indices"
            )
        if (
            compact_indices.ndim != 2
            or compact_indices.shape[0] != tokens
            or compact_indices.dtype not in _INTEGER
            or compact_indices.device != query.device
        ):
            raise ValueError(
                "Expected integer compact indices [tokens, topk] on query device"
            )
        sources.append(
            (
                compact_cache,
                compact_block_table,
                compact_lengths,
                compact_indices,
                compact_indices.shape[1],
            )
        )
    elif any(x is not None for x in compact_metadata):
        raise ValueError("Compressed metadata requires a compact cache")
    for cache, table, lengths, _, _ in sources:
        if cache.ndim != 3 or not min(cache.shape[:2]) or cache.shape[2] != dim:
            raise ValueError("Expected nonempty paged shared KV [pages, block, dim]")
        if cache.dtype != query.dtype:
            raise TypeError("Shared KV and query must have the same dtype")
        if (
            table is None
            or lengths is None
            or table.ndim != 2
            or lengths.shape != table.shape[:1]
            or table.dtype not in _INTEGER
            or lengths.dtype not in _INTEGER
        ):
            raise ValueError(
                "Expected integer block table and lengths for the same requests"
            )
        if any(x.device != query.device for x in (cache, table, lengths)):
            raise ValueError("Attention inputs must share a device")
    if attn_sink is not None and (
        attn_sink.shape != (heads,)
        or not attn_sink.is_floating_point()
        or attn_sink.device != query.device
    ):
        raise ValueError(
            "Expected one floating sink logit per query head on query device"
        )
    scale = dim**-0.5 if scale is None else scale
    out = query.new_empty((tokens, heads, value_dim))
    for start in range(0, tokens, 8):
        batch = slice(start, start + 8)
        req = request_ids[batch]
        pos = positions[batch].long()
        q = query[batch].float()
        result = q.new_zeros((*q.shape[:2], value_dim))
        maximum = q.new_full(q.shape[:2], -torch.inf)
        denominator = torch.zeros_like(maximum)
        if attn_sink is not None:
            maximum = attn_sink.float().expand_as(maximum)
            denominator = (~torch.isneginf(maximum)).float()
        for cache, table, lengths, indices, width in sources:
            limit = torch.where(pos >= 0, _lengths(lengths, req), 0)
            if indices is not None:
                limit = torch.minimum(limit, _completed(pos, compress_ratio))
            for offset in range(0, width, 1024):
                end = min(offset + 1024, width)
                if indices is None:
                    delta = torch.arange(offset, end, device=q.device) - (
                        window_size - 1
                    )
                    logical = pos.clamp_min(0)[:, None] + delta
                else:
                    logical = indices[batch, offset:end]
                kv, valid = _paged_read(cache, table, req, logical, limit)
                kv = kv.float()
                logits = torch.bmm(q, kv.transpose(1, 2)) * scale
                logits = logits.masked_fill(~valid[:, None], -torch.inf)
                new_max = torch.maximum(maximum, logits.amax(dim=-1))
                # An all-masked prefix has zero mass and a maximum of -inf.
                origin = torch.where(torch.isneginf(new_max), 0, new_max)
                carry = (maximum - origin).exp()
                weights = (logits - origin[..., None]).exp()
                denominator = denominator * carry + weights.sum(dim=-1)
                result = result * carry[..., None] + torch.bmm(
                    weights, kv[..., :value_dim]
                )
                maximum = new_max
        denominator = torch.where(denominator > 0, denominator, 1)
        out[batch] = (result / denominator[..., None]).to(query.dtype)
    return out

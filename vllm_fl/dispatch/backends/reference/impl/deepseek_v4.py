# SPDX-License-Identifier: Apache-2.0
"""Torch reference contracts for continuous paged DeepSeek V4 caches.

Projection, page allocation, query metadata and output-row scheduling belong to
the caller. These providers own numerical work and cache writes only; they do
not depend on an Ascend extension, runner or process group.
"""

import torch
import torch.nn.functional as F


def deepseek_v4_partial_rope(x, positions, cos_sin_cache, inverse=False):
    """Rotate the final head dimensions in FP32 with adjacent-pair convention."""
    width = cos_sin_cache.shape[-1]
    if x.ndim not in (2, 3) or positions.shape != x.shape[:1]:
        raise ValueError("Expected [tokens, (heads), dim] and one position per token")
    if width <= 0 or width % 2 or width > x.shape[-1]:
        raise ValueError("RoPE width must be positive, even and within the head")
    cache = cos_sin_cache[positions.long().clamp(0, cos_sin_cache.shape[0] - 1)]
    cos, sin = cache.float().chunk(2, -1)
    if x.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    if inverse:
        sin = -sin
    even, odd = x[..., -width::2].float(), x[..., -width + 1 :: 2].float()
    rotated = torch.stack([even * cos - odd * sin, odd * cos + even * sin], -1)
    result = torch.cat([x[..., :-width], rotated.flatten(-2).to(x.dtype)], -1)
    valid = (positions >= 0) & (positions < cos_sin_cache.shape[0])
    return torch.where(valid.reshape(-1, *([1] * (x.ndim - 1))), result, 0)


def deepseek_v4_hadamard(x, matrix):
    """Keep the projection and normalized scaling as separate dtype-round stages."""
    dim = x.shape[-1]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < dim:
        raise ValueError("Expected a covering square Hadamard matrix")
    padded = F.pad(x, (0, matrix.shape[0] - dim))
    return (F.linear(padded, matrix.to(x.dtype)) * dim**-0.5)[..., :dim]


def deepseek_v4_index_quant(x):
    from vllm_fl.quantization.w8a8.reference import dynamic_per_token_quant_int8

    if x.ndim < 2 or not x.is_floating_point():
        raise ValueError("Indexer quantization requires floating token/head rows")
    quantized, scale = dynamic_per_token_quant_int8(x.reshape(-1, x.shape[-1]))
    return quantized.reshape(x.shape), scale.reshape(*x.shape[:-1], 1).half()


def deepseek_v4_grouped_project(x, weight, groups):
    """Apply logical checkpoint [groups*out, in] weights, not a dense shared GEMM."""
    if x.ndim != 3 or groups < 1 or weight.ndim != 2:
        raise ValueError(
            "Grouped projection requires [tokens, heads, dim] and [out,in]"
        )
    if weight.shape[0] % groups or x.shape[1] * x.shape[2] != groups * weight.shape[1]:
        raise ValueError("Grouped output projection dimensions do not match")
    grouped = x.reshape(x.shape[0], groups, weight.shape[1])
    weights = weight.reshape(groups, weight.shape[0] // groups, weight.shape[1])
    return torch.einsum("tgi,goi->tgo", grouped, weights).flatten(1)


def deepseek_v4_project(x, wkv, wgate):
    """Retain FP32 compressor projections instead of rounding before state storage."""
    if x.ndim != 2 or wkv.ndim != 2 or wgate.shape != wkv.shape:
        raise ValueError(
            "Expected [tokens, hidden] inputs and matching projection weights"
        )
    if x.shape[-1] != wkv.shape[-1]:
        raise ValueError("Compressor projection input width mismatch")
    if any(
        t.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        for t in (x, wkv, wgate)
    ):
        raise TypeError("Compressor projections require floating, unquantized weights")
    if x.shape[0] == 0:
        shape = (0, wkv.shape[0])
        return x.new_empty(shape, dtype=torch.float32), x.new_empty(
            shape, dtype=torch.float32
        )
    return F.linear(x.float(), wkv.float()), F.linear(x.float(), wgate.float())


def deepseek_v4_hc_pre(x, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters):
    """Return native-order layer input, post coefficients and residual mix matrix."""
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch

    if x.ndim != 3 or x.dtype != torch.bfloat16:
        raise ValueError("HC pre requires BF16 [tokens, hc, hidden] residuals")
    hc, dim = x.shape[1:]
    mix = (2 + hc) * hc
    if fn.shape != (mix, hc * dim) or scale.shape != (3,) or base.shape != (mix,):
        raise ValueError("HC pre mixing parameter shapes do not match the residual")
    if any(t.dtype != torch.float32 for t in (fn, scale, base)):
        raise ValueError("HC mixing parameters must remain FP32")
    if sinkhorn_iters < 1:
        raise ValueError("HC requires at least one Sinkhorn iteration")
    if x.shape[0] == 0:
        return (
            x.new_empty((0, dim)),
            x.new_empty((0, hc), dtype=torch.float32),
            x.new_empty((0, hc, hc), dtype=torch.float32),
        )
    post, comb, layer_input = mhc_pre_torch(
        x.contiguous(), fn, scale, base, rms_eps, hc_eps, hc_eps, 2.0, sinkhorn_iters
    )
    return layer_input, post.squeeze(-1), comb


def deepseek_v4_hc_post(x, residual, post, comb):
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    if residual.ndim != 3 or x.shape != (residual.shape[0], residual.shape[2]):
        raise ValueError("HC post input and residual shapes do not match")
    if post.shape != residual.shape[:2] or comb.shape != (
        *residual.shape[:2],
        residual.shape[1],
    ):
        raise ValueError("HC post mixing shapes do not match the residual")
    if x.shape[0] == 0:
        return torch.empty_like(residual)
    return mhc_post_torch(x, residual, post.unsqueeze(-1), comb)


def deepseek_v4_hc_head(x, fn, scale, base, rms_eps, hc_eps):
    """Collapse residual streams with normalized, input-dependent head weights."""
    if x.ndim != 3:
        raise ValueError("HC head requires [tokens, hc, hidden] residuals")
    hc, dim = x.shape[1:]
    if fn.shape != (hc, hc * dim) or scale.numel() != 1 or base.shape != (hc,):
        raise ValueError("HC head parameter shapes do not match the residual")
    if any(t.dtype != torch.float32 for t in (fn, scale, base)):
        raise ValueError("HC mixing parameters must remain FP32")
    if x.shape[0] == 0:
        return x.new_empty((0, dim))
    flat = x.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes * scale + base) + hc_eps
    return (pre.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)


def _paged_read(cache, table, requests, positions):
    """Gather [rows, slots, dim], masking padding before any arithmetic."""
    if cache.ndim != 3 or table.ndim != 2:
        raise ValueError(
            "Expected [pages, block, dim] cache and [requests, blocks] table"
        )
    if positions.ndim != 2 or requests.shape != positions.shape[:1]:
        raise ValueError("Expected one request ID per positions row")
    block = cache.shape[1]
    logical = positions.div(block, rounding_mode="floor")
    valid = (
        (requests[:, None] >= 0)
        & (requests[:, None] < table.shape[0])
        & (positions >= 0)
        & (logical < table.shape[1])
    )
    pages = table[
        requests.long().clamp(0, table.shape[0] - 1)[:, None],
        logical.long().clamp(0, table.shape[1] - 1),
    ].long()
    valid = valid & (pages >= 0) & (pages < cache.shape[0])
    values = cache[
        pages.clamp(0, cache.shape[0] - 1), positions.long().clamp_min(0) % block
    ]
    return torch.where(valid[..., None], values, 0), valid


def _lengths(lengths, requests):
    return lengths[requests.long().clamp(0, lengths.shape[0] - 1)]


def deepseek_v4_cache_write(values, slots, cache):
    """Write unique valid slots; negative slots are padding, never slot zero."""
    if (
        cache.ndim != 3
        or values.ndim != 2
        or values.shape[1] != cache.shape[-1]
        or slots.shape != values.shape[:1]
    ):
        raise ValueError(
            "Expected matching [tokens, dim], [tokens], [pages, block, dim]"
        )
    if cache.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.int8):
        raise ValueError("Unsupported cache storage dtype")
    if values.shape[0] == 0:
        return
    if any(size == 0 for size in cache.shape):
        raise ValueError("Cannot write to an empty cache")
    num_slots = cache.shape[0] * cache.shape[1]
    valid = (slots >= 0) & (slots < num_slots)
    indices = slots.long().clamp(0, num_slots - 1)
    # Address the original storage so padded/noncontiguous cache views retain
    # their strides. This view allocates no cache-sized workspace.
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(cache.shape, cache.stride())
    )
    storage = cache.as_strided((span,), (1,))
    channels = torch.arange(cache.shape[-1], device=cache.device) * cache.stride(2)
    for token in range(values.shape[0]):
        slot = indices[token]
        addresses = (
            slot.div(cache.shape[1], rounding_mode="floor") * cache.stride(0)
            + (slot % cache.shape[1]) * cache.stride(1)
            + channels
        )
        previous = torch.index_select(storage, 0, addresses)
        update = torch.where(valid[token], values[token].to(cache.dtype), previous)
        # One row per call avoids duplicate-index writes when padding aliases
        # a valid slot, including slot zero. Scratch is bounded by row width.
        storage.index_put_((addresses,), update)


def deepseek_v4_save_state(
    kv, gate, ape, positions, slot_mapping, state_cache, compress_ratio
):
    """Save unique valid slots as FP32 [kv, gate + APE]; negative slots are padding."""
    if compress_ratio not in (4, 128):
        raise ValueError("Compression ratio must be 4 or 128")
    if state_cache.dtype != torch.float32 or state_cache.ndim != 3:
        raise ValueError("Compressor state must be FP32 [pages, block, packed dim]")
    if kv.ndim != 2 or gate.shape != kv.shape:
        raise ValueError(
            "KV and gate projections must have matching [tokens, width] shape"
        )
    if kv.dtype != torch.float32 or gate.dtype != torch.float32:
        raise ValueError("Compressor projections must retain FP32 accumulation outputs")
    if ape.shape != (compress_ratio, kv.shape[-1]):
        raise ValueError("APE shape must match compression ratio and projection width")
    if positions.shape != kv.shape[:1] or slot_mapping.shape != positions.shape:
        raise ValueError("Expected one position and slot per projected token")
    if state_cache.shape[-1] != 2 * kv.shape[-1]:
        raise ValueError("State width must pack both KV and gate projections")
    if kv.shape[0] == 0:
        return
    packed = torch.cat(
        [kv.float(), gate.float() + ape[positions.long() % compress_ratio].float()], -1
    )
    # The writer preserves padding without racing with a real slot-zero write,
    # including NaNs in padded projections, without data-dependent shapes.
    deepseek_v4_cache_write(packed, slot_mapping, state_cache)


def _query_tile_size(capacity, width, num_heads=0):
    # Bound the principal FP32 gather/score tensors to about 64 MiB per tile.
    elements_per_query = max(1, capacity * (width + num_heads))
    return min(32, max(1, (16 * 1024 * 1024) // elements_per_query))


def deepseek_v4_compress(
    state_cache,
    block_table,
    request_ids,
    positions,
    rms_weight,
    cos_sin_cache,
    compress_ratio=4,
    rms_eps=1e-6,
    output_dtype=torch.bfloat16,
):
    """Compress completed groups at original end positions; padded rows are zero.

    C4 mixes the previous group's first half with the current group's second
    half. C128 has no overlap. Softmax is per channel over source tokens; RMSNorm
    and adjacent-pair RoPE precede the single output cast. This is continuous
    paged state, not a cyclic state-cache implementation.
    """
    if compress_ratio not in (4, 128):
        raise ValueError("Compression ratio must be 4 or 128")
    dim = rms_weight.numel()
    coff = 2 if compress_ratio == 4 else 1
    if state_cache.dtype != torch.float32 or state_cache.shape[-1] != 2 * coff * dim:
        raise ValueError("Unexpected compressor state dtype or packed width")
    rope_dim = cos_sin_cache.shape[-1]
    if rope_dim % 2 or rope_dim > dim:
        raise ValueError("RoPE width must be even and no larger than the head")
    if positions.shape[0] == 0:
        return state_cache.new_empty((0, dim), dtype=output_dtype)
    total = coff * compress_ratio
    query_tile = _query_tile_size(total, state_cache.shape[-1])
    if positions.shape[0] > query_tile:
        return torch.cat(
            [
                deepseek_v4_compress(
                    state_cache,
                    block_table,
                    request_ids[start : start + query_tile],
                    positions[start : start + query_tile],
                    rms_weight,
                    cos_sin_cache,
                    compress_ratio,
                    rms_eps,
                    output_dtype,
                )
                for start in range(0, positions.shape[0], query_tile)
            ],
            dim=0,
        )
    source = (
        positions[:, None] - total + 1 + torch.arange(total, device=positions.device)
    )
    raw, valid = _paged_read(state_cache, block_table, request_ids, source)
    raw = raw.reshape(positions.shape[0], total, 2, coff, dim)
    part = torch.arange(total, device=positions.device) // compress_ratio
    part = part[None, :, None, None].expand(positions.shape[0], total, 1, dim)
    kv = raw[:, :, 0].gather(2, part).squeeze(2)
    scores = raw[:, :, 1].gather(2, part).squeeze(2)
    rope_position = (
        positions.div(compress_ratio, rounding_mode="floor") * compress_ratio
    )
    active = (
        (positions >= 0)
        & (request_ids >= 0)
        & ((positions + 1) % compress_ratio == 0)
        & (rope_position < cos_sin_cache.shape[0])
    )
    scores = scores.masked_fill(~valid[..., None], -torch.inf)
    has_source = valid.any(-1) & active
    scores = torch.where(has_source[:, None, None], scores, 0)
    compressed = (kv * scores.softmax(dim=1)).sum(dim=1)
    normed = compressed * torch.rsqrt(
        compressed.square().mean(-1, keepdim=True) + rms_eps
    )
    normed = normed * rms_weight.float()
    cache_pos = rope_position.long().clamp(0, cos_sin_cache.shape[0] - 1)
    cos, sin = cos_sin_cache[cache_pos].float().chunk(2, -1)
    nope, rope = normed.split([dim - rope_dim, rope_dim], -1)
    even, odd = rope[:, 0::2], rope[:, 1::2]
    rotated = torch.stack([even * cos - odd * sin, odd * cos + even * sin], -1)
    result = torch.cat([nope, rotated.flatten(1)], -1)
    return torch.where(has_source[:, None], result, 0).to(output_dtype)


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
    """C4 weighted ReLU index scores and causal logical compact-row IDs.

    Weights include model-side projection/head scaling. Quantized Q/K scales are
    dequantization scales, not additional attention scaling. C128 attends every
    completed compact row and does not use this indexer.
    """
    if compress_ratio != 4 or topk < 1:
        raise ValueError("The compressed indexer requires C4 and positive topk")
    if query.ndim != 3 or weights.shape != query.shape[:2]:
        raise ValueError(
            "Expected [tokens, heads, dim] queries and [tokens, heads] weights"
        )
    if query.dtype == torch.int8 and query_scale is None:
        raise ValueError("INT8 queries require per-token/head scales")
    if key_cache.dtype == torch.int8 and key_scale_cache is None:
        raise ValueError("INT8 keys require per-row scales")
    if query.shape[0] == 0:
        return torch.empty((0, topk), dtype=torch.int32, device=query.device)
    capacity = block_table.shape[1] * key_cache.shape[1]
    # Each query is independent. Bound gathered/dequantized keys and logits
    # instead of materializing the full prefill batch against every candidate.
    query_tile = _query_tile_size(capacity, query.shape[2], query.shape[1])
    if query.shape[0] > query_tile:
        scales = (
            query_scale.reshape(*query.shape[:2], 1)
            if query_scale is not None
            else None
        )
        return torch.cat(
            [
                deepseek_v4_indexer(
                    query[start : start + query_tile],
                    key_cache,
                    weights[start : start + query_tile],
                    block_table,
                    request_ids[start : start + query_tile],
                    positions[start : start + query_tile],
                    key_lengths,
                    topk,
                    compress_ratio,
                    None if scales is None else scales[start : start + query_tile],
                    key_scale_cache,
                )
                for start in range(0, query.shape[0], query_tile)
            ],
            dim=0,
        )
    logical = torch.arange(capacity, device=query.device)[None].expand(
        query.shape[0], -1
    )
    keys, valid = _paged_read(key_cache, block_table, request_ids, logical)
    q = query.float()
    if query_scale is not None:
        q = q * query_scale.reshape(*query.shape[:2], 1).float()
    if key_scale_cache is not None:
        scales, scale_valid = _paged_read(
            key_scale_cache, block_table, request_ids, logical
        )
        keys = keys.float() * scales.float()
        valid = valid & scale_valid
    logits = torch.einsum("thd,tkd->thk", q, keys.float()).relu()
    scores = (logits * weights.float()[:, :, None]).sum(1)
    limit = torch.minimum(
        (positions + 1) // compress_ratio, _lengths(key_lengths, request_ids)
    )
    valid = valid & (positions[:, None] >= 0) & (logical < limit[:, None])
    scores = scores.masked_fill(~valid, -torch.inf)
    if topk > capacity:
        scores = torch.cat(
            [scores, scores.new_full((query.shape[0], topk - capacity), -torch.inf)], -1
        )
    values, indices = scores.topk(topk, dim=-1)
    return torch.where(torch.isfinite(values), indices, -1).to(torch.int32)


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
    """Joint SWA/compact softmax with learned zero-value sink logits.

    Compact rows overlapping the SWA window remain visible. Bounds are per query
    and request, so chunked-prefill queries cannot see later completed groups.
    Passing no compact cache selects the SWA-only path used by uncompressed layers.
    """
    if window_size < 1 or compress_ratio not in (0, 4, 128):
        raise ValueError(
            "Expected positive SWA window and compression ratio 0, 4 or 128"
        )
    dim = query.shape[-1]
    value_dim = dim if value_dim is None else value_dim
    if not 0 < value_dim <= dim:
        raise ValueError("Value width must be within the shared KV width")
    if query.shape[0] == 0:
        return query.new_empty((0, query.shape[1], value_dim))
    if compact_cache is not None and (compress_ratio == 0 or compact_indices is None):
        raise ValueError("Compact attention requires compressed indices and a ratio")
    capacity = window_size + (
        compact_indices.shape[-1] if compact_cache is not None else 0
    )
    query_tile = _query_tile_size(capacity, dim, query.shape[1])
    if query.shape[0] > query_tile:
        return torch.cat(
            [
                deepseek_v4_sparse_attention(
                    query[start : start + query_tile],
                    swa_cache,
                    compact_cache,
                    swa_block_table,
                    compact_block_table,
                    request_ids[start : start + query_tile],
                    positions[start : start + query_tile],
                    swa_lengths,
                    compact_lengths,
                    None
                    if compact_indices is None
                    else compact_indices[start : start + query_tile],
                    window_size,
                    compress_ratio,
                    scale,
                    attn_sink,
                    value_dim,
                )
                for start in range(0, query.shape[0], query_tile)
            ],
            dim=0,
        )
    source = (
        positions[:, None]
        - window_size
        + 1
        + torch.arange(window_size, device=query.device)
    )
    kv, valid = _paged_read(swa_cache, swa_block_table, request_ids, source)
    valid = valid & (source < _lengths(swa_lengths, request_ids)[:, None])
    if compact_cache is not None:
        compact, compact_valid = _paged_read(
            compact_cache, compact_block_table, request_ids, compact_indices
        )
        limit = torch.minimum(
            (positions + 1) // compress_ratio, _lengths(compact_lengths, request_ids)
        )
        compact_valid = compact_valid & (compact_indices < limit[:, None])
        kv = torch.cat([kv, compact], dim=1)
        valid = torch.cat([valid, compact_valid], dim=1)
    valid = valid & (positions[:, None] >= 0)
    # Sanitize invalid physical rows before matmul: zero probability * NaN is NaN.
    kv = torch.where(valid[..., None], kv, 0).float()
    logits = torch.einsum("thd,tkd->thk", query.float(), kv)
    logits = (logits * (dim**-0.5 if scale is None else scale)).masked_fill(
        ~valid[:, None], -torch.inf
    )
    denominator = torch.logsumexp(logits, dim=-1)
    if attn_sink is not None:
        denominator = torch.logaddexp(denominator, attn_sink.float()[None])
    denominator = torch.where(torch.isneginf(denominator), torch.inf, denominator)
    probabilities = torch.exp(logits - denominator[..., None])
    result = torch.einsum("thk,tkd->thd", probabilities, kv[..., :value_dim])
    active = (positions >= 0) & (request_ids >= 0)
    return torch.where(active[:, None, None], result, 0).to(query.dtype)

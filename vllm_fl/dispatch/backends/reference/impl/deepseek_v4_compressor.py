# SPDX-License-Identifier: Apache-2.0
"""PyTorch precision reference for the continuous paged DeepSeek V4 compressor."""

import torch
import torch.nn.functional as F

from .deepseek_v4_ops import deepseek_v4_partial_rope
from .paged_cache import has_disjoint_paged_rows, paged_cache_write


def _byte_span(tensor):
    size = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    begin = tensor.data_ptr()
    return begin, begin + size * tensor.element_size()


def _compress_rows(state, table, requests, ends, weight, rope, ratio, eps):
    """Reduce each completed group, using separate projection lanes for previous and current C4 groups."""
    dim = weight.numel()
    halves = 2 if ratio == 4 else 1
    history = torch.arange(halves * ratio, device=state.device)
    rope_pos = ends - ends.remainder(ratio)
    active = (
        (requests >= 0)
        & (requests < table.shape[0])
        & (ends >= 0)
        & (ends.remainder(ratio) == ratio - 1)
        & (rope_pos < rope.shape[0])
    )
    source = ends.clamp_min(0)[:, None] - (history.numel() - 1 - history)
    logical_page = source.div(state.shape[1], rounding_mode="floor")
    page = table[
        requests.clamp(0, table.shape[0] - 1)[:, None],
        logical_page.clamp(0, table.shape[1] - 1),
    ].long()
    valid = (
        active[:, None]
        & (source >= 0)
        & (logical_page < table.shape[1])
        & (page >= 0)
        & (page < state.shape[0])
    )

    # Address only selected payload lanes, including in large shared allocations.
    span = 1 + sum((n - 1) * s for n, s in zip(state.shape, state.stride()))
    storage = state.as_strided((span,), (1,))
    row = page.clamp(0, state.shape[0] - 1) * state.stride(0) + source.remainder(
        state.shape[1]
    ) * state.stride(1)
    channel = history.div(ratio, rounding_mode="floor")[:, None] * dim + torch.arange(
        dim, device=state.device
    )
    addresses = row[:, :, None] + channel * state.stride(2)
    values = torch.index_select(storage, 0, addresses.flatten()).view_as(addresses)
    score_addresses = addresses + halves * dim * state.stride(2)
    scores = torch.index_select(storage, 0, score_addresses.flatten()).view_as(values)
    values = values.masked_fill(~valid[..., None], 0)
    scores = scores.masked_fill(~valid[..., None], -torch.inf)
    has_history = valid.any(dim=1)
    scores = torch.where(has_history[:, None, None], scores, 0)
    pooled = (values * scores.softmax(dim=1)).sum(dim=1)
    normed = pooled * torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + eps)
    normed = normed * weight.float()
    if rope.shape[1]:
        normed = deepseek_v4_partial_rope(normed, rope_pos, rope)
    return torch.where(has_history[:, None], normed, 0)


def deepseek_v4_compressor(
    hidden_states,
    wkv,
    wgate,
    ape,
    state_cache,
    positions,
    slot_mapping,
    state_block_table,
    output_request_ids,
    output_positions,
    rms_weight,
    cos_sin_cache,
    compress_ratio=4,
    rms_eps=1e-6,
):
    """Save FP32 projections, then return normalized, rotated rows in input dtype."""
    floating = (hidden_states, wkv, wgate, ape, rms_weight, cos_sin_cache)
    metadata = (
        positions,
        slot_mapping,
        state_block_table,
        output_request_ids,
        output_positions,
    )
    tensors = (*floating, *metadata, state_cache)
    if any(
        t.device != hidden_states.device or t.layout != torch.strided for t in tensors
    ):
        raise ValueError("Expected strided compressor tensors on one device")
    if any(
        t.dtype not in (torch.float16, torch.bfloat16, torch.float32) for t in floating
    ):
        raise TypeError("Compressor inputs and weights require FP16, BF16 or FP32")
    if any(t.dtype not in (torch.int32, torch.int64) for t in metadata):
        raise TypeError("Compressor metadata requires INT32 or INT64")
    if not isinstance(rms_eps, (float, int)):
        raise TypeError("RMS epsilon must be a scalar number")
    if compress_ratio not in (4, 128):
        raise ValueError("Compression ratio must be 4 or 128")
    if hidden_states.ndim != 2 or rms_weight.ndim != 1:
        raise ValueError("Expected [tokens, hidden] input and a vector RMS weight")
    n, hidden = hidden_states.shape
    dim = rms_weight.numel()
    width = (2 if compress_ratio == 4 else 1) * dim
    if (
        not dim
        or not hidden
        or wkv.shape != (width, hidden)
        or wgate.shape != wkv.shape
    ):
        raise ValueError("Projection and compressor dimensions do not match")
    if ape.shape != (compress_ratio, width):
        raise ValueError("APE shape must match compression ratio and projection width")
    if (
        state_cache.ndim != 3
        or state_cache.dtype != torch.float32
        or state_cache.shape[1] < 1
        or state_cache.shape[2] != 2 * width
    ):
        raise ValueError("Expected FP32 [pages, block, packed width] compressor state")
    if positions.shape != (n,) or slot_mapping.shape != (n,):
        raise ValueError("Expected one position and state slot per projected token")
    if (
        state_block_table.ndim != 2
        or output_request_ids.ndim != 1
        or output_positions.shape != output_request_ids.shape
    ):
        raise ValueError("Expected a block table and matching output metadata vectors")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] % 2
        or cos_sin_cache.shape[1] > dim
    ):
        raise ValueError("RoPE width must be even and no larger than the head")
    if state_cache.numel():
        if not has_disjoint_paged_rows(state_cache):
            raise ValueError("Expected disjoint compressor state row spans")
        begin, end = _byte_span(state_cache)
        for tensor in (*floating, *metadata):
            if tensor.numel():
                start, stop = _byte_span(tensor)
                if begin < stop and start < end:
                    raise ValueError("State cache must not overlap compressor inputs")

    if n and state_cache.numel():
        x = hidden_states.float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            kv = F.linear(x, wkv.float())
            gate = F.linear(x, wgate.float())
        gate = gate + ape[positions.long() % compress_ratio].float()
        # The writer receives independent slots even for disjoint shared-buffer views.
        paged_cache_write(torch.cat((kv, gate), -1), slot_mapping.clone(), state_cache)

    rows = output_positions.numel()
    if (
        not rows
        or not state_cache.numel()
        or not state_block_table.numel()
        or not cos_sin_cache.shape[0]
    ):
        return hidden_states.new_zeros((rows, dim))
    # Bound each device gather to at most 32 MiB of int64 addresses per tile.
    history = 8 if compress_ratio == 4 else 128
    tile = max(1, min(32, (4 * 1024 * 1024) // (dim * history)))
    return torch.cat(
        [
            _compress_rows(
                state_cache,
                state_block_table,
                output_request_ids[start : start + tile].long(),
                output_positions[start : start + tile].long(),
                rms_weight,
                cos_sin_cache,
                compress_ratio,
                rms_eps,
            ).to(hidden_states.dtype)
            for start in range(0, rows, tile)
        ]
    )

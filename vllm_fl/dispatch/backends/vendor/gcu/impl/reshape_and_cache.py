# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import sys

import torch
from flag_gems.fused.reshape_and_cache_flash import (
    reshape_and_cache_flash as _gems_reshape_and_cache_flash,
)

_INT32_MAX = 2**31 - 1
_MAX_GRID_X = 65535


def _slots_as_int32(slot_mapping: torch.Tensor) -> torch.Tensor:
    if slot_mapping.dtype == torch.int32:
        return slot_mapping
    if sys.byteorder != "little":
        raise RuntimeError("GCU slot conversion requires little-endian storage")
    # Valid positive slots fit int32 (checked from the cache metadata below).
    # Inspect the int64 sign via its high word, preserving all negative
    # sentinels without unsupported native int64 clamp or float conversion.
    words = slot_mapping.view(torch.int32).view(-1, 2)
    return torch.where(words[:, 1] < 0, -1, words[:, 0])


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Write paged K/V using FlagGems' Enflame-compatible cache kernel.

    Native GCU advanced assignment rejects the interleaved K/V block layout
    used by vLLM. FlagGems accepts its padded block stride and skips negative
    slots. Its Enflame path uses int32 indexing, so validate layout and index
    arithmetic from metadata before launching. Nonnegative slots must already
    be valid cache indices, as required by the upstream cache-write contract.
    """
    if kv_cache_dtype not in ("auto", "float16", "bfloat16", "float32"):
        raise NotImplementedError("GCU cache writes do not support quantized KV cache")
    tensors = (key, value, key_cache, value_cache)
    if key.dtype not in (torch.float16, torch.bfloat16, torch.float32) or any(
        t.dtype != key.dtype for t in tensors
    ):
        raise ValueError("GCU cache writes require matching floating-point dtypes")
    if any(t.device != key.device for t in (*tensors, slot_mapping)):
        raise ValueError("GCU cache tensors and slots must be on the same device")
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError("GCU cache keys and values must have rank 3")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("GCU key/value caches must have matching rank-4 shapes")
    if (
        slot_mapping.ndim != 1
        or slot_mapping.stride(0) != 1
        or slot_mapping.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("GCU cache slots must be contiguous int32 or int64")

    num_tokens = slot_mapping.numel()
    blocks, block_size, heads, head_size = key_cache.shape
    if min(blocks, block_size, heads, head_size) <= 0:
        raise ValueError("GCU cache dimensions must be positive")
    width = heads * head_size
    block_elements = block_size * width
    for t in (key, value):
        if (
            t.shape[0] < num_tokens
            or t.shape[1:] != (heads, head_size)
            or t.stride()[1:] != (head_size, 1)
            or t.stride(0) < width
        ):
            raise ValueError("GCU cache inputs require contiguous heads within tokens")
    for t in (key_cache, value_cache):
        if (
            t.stride()[1:] != (width, head_size, 1)
            or t.stride(0) < block_elements
            or t.stride(0) != key_cache.stride(0)
        ):
            raise ValueError(
                "GCU caches require matching strides and contiguous blocks"
            )
    if (
        blocks * block_size - 1 > _INT32_MAX
        or key_cache.stride(0) > _INT32_MAX
        or (blocks - 1) * key_cache.stride(0) + block_elements - 1 > _INT32_MAX
        or any(
            t.stride(0) > _INT32_MAX
            or (min(num_tokens, _MAX_GRID_X) - 1) * t.stride(0) + width - 1 > _INT32_MAX
            for t in (key, value)
        )
    ):
        raise ValueError("GCU cache indexing exceeds the FlagGems int32 range")
    if num_tokens == 0:
        return

    # In particular, -2**32 must not wrap to slot zero. The int32 operations
    # stay on device and are captured together with the cache-write kernel.
    slots = _slots_as_int32(slot_mapping)
    # Bound the GCU launch grid. Narrowing only changes metadata and keeps
    # all calls compatible with graph capture, including long prefills.
    for start in range(0, num_tokens, _MAX_GRID_X):
        count = min(num_tokens - start, _MAX_GRID_X)
        _gems_reshape_and_cache_flash(
            key.narrow(0, start, count),
            value.narrow(0, start, count),
            key_cache,
            value_cache,
            slots.narrow(0, start, count),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

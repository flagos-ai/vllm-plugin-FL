# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference for writes to strided paged caches."""

import torch


def has_disjoint_paged_rows(cache):
    """Check packed row spans without walking the physical page allocation."""
    pages, block, width = cache.shape
    page_stride, token_stride, channel_stride = cache.stride()
    if any(
        size > 1 and stride <= 0
        for size, stride in zip(
            cache.shape, (page_stride, token_stride, channel_stride)
        )
    ):
        return False
    row_span = (width - 1) * channel_stride + 1
    if pages == 1:
        return block == 1 or token_stride >= row_span
    if block == 1:
        return page_stride >= row_span
    if (
        token_stride >= row_span
        and page_stride >= (block - 1) * token_stride + row_span
    ):
        return True
    if page_stride >= row_span and token_stride >= (pages - 1) * page_stride + row_span:
        return True

    if pages > block:
        pages, block = block, pages
        page_stride, token_stride = token_stride, page_stride
    if token_stride < row_span:
        return False
    # For each displacement on the shorter axis, the nearest feasible
    # displacement on the other axis minimizes the distance between rows.
    bound = block - 1
    for delta in range(1, pages):
        distance = delta * page_stride
        lower = min(distance // token_stride, bound)
        upper = min(lower + 1, bound)
        gap = min(
            abs(distance - lower * token_stride), abs(distance - upper * token_stride)
        )
        if gap < row_span:
            return False
    return True


def paged_cache_write(values, slots, cache):
    """Write valid token rows without reshaping or copying the paged cache."""
    if (
        cache.ndim != 3
        or values.ndim != 2
        or values.shape[1] != cache.shape[-1]
        or slots.shape != values.shape[:1]
    ):
        raise ValueError(
            "Expected matching [tokens, dim], [tokens], [pages, block, dim]"
        )
    inputs = (values, slots)
    if any(t.device != cache.device for t in inputs):
        raise ValueError("Cache, slots and values must use the same device")
    if (
        cache.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.int8)
        or slots.dtype not in (torch.int32, torch.int64)
        or (
            values.dtype != cache.dtype
            and not (
                values.dtype == torch.float32
                and cache.dtype in (torch.bfloat16, torch.float16)
            )
        )
    ):
        raise NotImplementedError("Unsupported paged cache dtype")
    if any(t.layout != torch.strided for t in (cache, *inputs)):
        raise NotImplementedError("Paged cache write needs strided tensors")
    if not values.numel():
        return
    pages, block_size, width = cache.shape
    if not pages or not block_size:
        raise ValueError("Cannot write nonempty values to an empty cache")
    if not has_disjoint_paged_rows(cache):
        raise NotImplementedError("Paged cache requires disjoint row spans")
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(cache.shape, cache.stride())
    )
    if any(
        t.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        for t in inputs
    ):
        raise ValueError("Paged cache storage must not alias its inputs")

    num_slots = pages * block_size
    valid = (slots >= 0) & (slots < num_slots)
    indices = slots.long().clamp(0, num_slots - 1)
    storage = cache.as_strided((span,), (1,))
    channels = torch.arange(width, device=cache.device) * cache.stride(2)
    offsets = indices.div(block_size, rounding_mode="floor") * cache.stride(0) + (
        indices % block_size
    ) * cache.stride(1)
    for token in range(values.shape[0]):
        addresses = offsets[token] + channels
        previous = torch.index_select(storage, 0, addresses)
        update = torch.where(valid[token], values[token].to(cache.dtype), previous)
        # Serial writes preserve valid rows when padding aliases a real slot.
        storage.index_put_((addresses,), update)

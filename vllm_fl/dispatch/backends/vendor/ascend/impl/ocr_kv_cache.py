# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct BF16 writes to the paged, head-major Unlimited-OCR cache."""

import torch


def expand_head_major_slots(
    slots: torch.Tensor, num_kv_heads: int, block_size: int
) -> torch.Tensor:
    """Build ATB int32 slots in token/head order on the current stream.

    The runner supplies distinct, in-range live token slots and negative
    padding. A live slot p * B + s maps to (p * H + h) * B + s for head h.
    The caller can share the resulting indices across layers in the same step.
    """
    if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("OCR KV slots must be a vector of int32 or int64 indices")
    if num_kv_heads <= 0 or block_size <= 0:
        raise ValueError("OCR KV head count and page size must be positive")
    num_rows = slots.numel() * num_kv_heads
    if num_rows > torch.iinfo(torch.int32).max:
        raise ValueError("OCR KV scatter row count exceeds the ATB int32 range")
    logical = slots.to(torch.int32)
    if num_rows == 0:
        return logical

    page = torch.div(logical, block_size, rounding_mode="trunc")
    first_head_slot = torch.add(logical, page, alpha=(num_kv_heads - 1) * block_size)
    head_offsets = torch.arange(
        0, num_kv_heads * block_size, block_size, device=slots.device, dtype=torch.int32
    )
    live_slots = first_head_slot[:, None] + head_offsets
    # ATB skips negative slots. Use a distinct negative index for every
    # padded token/head so all slot indices satisfy the uniqueness contract.
    pad_slots = torch.arange(
        -1, -num_rows - 1, -1, device=slots.device, dtype=torch.int32
    ).view_as(live_slots)
    return torch.where(logical[:, None] < 0, pad_slots, live_slots).view(-1)


def write_head_major_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    expanded_slots: torch.Tensor,
) -> None:
    """Write current [T,H,D] rows into [P,H,B,D] K and V pools in place.

    The ATB writer sees each (page, head) as a one-head page of B tokens.
    All four views share their inputs' storage, including storage offsets.
    """
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("OCR KV pools must have matching [page,head,slot,dim] shapes")
    pages, heads, block_size, head_size = key_cache.shape
    if min(pages, heads, block_size, head_size) <= 0:
        raise ValueError("Head-major OCR KV pools require positive dimensions")
    if key.ndim != 3 or value.shape != key.shape or key.shape[1:] != (heads, head_size):
        raise ValueError("OCR K and V rows must match the cache head count and size")
    if pages * heads * block_size > torch.iinfo(torch.int32).max + 1:
        raise ValueError("OCR KV pool exceeds the ATB int32 slot range")
    rows = key.shape[0] * heads
    if expanded_slots.shape != (rows,) or expanded_slots.dtype != torch.int32:
        raise ValueError("OCR KV scatter requires one int32 index per token and head")
    buffers = (key, value, key_cache, value_cache)
    if any(t.dtype != torch.bfloat16 for t in buffers):
        raise ValueError("Head-major OCR KV writes require BF16 keys, values and pools")
    if any(
        t.device != key.device or not t.is_contiguous()
        for t in (*buffers, expanded_slots)
    ):
        raise ValueError("OCR KV scatter needs contiguous tensors on the same device")
    if rows == 0:
        return

    import torch_npu

    token_shape = (rows, 1, head_size)
    page_shape = (pages * heads, block_size, 1, head_size)
    torch_npu._npu_reshape_and_cache(
        key=key.view(token_shape),
        value=value.view(token_shape),
        key_cache=key_cache.view(page_shape),
        value_cache=value_cache.view(page_shape),
        slot_indices=expanded_slots,
    )

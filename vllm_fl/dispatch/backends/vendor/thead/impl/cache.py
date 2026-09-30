# Copyright (c) 2026 BAAI. All rights reserved.

"""Graph-safe, unquantized KV cache writes for the THead PPU backend."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _reshape_and_cache_flash_kernel_thead(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    KEY_TOKEN_STRIDE: tl.constexpr,
    KEY_HEAD_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    VALUE_TOKEN_STRIDE: tl.constexpr,
    VALUE_HEAD_STRIDE: tl.constexpr,
    VALUE_DIM_STRIDE: tl.constexpr,
    KEY_CACHE_BLOCK_STRIDE: tl.constexpr,
    KEY_CACHE_TOKEN_STRIDE: tl.constexpr,
    KEY_CACHE_HEAD_STRIDE: tl.constexpr,
    KEY_CACHE_DIM_STRIDE: tl.constexpr,
    VALUE_CACHE_BLOCK_STRIDE: tl.constexpr,
    VALUE_CACHE_TOKEN_STRIDE: tl.constexpr,
    VALUE_CACHE_HEAD_STRIDE: tl.constexpr,
    VALUE_CACHE_DIM_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    TILES_PER_TOKEN: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
):
    program = tl.program_id(0)
    token = (program // TILES_PER_TOKEN).to(tl.int64)
    tile = program % TILES_PER_TOKEN
    slot = tl.load(slot_mapping + token * SLOT_STRIDE).to(tl.int64)
    valid_slot = (slot >= 0) & (slot < NUM_BLOCKS * BLOCK_SIZE)

    # A safe address alone is insufficient: padding must never store to slot 0.
    # Mask the loads/stores on the device without compacting the token tensors.
    safe_slot = tl.where(valid_slot, slot, 0)
    block = safe_slot // BLOCK_SIZE
    position = safe_slot % BLOCK_SIZE
    element = (tile * BLOCK_ELEMENTS + tl.arange(0, BLOCK_ELEMENTS)).to(tl.int64)
    head = element // HEAD_SIZE
    dim = element % HEAD_SIZE
    mask = valid_slot & (element < NUM_HEADS * HEAD_SIZE)

    key_value = tl.load(
        key + token * KEY_TOKEN_STRIDE + head * KEY_HEAD_STRIDE + dim * KEY_DIM_STRIDE,
        mask=mask,
        other=0,
    )
    value_value = tl.load(
        value
        + token * VALUE_TOKEN_STRIDE
        + head * VALUE_HEAD_STRIDE
        + dim * VALUE_DIM_STRIDE,
        mask=mask,
        other=0,
    )
    tl.store(
        key_cache
        + block * KEY_CACHE_BLOCK_STRIDE
        + position * KEY_CACHE_TOKEN_STRIDE
        + head * KEY_CACHE_HEAD_STRIDE
        + dim * KEY_CACHE_DIM_STRIDE,
        key_value,
        mask=mask,
    )
    tl.store(
        value_cache
        + block * VALUE_CACHE_BLOCK_STRIDE
        + position * VALUE_CACHE_TOKEN_STRIDE
        + head * VALUE_CACHE_HEAD_STRIDE
        + dim * VALUE_CACHE_DIM_STRIDE,
        value_value,
        mask=mask,
    )


def reshape_and_cache_flash_thead(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Copy mapped tokens into logical ``[blocks, block_size, heads, dim]`` caches.

    The physical cache layout may be NHD, HND, or another noncontiguous view.
    All strides are respected independently for keys and values. Negative slots
    are padding and leave the cache untouched; out-of-range slots are also masked
    to prevent out-of-bounds stores. Valid slots must be unique within one call,
    as in the vLLM cache op. Only ``slot_mapping.numel()`` source rows are used.

    Validation reads tensor metadata only. The kernel uses no CPU synchronization,
    data-dependent shapes, or cache-sized temporaries, so warmed-up launches can
    be captured and replayed with different slot mapping contents in a CUDA graph.
    Quantized cache formats require scale conversion and are deliberately rejected.
    """
    del k_scale, v_scale  # Unquantized cache writes do not apply scales.
    cache_dtypes = {
        "auto": None,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if kv_cache_dtype not in cache_dtypes:
        raise NotImplementedError(
            f"THead KV cache writes do not support cache dtype {kv_cache_dtype!r}; "
            "use auto, fp16, or bf16"
        )
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError("THead keys and values must have shape [tokens, heads, dim]")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError(
            "THead caches must have shape [blocks, block_size, heads, dim]"
        )
    if slot_mapping.ndim != 1 or slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("THead slot_mapping must be a one-dimensional integer tensor")
    num_tokens = slot_mapping.numel()
    if num_tokens > key.shape[0] or num_tokens > value.shape[0]:
        raise ValueError("THead slot_mapping cannot exceed the key/value token count")
    if key.shape[1:] != value.shape[1:]:
        raise ValueError("THead keys and values must have the same heads and head size")
    if key_cache.shape != value_cache.shape or key_cache.shape[2:] != key.shape[1:]:
        raise ValueError("THead key/value cache shapes must match their source heads")
    if key_cache.shape[1] <= 0 or key.shape[1] <= 0 or key.shape[2] <= 0:
        raise ValueError("THead block size, head count, and head size must be positive")
    tensors = (key, value, key_cache, value_cache)
    if any(t.dtype not in (torch.float16, torch.bfloat16) for t in tensors):
        raise NotImplementedError("THead KV cache writes require FP16 or BF16 tensors")
    expected_cache_dtype = cache_dtypes[kv_cache_dtype]
    if expected_cache_dtype is not None and (
        key_cache.dtype != expected_cache_dtype
        or value_cache.dtype != expected_cache_dtype
    ):
        raise ValueError("THead cache tensor dtype does not match kv_cache_dtype")
    if key_cache.dtype != value_cache.dtype:
        raise ValueError("THead key and value caches must have the same dtype")
    if not key.is_cuda or any(t.device != key.device for t in (*tensors, slot_mapping)):
        raise ValueError(
            "THead cache writes require all tensors on the same CUDA device"
        )
    if num_tokens == 0:
        return

    block_elements = 256
    tiles_per_token = triton.cdiv(key.shape[1] * key.shape[2], block_elements)
    _reshape_and_cache_flash_kernel_thead[(num_tokens * tiles_per_token,)](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        *key.stride(),
        *value.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        slot_mapping.stride(0),
        key_cache.shape[0],
        key_cache.shape[1],
        key.shape[1],
        key.shape[2],
        tiles_per_token,
        block_elements,
        num_warps=4,
    )

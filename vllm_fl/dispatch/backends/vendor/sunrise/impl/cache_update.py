# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Stride-aware KV writes for vLLM 0.32 combined K/V cache storage.

The pinned FlagGems-vllm writer assumes dense token/head dimensions. Native
0.32 cache views interleave K and V per token, so pass all tensor strides here.
"""

import triton
import triton.language as tl


@triton.jit
def _write_kv_cache(
    key,
    value,
    key_cache,
    value_cache,
    slots,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    kc0: tl.constexpr,
    kc1: tl.constexpr,
    kc2: tl.constexpr,
    kc3: tl.constexpr,
    vc0: tl.constexpr,
    vc1: tl.constexpr,
    vc2: tl.constexpr,
    vc3: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, WIDTH)
    slot = tl.load(slots + token)
    mask = (slot >= 0) & (dim < HEAD_SIZE)
    block = slot // BLOCK_SIZE
    offset = slot % BLOCK_SIZE
    k = tl.load(key + token * ks0 + head * ks1 + dim * ks2, mask=mask, other=0)
    v = tl.load(value + token * vs0 + head * vs1 + dim * vs2, mask=mask, other=0)
    tl.store(key_cache + block * kc0 + offset * kc1 + head * kc2 + dim * kc3, k, mask)
    tl.store(value_cache + block * vc0 + offset * vc1 + head * vc2 + dim * vc3, v, mask)


def write_kv_cache(key, value, key_cache, value_cache, slot_mapping):
    """Update valid slots in strided [blocks, tokens, heads, dim] K/V views."""
    if slot_mapping.numel() == 0:
        return
    if key.shape[1:] != key_cache.shape[2:] or value.shape != key.shape:
        raise ValueError("Sunrise KV input and cache head dimensions must match")
    if key_cache.shape != value_cache.shape:
        raise ValueError("Sunrise K/V caches must have equal shapes")
    _write_kv_cache[(slot_mapping.numel(), key.shape[1])](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        *key.stride(),
        *value.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        HEAD_SIZE=key.shape[2],
        BLOCK_SIZE=key_cache.shape[1],
        WIDTH=triton.next_power_of_2(key.shape[2]),
        num_warps=4,
    )

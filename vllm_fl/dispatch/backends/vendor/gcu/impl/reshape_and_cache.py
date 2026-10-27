# Copyright (c) 2026 BAAI. All rights reserved.
"""GCU Triton fixed-grid implementation for reshape_and_cache_flash.

Replaces the PyTorch-native implementation with a Triton kernel using a
fixed grid (capped at GCU_NUM_GRID) + strided loop.  slot_mapping is int64;
we cast to int32 inside the kernel to avoid GCU int64 div/mod issues (slot
values fit in int32 for all practical block counts).
"""
from __future__ import annotations
import logging
import torch
from vllm.triton_utils import tl, triton
logger = logging.getLogger(__name__)
GCU_NUM_GRID = 48
GCU_MAX_WARPS = 4

@triton.jit(do_not_specialize=["N"])
def _reshape_and_cache_flash_kernel_gcu(
    key, value, key_cache, value_cache, slot_mapping, N,
    BLOCK_SIZE: tl.constexpr, KV_DIM: tl.constexpr, BLOCK_KV: tl.constexpr,
    KC_S0: tl.constexpr, KC_S1: tl.constexpr, K_S0: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    pid = tl.program_id(0)
    for i in range(pid, N, NUM_SPC):
        # slot_mapping is int64.  int64 div/mod is unavailable on GCU300 (see
        # env-vllm-plugin-fl skill section 2.1), so cast to int32 first and then
        # decompose the address; slot values fit in int32 (the block count
        # never exceeds 2^31).
        slot = tl.load(slot_mapping + i).to(tl.int32)
        if slot >= 0:
            bi = slot // BLOCK_SIZE
            bo = slot % BLOCK_SIZE
            # Address offsets must use int64: the flat offset into the KV cache
            # (bi * stride0) exceeds the int32 range (2^31) for large block
            # numbers, and int32 would silently overflow and write to the wrong
            # location.  Only div/mod use int32 (to avoid the GCU300 int64
            # div/mod restriction).
            sb = i.to(tl.int64) * K_S0
            db = bi.to(tl.int64) * KC_S0 + bo.to(tl.int64) * KC_S1
            for ko in range(0, KV_DIM, BLOCK_KV):
                off = ko + tl.arange(0, BLOCK_KV)
                mask = off < KV_DIM
                kv = tl.load(key + sb + off, mask=mask)
                vv = tl.load(value + sb + off, mask=mask)
                tl.store(key_cache + db + off, kv, mask=mask)
                tl.store(value_cache + db + off, vv, mask=mask)


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
    """Write per-token K/V into the paged KV cache (Triton fixed-grid)."""
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    key = key[:num_tokens]
    value = value[:num_tokens]
    block_size = key_cache.size(1)
    kv_dim = key.shape[1] * key.shape[2]
    key = key.contiguous()
    value = value.contiguous()
    block_kv = triton.next_power_of_2(kv_dim)
    _reshape_and_cache_flash_kernel_gcu[(min(num_tokens, GCU_NUM_GRID),)](
        key, value, key_cache, value_cache, slot_mapping, num_tokens,
        BLOCK_SIZE=block_size, KV_DIM=kv_dim, BLOCK_KV=block_kv,
        KC_S0=key_cache.stride(0), KC_S1=key_cache.stride(1), K_S0=key.stride(0),
        NUM_SPC=GCU_NUM_GRID, num_warps=GCU_MAX_WARPS)

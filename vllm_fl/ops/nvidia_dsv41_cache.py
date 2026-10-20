# Copyright (c) 2026 BAAI. All rights reserved.

"""vLLM 0.32's Hopper 584-byte cache contract using the real Gems writer."""

import torch
import triton
import triton.language as tl

from vllm_fl.ops.nvidia_dsv41_indexer import _same_cuda


@triton.jit
def _prepare_query(
    Q,
    OUT,
    POSITIONS,
    ROPE,
    HEADS: tl.constexpr,
    PADDED: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    ROTATE: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    dims = tl.arange(0, 512)
    target = OUT + (token * PADDED + head) * 512
    if head >= HEADS:
        tl.store(target + dims, 0)
        return
    source = Q + (token * HEADS + head) * 512
    values = tl.load(source + dims).to(tl.float32)
    inverse = 1.0
    if NORM:
        inverse = tl.rsqrt(tl.sum(values * values, 0) / 512.0 + EPS)
        values = values * inverse
    if ROTATE:
        partner = tl.load(source + (dims ^ 1)).to(tl.float32) * inverse
        position = tl.load(POSITIONS + token)
        half = (dims - 448) // 2
        cosine = tl.load(ROPE + position * 64 + half, dims >= 448, other=1)
        sine = tl.load(ROPE + position * 64 + 32 + half, dims >= 448, other=0)
        rotated = values * cosine + tl.where(dims % 2 == 0, -partner, partner) * sine
        values = tl.where(dims >= 448, rotated, values)
    tl.store(target + dims, values)


def _validate_cache_inputs(kv, cache, slots, positions, rope, block):
    _same_cuda(kv, cache, slots, positions, rope)
    if (
        kv.dtype != torch.bfloat16
        or kv.ndim != 2
        or kv.shape[1] != 512
        or not kv.is_contiguous()
    ):
        raise NotImplementedError(
            "Hopper DSV4.1 cache requires contiguous BF16 KV [tokens, 512]"
        )
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 2
        or cache.shape[1] != block * 584
        or cache.stride(1) != 1
    ):
        raise ValueError(
            "Hopper DSV4.1 cache requires packed uint8 pages of block*584 bytes"
        )
    if block <= 0 or cache.stride(0) < block * 584:
        raise ValueError("Invalid Hopper DSV4.1 cache block size or page stride")
    if (
        slots.dtype != torch.int64
        or slots.ndim != 1
        or not slots.is_contiguous()
        or slots.numel() > kv.shape[0]
    ):
        raise ValueError(
            "Slot mapping must be contiguous int64 and no longer than KV rows"
        )
    if (
        positions.dtype != torch.int64
        or positions.shape != (kv.shape[0],)
        or not positions.is_contiguous()
    ):
        raise ValueError("Positions must be contiguous int64 [tokens]")
    if (
        rope.dtype != torch.float32
        or rope.ndim != 2
        or rope.shape[1] != 64
        or not rope.is_contiguous()
    ):
        raise ValueError("RoPE cache must be contiguous FP32 [positions, 64]")


def _insert_kv(kv, cache, slots, positions, rope, block):
    from flag_gems.fused.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
        fused_qnorm_rope_kv_insert_kernel,
    )

    if slots.numel():
        # num_heads=0 selects only the existing KV branch. That branch never
        # reads/writes the Q argument and preserves input KV. It implements
        # GPT-J RoPE, 7 UE8M0 scale groups, the zero scale pad and negative slots.
        fused_qnorm_rope_kv_insert_kernel[(kv.shape[0],)](
            kv,
            kv,
            cache,
            slots,
            positions,
            rope,
            0.0,
            block,
            kv.shape[0],
            0,
            cache.stride(0),
            slots.numel(),
        )


def qnorm_rope_kv_rope_quant_insert(
    q_in,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    q_head_padded,
    eps,
    cache_block_size,
    apply_q_norm=True,
    kv_mxfp8=False,
    apply_q_rope=True,
    is_q_interleaved=False,
):
    if kv_mxfp8 or is_q_interleaved:
        raise NotImplementedError(
            "Hopper DSV4.1 supports the 584-byte cache and head-major Q only"
        )
    _validate_cache_inputs(
        kv, k_cache, slot_mapping, position_ids, cos_sin_cache, cache_block_size
    )
    _same_cuda(q_in, kv)
    if (
        q_in.dtype != torch.bfloat16
        or q_in.ndim != 3
        or q_in.shape[0] != kv.shape[0]
        or q_in.shape[2] != 512
        or not q_in.is_contiguous()
    ):
        raise ValueError("Q must be contiguous BF16 [tokens, heads, 512]")
    if q_head_padded < 0 or (q_head_padded and q_head_padded < q_in.shape[1]):
        raise ValueError("Padded Q heads must be zero or at least the live head count")
    result = q_in.new_empty(
        (q_in.shape[0], q_head_padded, 512) if q_head_padded else (0,)
    )
    if q_head_padded and q_in.shape[0]:
        _prepare_query[(q_in.shape[0], q_head_padded)](
            q_in,
            result,
            position_ids,
            cos_sin_cache,
            q_in.shape[1],
            q_head_padded,
            eps,
            apply_q_norm,
            apply_q_rope,
        )
    _insert_kv(kv, k_cache, slot_mapping, position_ids, cos_sin_cache, cache_block_size)
    return result


def kv_rope_insert(
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    cache_block_size,
    fp8_scale=None,
    kv_mxfp8=False,
):
    if fp8_scale is not None or kv_mxfp8:
        raise NotImplementedError(
            "Hopper DSV4.1 supports only the 584-byte UE8M0 cache"
        )
    if k_cache.ndim != 3 or k_cache.shape[1:] != (cache_block_size, 584):
        raise ValueError("KV-only insert requires [pages, block, 584] cache")
    pages = k_cache.view(k_cache.shape[0], -1)
    _validate_cache_inputs(
        kv, pages, slot_mapping, position_ids, cos_sin_cache, cache_block_size
    )
    _insert_kv(kv, pages, slot_mapping, position_ids, cos_sin_cache, cache_block_size)

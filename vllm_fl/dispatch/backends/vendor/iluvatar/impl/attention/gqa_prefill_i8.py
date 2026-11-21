# Copyright (c) 2026 BAAI. All rights reserved.
"""GQA-packed paged prefill attention over the K8V5 cache (Triton fallback when batch context
exceeds the shared BF16 scratch pool).
"""

import torch
import triton
import triton.language as tl

from .kv_i8 import K_ROW_BYTES, ROW_SLOTS, V_ROW_BYTES


@triton.jit
def _find_seq_idx(cu_seqlens_q_ptr, q_block_global_idx, num_seqs, origin, BLOCK_Q: tl.constexpr):
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = (tl.load(cu_seqlens_q_ptr + mid) - origin) // BLOCK_Q + mid
        if val <= q_block_global_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _dequant_v_group32(vp, g: tl.constexpr, offs_lo, sh_lo, offs_hi, sh_hi, kmask):
    v_lo = tl.load(vp + (16 * g) + offs_lo[None, :], mask=kmask[:, None], other=0).to(tl.int32)
    v_hi = tl.load(vp + (64 + 4 * g) + offs_hi[None, :], mask=kmask[:, None], other=0).to(tl.int32)
    sv = tl.load(
        (vp + 80).to(tl.pointer_type(tl.bfloat16)) + g,
        mask=kmask[:, None],
        other=0.0,
    ).to(tl.float32)
    u5 = ((v_lo >> sh_lo[None, :]) & 0xF) | (((v_hi >> sh_hi[None, :]) & 1) << 4)
    return ((u5 - 16).to(tl.float32) * sv).to(tl.bfloat16)


@triton.jit
def _gqa_prefill_k8v5_kernel(
    q_ptr, kv_u8_ptr, o_ptr,
    cu_seqlens_q_ptr, seq_lens_ptr, block_table_ptr,
    scale_log2e,
    num_seqs,
    stride_qt, stride_qh,
    stride_ot, stride_oh,
    stride_kb,               # BYTES per block
    stride_bt,
    v_block_off: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    q_block_global = tl.program_id(0)
    kv_head = tl.program_id(1)

    q_origin = tl.load(cu_seqlens_q_ptr)
    seq_idx = _find_seq_idx(cu_seqlens_q_ptr, q_block_global, num_seqs, q_origin, BLOCK_Q)
    q_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    q_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    q_len = q_end - q_start
    q_block_local = q_block_global - ((q_start - q_origin) // BLOCK_Q + seq_idx)
    if q_block_local * BLOCK_Q >= q_len:
        return
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - q_len

    offs_m = tl.arange(0, BLOCK_M)
    offs_32 = tl.arange(0, 32)
    offs_half = offs_32 // 16
    offs_lo = offs_32 // 2
    sh_lo = (offs_32 & 1) * 4
    offs_hi = offs_32 // 8
    sh_hi = offs_32 & 7
    offs_n = tl.arange(0, TILE_N)

    qpos = q_block_local * BLOCK_Q + offs_m // G
    head = kv_head * G + offs_m % G
    q_valid = qpos < q_len
    qpos_c = tl.minimum(qpos, q_len - 1)
    q_rows = (q_start + qpos_c).to(tl.int64) * stride_qt + head.to(tl.int64) * stride_qh
    q0 = (tl.load(q_ptr + q_rows[:, None] + offs_32[None, :], mask=q_valid[:, None], other=0.0).to(tl.float32) * scale_log2e).to(tl.bfloat16)
    q1 = (tl.load(q_ptr + q_rows[:, None] + 32 + offs_32[None, :], mask=q_valid[:, None], other=0.0).to(tl.float32) * scale_log2e).to(tl.bfloat16)
    q2 = (tl.load(q_ptr + q_rows[:, None] + 64 + offs_32[None, :], mask=q_valid[:, None], other=0.0).to(tl.float32) * scale_log2e).to(tl.bfloat16)
    q3 = (tl.load(q_ptr + q_rows[:, None] + 96 + offs_32[None, :], mask=q_valid[:, None], other=0.0).to(tl.float32) * scale_log2e).to(tl.bfloat16)

    q_abs = context_len + qpos
    kv_hi = context_len + tl.minimum(q_len, (q_block_local + 1) * BLOCK_Q)
    kv_lo_mask = context_len + q_block_local * BLOCK_Q
    n_full = kv_lo_mask // TILE_N
    n_tiles = (kv_hi + TILE_N - 1) // TILE_N

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)

    bt_base = block_table_ptr + seq_idx.to(tl.int64) * stride_bt

    for j in range(0, n_tiles):
        key = j * TILE_N + offs_n
        kmask = key < kv_hi
        blk = tl.load(bt_base + key // BLOCK_SIZE, mask=kmask, other=0).to(tl.int64)
        idx = (key % BLOCK_SIZE).to(tl.int64) * NUM_KV_HEADS + kv_head.to(tl.int64)
        blk_off = blk * stride_kb
        kp = kv_u8_ptr + (blk_off + idx * 144)[:, None]
        vp = kv_u8_ptr + (blk_off + v_block_off + idx * 88)[:, None]
        ksp = (kp + 128).to(tl.pointer_type(tl.bfloat16))

        sk0 = tl.load(ksp + offs_half[None, :], mask=kmask[:, None], other=0.0).to(tl.float32)
        k0 = (tl.load(kp + offs_32[None, :], mask=kmask[:, None], other=0).to(tl.int8, bitcast=True).to(tl.float32) * sk0).to(tl.bfloat16)
        s = tl.dot(q0, tl.trans(k0))

        sk1 = tl.load(ksp + (2 + offs_half)[None, :], mask=kmask[:, None], other=0.0).to(tl.float32)
        k1 = (tl.load(kp + 32 + offs_32[None, :], mask=kmask[:, None], other=0).to(tl.int8, bitcast=True).to(tl.float32) * sk1).to(tl.bfloat16)
        s += tl.dot(q1, tl.trans(k1))

        sk2 = tl.load(ksp + (4 + offs_half)[None, :], mask=kmask[:, None], other=0.0).to(tl.float32)
        k2 = (tl.load(kp + 64 + offs_32[None, :], mask=kmask[:, None], other=0).to(tl.int8, bitcast=True).to(tl.float32) * sk2).to(tl.bfloat16)
        s += tl.dot(q2, tl.trans(k2))

        sk3 = tl.load(ksp + (6 + offs_half)[None, :], mask=kmask[:, None], other=0.0).to(tl.float32)
        k3 = (tl.load(kp + 96 + offs_32[None, :], mask=kmask[:, None], other=0).to(tl.int8, bitcast=True).to(tl.float32) * sk3).to(tl.bfloat16)
        s += tl.dot(q3, tl.trans(k3))

        if j >= n_full:
            causal = key[None, :] <= q_abs[:, None]
            s = tl.where(causal & kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p16 = p.to(tl.bfloat16)

        v0 = _dequant_v_group32(vp, 0, offs_lo, sh_lo, offs_hi, sh_hi, kmask)
        acc0 = acc0 * alpha[:, None] + tl.dot(p16, v0)
        v1 = _dequant_v_group32(vp, 1, offs_lo, sh_lo, offs_hi, sh_hi, kmask)
        acc1 = acc1 * alpha[:, None] + tl.dot(p16, v1)
        v2 = _dequant_v_group32(vp, 2, offs_lo, sh_lo, offs_hi, sh_hi, kmask)
        acc2 = acc2 * alpha[:, None] + tl.dot(p16, v2)
        v3 = _dequant_v_group32(vp, 3, offs_lo, sh_lo, offs_hi, sh_hi, kmask)
        acc3 = acc3 * alpha[:, None] + tl.dot(p16, v3)

        m_i = m_new

    inv = 1.0 / tl.maximum(l_i, 1e-30)
    o_rows = (q_start + qpos_c).to(tl.int64) * stride_ot + head.to(tl.int64) * stride_oh
    tl.store(o_ptr + o_rows[:, None] + offs_32[None, :], (acc0 * inv[:, None]).to(o_ptr.dtype.element_ty), mask=q_valid[:, None])
    tl.store(o_ptr + o_rows[:, None] + 32 + offs_32[None, :], (acc1 * inv[:, None]).to(o_ptr.dtype.element_ty), mask=q_valid[:, None])
    tl.store(o_ptr + o_rows[:, None] + 64 + offs_32[None, :], (acc2 * inv[:, None]).to(o_ptr.dtype.element_ty), mask=q_valid[:, None])
    tl.store(o_ptr + o_rows[:, None] + 96 + offs_32[None, :], (acc3 * inv[:, None]).to(o_ptr.dtype.element_ty), mask=q_valid[:, None])


def gqa_prefill_attention_i8(q, kv_cache: torch.Tensor, out, cu_seqlens_q, seq_lens, block_table,
                             softmax_scale, block_q=32, tile_n=128, num_warps=16, num_stages=2,
                             num_prefill_tokens=None):
    T, H, D = q.shape
    _, two, block_size, Hkv, rs = kv_cache.shape
    assert two == 2 and rs == ROW_SLOTS and D == 128
    G = H // Hkv
    num_seqs = seq_lens.shape[0]
    BLOCK_M = block_q * G
    if num_prefill_tokens is None:
        num_prefill_tokens = T
    kv_u8 = kv_cache.view(torch.uint8)
    grid = (num_prefill_tokens // block_q + num_seqs, Hkv)
    _gqa_prefill_k8v5_kernel[grid](
        q, kv_u8, out, cu_seqlens_q, seq_lens, block_table,
        softmax_scale * 1.4426950408889634, num_seqs,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        kv_u8.stride(0),
        block_table.stride(0),
        v_block_off=block_size * Hkv * K_ROW_BYTES,
        NUM_KV_HEADS=Hkv, G=G, HEAD_DIM=D, BLOCK_SIZE=block_size,
        BLOCK_Q=block_q, BLOCK_M=BLOCK_M, TILE_N=tile_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out

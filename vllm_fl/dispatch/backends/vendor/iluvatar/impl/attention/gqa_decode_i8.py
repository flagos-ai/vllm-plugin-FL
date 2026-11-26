# Copyright (c) 2026 BAAI. All rights reserved.
"""GQA-packed split-KV paged decode attention over the K8V5 cache (Triton fallback)."""

import torch
import triton
import triton.language as tl

from .kv_i8 import K_ROW_BYTES, ROW_SLOTS, V_ROW_BYTES


@triton.jit
def _gqa_decode_k8v5_split_kernel(
    q_ptr, kv_u8_ptr, out_ptr,
    seq_lens_ptr, block_table_ptr,
    ws_acc_ptr, ws_m_ptr, ws_l_ptr,
    scale_log2e,
    stride_qs, stride_qh, stride_os, stride_oh,
    stride_kb,          # cache block stride in BYTES
    stride_bt,
    v_block_off: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    MIN_TILES: tl.constexpr,
    G: tl.constexpr,
    ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    offs_r = tl.arange(0, ROWS)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_r < G
    hq = kv_head * G + offs_r

    seq_len = tl.load(seq_lens_ptr + seq)
    if seq_len <= 0:
        if split == 0:
            z = tl.zeros([ROWS, HEAD_DIM], dtype=out_ptr.dtype.element_ty)
            tl.store(out_ptr + seq.to(tl.int64) * stride_os + hq[:, None] * stride_oh + offs_d[None, :],
                     z, mask=row_ok[:, None])
        return

    n_tiles = (seq_len + TILE_N - 1) // TILE_N
    n_splits = tl.minimum(MAX_SPLITS, tl.maximum(1, n_tiles // MIN_TILES))
    if split >= n_splits:
        return
    tps = (n_tiles + n_splits - 1) // n_splits
    t_begin = split * tps
    t_end = tl.minimum(t_begin + tps, n_tiles)
    if t_begin >= t_end:
        ws_row = (seq.to(tl.int64) * NUM_HEADS + hq) * MAX_SPLITS + split
        tl.store(ws_acc_ptr + ws_row[:, None] * HEAD_DIM + offs_d[None, :],
                 tl.zeros([ROWS, HEAD_DIM], dtype=tl.float32), mask=row_ok[:, None])
        tl.store(ws_m_ptr + ws_row, tl.full([ROWS], float("-inf"), dtype=tl.float32), mask=row_ok)
        tl.store(ws_l_ptr + ws_row, tl.zeros([ROWS], dtype=tl.float32), mask=row_ok)
        return

    offs_n = tl.arange(0, TILE_N)
    head = kv_head * G + tl.minimum(offs_r, G - 1)
    q = tl.load(q_ptr + seq.to(tl.int64) * stride_qs + head[:, None] * stride_qh + offs_d[None, :],
                mask=row_ok[:, None], other=0.0)

    m_i = tl.full([ROWS], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([ROWS], dtype=tl.float32)
    acc = tl.zeros([ROWS, HEAD_DIM], dtype=tl.float32)
    bt_base = block_table_ptr + seq.to(tl.int64) * stride_bt

    v_lo_idx = offs_d // 2
    v_lo_sh = (offs_d & 1) * 4
    v_hi_idx = offs_d // 8
    v_hi_sh = offs_d & 7
    v_sc_idx = offs_d // 32

    key = t_begin * TILE_N + offs_n
    blk = tl.load(bt_base + key // BLOCK_SIZE, mask=key < seq_len, other=0).to(tl.int64)
    for _ in range(t_begin, t_end):
        kmask = key < seq_len
        slot = key % BLOCK_SIZE
        idx = slot.to(tl.int64) * NUM_KV_HEADS + kv_head.to(tl.int64)
        blk_off = blk * stride_kb
        k_off = blk_off + idx * 144
        v_off = blk_off + v_block_off + idx * 88
        key = key + TILE_N
        blk = tl.load(bt_base + key // BLOCK_SIZE, mask=key < seq_len, other=0).to(tl.int64)

        sc_ptr = (kv_u8_ptr + k_off[:, None] + 128).to(tl.pointer_type(tl.bfloat16)) + (offs_d // 16)[None, :]
        k8 = tl.load(kv_u8_ptr + k_off[:, None] + offs_d[None, :], mask=kmask[:, None], other=0)
        ks = tl.load(sc_ptr, mask=kmask[:, None], other=0.0)
        kb = (k8.to(tl.int8, bitcast=True).to(tl.float32) * ks.to(tl.float32)).to(q.dtype)
        s = tl.dot(q, tl.trans(kb)) * scale_log2e
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v_lo = tl.load(kv_u8_ptr + v_off[:, None] + v_lo_idx[None, :], mask=kmask[:, None], other=0).to(tl.int32)
        v_hi = tl.load(kv_u8_ptr + v_off[:, None] + 64 + v_hi_idx[None, :], mask=kmask[:, None], other=0).to(tl.int32)
        vs = tl.load(
            (kv_u8_ptr + v_off[:, None] + 80).to(tl.pointer_type(tl.bfloat16)) + v_sc_idx[None, :],
            mask=kmask[:, None],
            other=0.0,
        ).to(tl.float32)
        u5 = ((v_lo >> v_lo_sh[None, :]) & 0xF) | (((v_hi >> v_hi_sh[None, :]) & 1) << 4)
        vb = ((u5 - 16).to(tl.float32) * vs).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), vb)
        m_i = m_new

    if n_splits == 1:
        o = acc / tl.maximum(l_i, 1e-30)[:, None]
        tl.store(out_ptr + seq.to(tl.int64) * stride_os + hq[:, None] * stride_oh + offs_d[None, :],
                 o.to(out_ptr.dtype.element_ty), mask=row_ok[:, None])
    else:
        ws_row = (seq.to(tl.int64) * NUM_HEADS + hq) * MAX_SPLITS + split
        tl.store(ws_acc_ptr + ws_row[:, None] * HEAD_DIM + offs_d[None, :], acc, mask=row_ok[:, None])
        tl.store(ws_m_ptr + ws_row, m_i, mask=row_ok)
        tl.store(ws_l_ptr + ws_row, l_i, mask=row_ok)


def gqa_decode_attention_i8(q, kv_cache: torch.Tensor, out, seq_lens, block_table, softmax_scale,
                            combine_kernel, num_splits=8, tile_n=32, num_warps=2, min_tiles=8, workspace=None):
    S, H, D = q.shape
    _, two, block_size, Hkv, rs = kv_cache.shape
    assert two == 2 and rs == ROW_SLOTS and D == 128
    G = H // Hkv
    kv_u8 = kv_cache.view(torch.uint8)
    ws_acc, ws_m, ws_l = workspace.get(S, H, num_splits, D, q.device)
    _gqa_decode_k8v5_split_kernel[(S, Hkv, num_splits)](
        q, kv_u8, out, seq_lens, block_table, ws_acc, ws_m, ws_l,
        softmax_scale * 1.4426950408889634,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        kv_u8.stride(0),
        block_table.stride(0),
        v_block_off=block_size * Hkv * K_ROW_BYTES,
        NUM_HEADS=H, NUM_KV_HEADS=Hkv, MAX_SPLITS=num_splits, MIN_TILES=min_tiles,
        G=G, ROWS=16, HEAD_DIM=D, BLOCK_SIZE=block_size, TILE_N=tile_n,
        num_warps=num_warps, num_stages=1,
    )
    if num_splits > 1:
        combine_kernel[(S, H)](
            ws_acc, ws_m, ws_l, out, seq_lens, out.stride(0), out.stride(1),
            NUM_HEADS=H, MAX_SPLITS=num_splits, SPLITS_POW2=triton.next_power_of_2(num_splits),
            MIN_TILES=min_tiles, TILE_N=tile_n, HEAD_DIM=D, num_warps=1,
        )
    return out

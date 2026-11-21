# Copyright (c) 2026 BAAI. All rights reserved.
"""GQA-packed split-KV paged decode attention kernel (Triton)."""

import torch
import triton
import triton.language as tl

_SM_COUNT = {}
_SMEM = {}


def _dev_index(device):
    return device.index if device.index is not None else torch.cuda.current_device()


def _num_sms(device):
    idx = _dev_index(device)
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def default_decode_config(device, head_dim=128):
    return 32, 1


def pick_num_splits(num_seqs: int, num_kv_heads: int, device, max_splits: int = 8) -> int:
    # SM-wave balanced Split-KV: when B * Hkv >= 96 (e.g. B=64, Hkv=2 -> 128 pairs),
    # 4 splits = 512 warps saturates all 16 SMs (32 warps/SM) in 1 uniform wave without
    # tail-wave fragmentation or extra combine workspace traffic.
    if num_seqs * num_kv_heads >= 96:
        return min(4, int(max_splits))
    return int(max_splits)


@triton.jit
def _gqa_decode_split_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_lens_ptr, block_table_ptr,
    ws_acc_ptr, ws_m_ptr, ws_l_ptr,
    scale_log2e,
    stride_qs, stride_qh, stride_os, stride_oh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    NUM_HEADS: tl.constexpr,
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
    kbase = k_ptr + kv_head * stride_kh
    vbase = v_ptr + kv_head * stride_vh

    key = t_begin * TILE_N + offs_n
    blk = tl.load(bt_base + key // BLOCK_SIZE, mask=key < seq_len, other=0).to(tl.int64)
    for t in range(t_begin, t_end):
        kmask = key < seq_len
        slot = key % BLOCK_SIZE
        k_off = blk * stride_kb + slot * stride_ks
        v_off = blk * stride_vb + slot * stride_vs
        key = key + TILE_N
        blk = tl.load(bt_base + key // BLOCK_SIZE, mask=key < seq_len, other=0).to(tl.int64)
        k = tl.load(kbase + k_off[:, None] + offs_d[None, :], mask=kmask[:, None], other=0.0)
        v = tl.load(vbase + v_off[:, None] + offs_d[None, :], mask=kmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2e
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
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


@triton.jit
def _gqa_decode_combine_kernel(
    ws_acc_ptr, ws_m_ptr, ws_l_ptr, out_ptr, seq_lens_ptr,
    stride_os, stride_oh,
    NUM_HEADS: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    SPLITS_POW2: tl.constexpr,
    MIN_TILES: tl.constexpr,
    TILE_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + seq)
    if seq_len <= 0:
        return  # zeroed directly by split == 0
    n_tiles = (seq_len + TILE_N - 1) // TILE_N
    n_splits = tl.minimum(MAX_SPLITS, tl.maximum(1, n_tiles // MIN_TILES))
    if n_splits == 1:
        return  # written directly by the split kernel
    tps = (n_tiles + n_splits - 1) // n_splits
    active_splits = (n_tiles + tps - 1) // tps
    offs_s = tl.arange(0, SPLITS_POW2)
    offs_d = tl.arange(0, HEAD_DIM)
    smask = offs_s < active_splits
    base = (seq.to(tl.int64) * NUM_HEADS + head) * MAX_SPLITS
    m = tl.load(ws_m_ptr + base + offs_s, mask=smask, other=float("-inf"))
    l = tl.load(ws_l_ptr + base + offs_s, mask=smask, other=0.0)
    m_all = tl.max(m, 0)
    m_safe = tl.where(m_all == float("-inf"), 0.0, m_all)
    w = tl.where(smask, tl.math.exp2(m - m_safe), 0.0)
    l_all = tl.sum(l * w, 0)
    acc_mask = smask & (w > 0.0)
    acc = tl.load(ws_acc_ptr + (base + offs_s)[:, None] * HEAD_DIM + offs_d[None, :],
                  mask=acc_mask[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], 0) / tl.maximum(l_all, 1e-30)
    tl.store(out_ptr + seq.to(tl.int64) * stride_os + head * stride_oh + offs_d,
             o.to(out_ptr.dtype.element_ty))


class _Workspace:
    def __init__(self):
        self.acc = None
        self.m = None
        self.l = None

    def get(self, num_seqs, num_heads, num_splits, head_dim, device):
        need = num_seqs * num_heads * num_splits
        if (self.acc is None or self.acc.shape[0] < need or self.acc.device != device
                or self.acc.shape[1] != head_dim):
            cap = max(need, 256 * 16 * 16)
            self.acc = torch.empty((cap, head_dim), dtype=torch.float32, device=device)
            self.m = torch.empty((cap,), dtype=torch.float32, device=device)
            self.l = torch.empty((cap,), dtype=torch.float32, device=device)
        return self.acc, self.m, self.l


_WS = _Workspace()


def gqa_decode_attention(q, k_cache, v_cache, out, seq_lens, block_table, softmax_scale,
                         num_splits=None, tile_n=None, num_warps=2, num_stages=None,
                         min_tiles=8, workspace=None):
    S, H, D = q.shape
    _, block_size, Hkv, _ = k_cache.shape
    G = H // Hkv
    assert G <= 16 and H % Hkv == 0
    if tile_n is None or num_stages is None:
        d_tile, d_stages = default_decode_config(q.device, D)
        tile_n = tile_n or d_tile
        num_stages = num_stages or d_stages
    if num_splits is None:
        num_splits = pick_num_splits(S, Hkv, q.device)
    ws = workspace if workspace is not None else _WS
    ws_acc, ws_m, ws_l = ws.get(S, H, num_splits, D, q.device)
    _gqa_decode_split_kernel[(S, Hkv, num_splits)](
        q, k_cache, v_cache, out, seq_lens, block_table, ws_acc, ws_m, ws_l,
        softmax_scale * 1.4426950408889634,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0),
        NUM_HEADS=H, MAX_SPLITS=num_splits, MIN_TILES=min_tiles, G=G, ROWS=16, HEAD_DIM=D,
        BLOCK_SIZE=block_size, TILE_N=tile_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    if num_splits > 1:
        _gqa_decode_combine_kernel[(S, H)](
            ws_acc, ws_m, ws_l, out, seq_lens, out.stride(0), out.stride(1),
            NUM_HEADS=H, MAX_SPLITS=num_splits, SPLITS_POW2=triton.next_power_of_2(num_splits),
            MIN_TILES=min_tiles, TILE_N=tile_n, HEAD_DIM=D, num_warps=1,
        )
    return out

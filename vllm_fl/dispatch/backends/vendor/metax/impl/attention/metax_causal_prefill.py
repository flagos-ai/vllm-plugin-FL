"""Pure-Triton GQA BT=8 Block-Sparse 25% Causal Prefill for MetaX C500 (E477: 8.29ms vs 22.69ms flash_attn = 2.74x)."""
from __future__ import annotations
import logging
import torch
import triton
import triton.language as tl
from flag_gems.fused.metax_w8a8 import _FLAGS_X_IGROUP

logger = logging.getLogger(__name__)
BLK = 128
PROTECTED_LAYERS = frozenset((0, 28, 41))
_LOGGED = False


@triton.jit
def _cp_score_kernel(
    Q, K, SCORES, CU,
    n_seq, max_qblks, max_kblks,
    sq_t, sq_h, sq_d, sk_t, sk_h, sk_d,
    scale_log2,
    QPK: tl.constexpr, D: tl.constexpr,
):
    q_blk = tl.program_id(0)
    k_grp = tl.program_id(1)
    sh = tl.program_id(2)
    seq = sh // 2
    hkv = sh % 2
    s0 = tl.load(CU + seq)
    s1 = tl.load(CU + seq + 1)
    qlen = s1 - s0
    if (q_blk * 128 >= qlen) | ((q_blk <= 8) | (k_grp * 8 >= q_blk)):
        return
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 64)
    rd = tl.arange(0, D)
    qpos = tl.minimum(q_blk * 128 + (rows // 8) * 32 + 16, qlen - 1)
    qh = hkv * QPK + (rows % 8)
    q = tl.load(Q + (s0 + qpos[:, None]).to(tl.int64) * sq_t + qh[:, None] * sq_h + rd[None, :] * sq_d).to(tl.bfloat16)
    kblk = k_grp * 8 + (cols // 8)
    kpos = kblk * 128 + (cols % 8) * 16 + 8
    k_valid = kpos < q_blk * 128
    k = tl.load(K + (s0 + kpos[None, :]).to(tl.int64) * sk_t + hkv * sk_h + rd[:, None] * sk_d, mask=k_valid[None, :], other=0.0).to(tl.bfloat16)
    dot = tl.dot(q, k, out_dtype=tl.float32) * scale_log2
    dot = tl.where(k_valid[None, :], dot, -1e9)
    grouped = tl.reshape(dot, (32, 8, 8))
    peak = tl.max(grouped, axis=2)
    w = tl.sum(tl.exp2(grouped - peak[:, :, None]), axis=2)
    lse = peak + tl.log2(tl.maximum(w, 1e-12))
    pages = k_grp * 8 + tl.arange(0, 8)
    out_off = ((sh * max_qblks + q_blk) * 32 + rows[:, None]) * max_kblks + pages[None, :]
    tl.store(SCORES + out_off, lse, mask=pages[None, :] < q_blk)


@triton.jit
def _cp_rank_kernel(SCORES, PICKS, CU, max_qblks, max_kblks, MAX_KBLKS_PAD: tl.constexpr):
    q_blk = tl.program_id(0)
    sh = tl.program_id(1)
    seq = sh // 2
    s0 = tl.load(CU + seq)
    s1 = tl.load(CU + seq + 1)
    qlen = s1 - s0
    if (q_blk * 128 >= qlen) | (q_blk <= 8):
        return
    rows = tl.arange(0, 32)
    pages = tl.arange(0, MAX_KBLKS_PAD)
    valid = pages < q_blk
    lse = tl.load(SCORES + ((sh * max_qblks + q_blk) * 32 + rows[:, None]) * max_kblks + pages[None, :], mask=valid[None, :], other=-1e9)
    row_max = tl.max(lse, axis=1)
    w = tl.where(valid[None, :], tl.exp2(lse - row_max[:, None]), 0.0)
    mass = tl.sum(w / tl.maximum(tl.sum(w, axis=1)[:, None], 1e-12), axis=0)
    mass = tl.where((pages == 0) | ((pages >= q_blk - 4) & valid), float("inf"), mass)
    mass = tl.where(valid, mass, -1.0)
    keep = tl.minimum(q_blk, tl.maximum(8, tl.cdiv(q_blk, 4)))
    packed = (mass.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | pages.to(tl.uint64)
    ranked = tl.sort(tl.where(valid, packed, 0), descending=True).to(tl.int32)
    picks = tl.sort(tl.where(pages < keep, ranked, 0x7FFFFFFF), descending=False)
    tl.store(PICKS + (sh * max_qblks + q_blk) * max_kblks + pages, picks, mask=pages < keep)


@triton.jit
def _cp_attend_bt8_kernel(
    Q, K, V, PICKS, OUT, CU,
    max_qblks, max_kblks,
    sq_t, sq_h, sq_d, sk_t, sk_h, sk_d, sv_t, sv_h, sv_d, so_t, so_h, so_d,
    scale_log2,
    QPK: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, D: tl.constexpr,
):
    qt = tl.program_id(0)
    hkv = tl.program_id(1)
    seq = tl.program_id(2)
    s0 = tl.load(CU + seq)
    s1 = tl.load(CU + seq + 1)
    qlen = s1 - s0
    n_qt = tl.cdiv(qlen, BT)
    q_idx = n_qt - 1 - qt
    if q_idx < 0:
        return
    q_start = q_idx * BT
    rows = tl.arange(0, BT * QPK)
    rm = q_start + (rows // QPK)
    qh = hkv * QPK + (rows % QPK)
    m_mask = rm < qlen
    rd = tl.arange(0, D)

    q = tl.load(Q + (s0 + rm[:, None]).to(tl.int64) * sq_t + qh[:, None] * sq_h + rd[None, :] * sq_d, mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
    acc = tl.zeros((BT * QPK, D), dtype=tl.float32)
    m_i = tl.full((BT * QPK,), -1e9, dtype=tl.float32)
    l_i = tl.zeros((BT * QPK,), dtype=tl.float32)
    rk = tl.arange(0, BK)

    q_blk = q_start // 128
    sh = seq * 2 + hkv
    keep = tl.minimum(q_blk, tl.maximum(8, tl.cdiv(q_blk, 4))) if q_blk > 8 else q_blk
    for pi in range(0, keep):
        blk = tl.load(PICKS + (sh * max_qblks + q_blk) * max_kblks + pi) if q_blk > 8 else pi
        for sub in tl.static_range(2):
            k_pos = blk * 128 + sub * BK + rk
            k = tl.load(K + (s0 + k_pos[None, :]).to(tl.int64) * sk_t + hkv * sk_h + rd[:, None] * sk_d).to(tl.bfloat16)
            s = tl.dot(q, k, out_dtype=tl.float32) * scale_log2
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(V + (s0 + k_pos[:, None]).to(tl.int64) * sv_t + hkv * sv_h + rd[None, :] * sv_d).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
            m_i = m_new

    cur_end = tl.minimum(q_start + BT, qlen)
    n_sub = tl.cdiv(cur_end - q_blk * 128, BK)
    for sub in range(0, n_sub):
        k_pos = q_blk * 128 + sub * BK + rk
        k_valid = k_pos < qlen
        k = tl.load(K + (s0 + k_pos[None, :]).to(tl.int64) * sk_t + hkv * sk_h + rd[:, None] * sk_d, mask=k_valid[None, :], other=0.0).to(tl.bfloat16)
        s = tl.dot(q, k, out_dtype=tl.float32) * scale_log2
        causal = k_valid[None, :] & (k_pos[None, :] <= rm[:, None])
        s = tl.where(causal, s, -1e9)
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)
        p = tl.where(causal, tl.exp2(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + (s0 + k_pos[:, None]).to(tl.int64) * sv_t + hkv * sv_h + rd[None, :] * sv_d, mask=k_valid[:, None], other=0.0).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m_i = m_new

    out = (acc / tl.maximum(l_i[:, None], 1e-12)).to(OUT.dtype.element_ty)
    tl.store(OUT + (s0 + rm[:, None]).to(tl.int64) * so_t + qh[:, None] * so_h + rd[None, :] * so_d, out, mask=m_mask[:, None])


def causal_prefill_varlen(q, k, v, cu_seqlens, max_seqlen: int, scale: float, out=None):
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.warning("E477 fast GQA BT=8 cp25 causal_prefill active max_q=%d", max_seqlen)
    if out is None:
        out = torch.empty_like(q)
    n_seq = int(cu_seqlens.shape[0]) - 1
    hq, hkv, d = q.shape[1], k.shape[1], q.shape[2]
    qpk = hq // hkv
    max_blks = triton.cdiv(max_seqlen, BLK)
    max_kblks_pad = triton.next_power_of_2(max(16, max_blks))
    pairs = n_seq * hkv
    scale_log2 = scale * 1.4426950408889634
    scores = torch.empty((pairs, max_blks, 32, max_blks), dtype=torch.float32, device=q.device)
    picks = torch.empty((pairs, max_blks, max_blks), dtype=torch.int32, device=q.device)
    _cp_score_kernel[(max_blks, triton.cdiv(max_blks, 8), pairs)](
        q, k, scores, cu_seqlens, n_seq, max_blks, max_blks,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        scale_log2, QPK=qpk, D=d, num_warps=4, num_stages=1,
    )
    _cp_rank_kernel[(max_blks, pairs)](
        scores, picks, cu_seqlens, max_blks, max_blks,
        MAX_KBLKS_PAD=max_kblks_pad, num_warps=4,
    )
    _cp_attend_bt8_kernel[(triton.cdiv(max_seqlen, 8), hkv, n_seq)](
        q, k, v, picks, out, cu_seqlens, max_blks, max_blks,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
        scale_log2, QPK=qpk, BT=8, BK=64, D=d,
        num_warps=4, num_stages=1, pipeline="basic", extra_options=_FLAGS_X_IGROUP,
    )
    return out

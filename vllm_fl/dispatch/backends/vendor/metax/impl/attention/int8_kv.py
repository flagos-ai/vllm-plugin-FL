# Copyright (c) 2026 BAAI. All rights reserved.

"""
Self-developed int8 KV cache attention for the MetaX backend (E034).

vLLM's ``int8_per_token_head`` cache keeps K and V as int8 codes with one fp32 scale per (token, head) inside the cache
pages ([blocks, 2, 16, heads, 128 + 4]). The page is re-read here head-contiguous (``hc_views``): [K codes: head, token,
128 B][V codes][K scales: head, token][V scales]. All consumers (cache write, decode kernel, prefill dequantisation) take
the views' strides.

* decode rows (query length 1): split-KV Triton kernel with the score matrix transposed (tokens on the MMA M axis,
  S^T = K Q^T, O^T += V^T P^T), so the softmax reduces inside a lane group and no cross-lane chains are needed; int8 codes
  are unpacked from packed int32 words with shl / ashr / cvt. Per (KV head, sequence, split) one single-warp program holds
  the 8 query heads of the KV head. Then a reduce over the splits. (E034 attempts 7-11: 0.98x / 0.875x of the vendor bf16
  flash_attn_with_kvcache at 64 x 4608 / 64 x 17000.)
* rows with query length > 1: the K / V of those rows are dequantised once per step into a bf16 workspace
  [tokens, heads, 128] and attended with the vendor flash_attn_varlen_func. Rows with query length 1 get key length 0 in
  that call (their output is overwritten by the decode kernel afterwards).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_LOG2E = 1.4426950408889634
_QPK = 8
_HEAD = 128
_PAGE = 16
_NP = 2    # pages (16 tokens) per loop iteration
_REF_ROUND_BF16 = False

# mxcc flags of the decode kernel (E034 attempt 8-10: instruction scheduling by register pressure, no load sinking, igroup).
_DEC_FLAGS = (" -mllvm -metaxgpu-sched-regpressure=true -mllvm -metaxgpu-sinkload=false -mllvm -metaxgpu-vectorize-slp=true"
              " -mllvm -metaxgpu-igroup -mllvm -metaxgpu-aggressive-4g-addr-opt=true -mllvm -metaxgpu-shl-add-combine=false"
              " -mllvm -misched-postra=true -mllvm -enable-post-misched=true -mllvm -metaxgpu-mma-unroll-count=1 ")


def hc_views(kv_cache):
    """Head-contiguous views of an int8 per-token-head KV cache [blocks, 2, 16, heads, 128 + 4]: key / value codes
    [blocks, 16, heads, 128] (int8) and fp32 scales [blocks, 16, heads]."""
    nb, two, bs, nkv, ph = kv_cache.shape
    hs = ph - 4
    page = kv_cache.stride(0)
    off = kv_cache.storage_offset()
    codes = nkv * bs * hs
    kc = torch.as_strided(kv_cache, (nb, bs, nkv, hs), (page, hs, bs * hs, 1), off)
    vc = torch.as_strided(kv_cache, (nb, bs, nkv, hs), (page, hs, bs * hs, 1), off + codes)
    base = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(kv_cache.untyped_storage())
    ksc = torch.as_strided(base, (nb, bs, nkv), (page // 4, 1, bs), (off + 2 * codes) // 4)
    vsc = torch.as_strided(base, (nb, bs, nkv), (page // 4, 1, bs), (off + 2 * codes) // 4 + nkv * bs)
    return kc, vc, ksc, vsc


@dataclass
class CacheViews:
    kc: torch.Tensor     # int8 [blocks, 16, heads, 128]
    vc: torch.Tensor
    k32: torch.Tensor    # the same bytes as int32 [blocks, 16, heads, 32]
    v32: torch.Tensor
    ksc: torch.Tensor    # fp32 [blocks, 16, heads]
    vsc: torch.Tensor


def make_views(kv_cache, init_scales: bool) -> CacheViews:
    kc, vc, ksc, vsc = hc_views(kv_cache)
    if init_scales:                      # not needed by the kernels here (every read of an unwritten token is masked)
        ksc.fill_(1.0)
        vsc.fill_(1.0)
    return CacheViews(kc, vc, kc.view(torch.int32), vc.view(torch.int32), ksc, vsc)


# --------------------------------------------------------------------------- kernels


@triton.jit
def _unp(w, i: tl.constexpr):
    """Byte i of an int32 word holding four int8 codes, sign-extended, as fp16."""
    return ((w << (24 - 8 * i)) >> 24).to(tl.float16)


@triton.jit
def _lo16(y):
    return (y & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)


@triton.jit
def _hi16(y):
    return (y >> 16).to(tl.int16).to(tl.float16, bitcast=True)


@triton.jit
def _pair8(w, i: tl.constexpr):
    """bytes i and i + 2 of each word as fp16 (code + 1152) in one shift / and / xor / or: 0x6400 = 1024, xor 0x80 = code + 128"""
    return (((w >> (8 * i)) & 0x00FF00FF) ^ 0x00800080) | 0x64006400


@triton.jit(do_not_specialize=["S"])
def _decode_kernel_i8(Q, K, V, KS, VS, TABLE, SEQ_LENS, CU_Q, PO, PM, PL, qk_scale,
                      s_q0, s_q1, s_kb, s_kt, s_kh, s_sb, s_st, s_sh, s_tab, S, CNT, O, s_o0, s_o1,
                      SP_TABLE, SP_SEQ, ROW_SLOT, NEED_REFRESH, PAGE_SCORES, s_sp_slot,
                      HQ: tl.constexpr, D: tl.constexpr, QPK: tl.constexpr, NP: tl.constexpr, MAGIC: tl.constexpr = False,
                      FOLD: tl.constexpr = False, SP: tl.constexpr = 16,
                      SPARSE: tl.constexpr = False, WRITE_SCORES: tl.constexpr = False, MAX_PAGES_C: tl.constexpr = 2048):
    T: tl.constexpr = 16 * NP
    h = tl.program_id(0)
    seq = tl.program_id(1)
    sp = tl.program_id(2)
    qrow = tl.load(CU_Q + seq)
    if tl.load(CU_Q + seq + 1) - qrow != 1:
        return
    use_sp = False
    sp_slot = 0
    if SPARSE:
        sp_L = tl.load(SP_SEQ + seq)
        use_sp = sp_L > 0
        sp_slot = tl.load(ROW_SLOT + seq)
        L = tl.where(use_sp, sp_L, tl.load(SEQ_LENS + seq))
    else:
        L = tl.load(SEQ_LENS + seq)
    do_write_sc = False
    if WRITE_SCORES:
        do_write_sc = (tl.load(NEED_REFRESH + seq) != 0) & (~use_sp)
    npg = tl.cdiv(L, 16)
    per = tl.cdiv(npg, S)
    lo = sp * per
    hi = tl.maximum(tl.minimum(lo + per, npg), lo)
    rows = tl.arange(0, 16)
    rmask = rows < QPK
    c = tl.arange(0, 32)
    t = tl.arange(0, T)
    pg = t // 16
    tt = t % 16
    qb = Q + qrow.to(tl.int64) * s_q0 + (h * QPK + rows)[None, :] * s_q1 + (4 * c)[:, None]
    q0 = tl.load(qb, mask=rmask[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(qb + 1, mask=rmask[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(qb + 2, mask=rmask[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(qb + 3, mask=rmask[None, :], other=0.0).to(tl.float16)
    if MAGIC:
        qsum = tl.sum(q0.to(tl.float32) + q1.to(tl.float32) + q2.to(tl.float32) + q3.to(tl.float32), 0)
    cv = tl.zeros([16], tl.float32)
    m_i = tl.full([16], float("-inf"), tl.float32)
    l_i = tl.zeros([16], tl.float32)
    a0 = tl.zeros([32, 16], tl.float32)
    a1 = tl.zeros([32, 16], tl.float32)
    a2 = tl.zeros([32, 16], tl.float32)
    a3 = tl.zeros([32, 16], tl.float32)
    tab_dense = TABLE + seq.to(tl.int64) * s_tab
    if SPARSE:
        tab_sp = SP_TABLE + sp_slot.to(tl.int64) * s_sp_slot
    off = tt[:, None] * s_kt + h * s_kh + c[None, :]
    for j in range(lo, hi, NP):
        pj = j + pg
        msk = (pj < hi) & (pj * 16 + tt < L)
        if SPARSE:
            blk_s = tl.load(tab_sp + pj, mask=use_sp & (pj < hi), other=0)
            blk_d = tl.load(tab_dense + pj, mask=(~use_sp) & (pj < hi), other=0)
            blk = tl.where(use_sp, blk_s, blk_d).to(tl.int64)
        else:
            blk = tl.load(tab_dense + pj, mask=pj < hi, other=0).to(tl.int64)
        kw = tl.load(K + blk[:, None] * s_kb + off, mask=msk[:, None], other=0)
        vw = tl.trans(tl.load(V + blk[:, None] * s_kb + off, mask=msk[:, None], other=0))
        sk = tl.load(KS + blk * s_sb + tt * s_st + h * s_sh, mask=msk, other=0.0)
        sv = tl.load(VS + blk * s_sb + tt * s_st + h * s_sh, mask=msk, other=0.0)
        if MAGIC:
            y0 = _pair8(kw, 0)
            y1 = _pair8(kw, 1)
            sc = tl.dot(_lo16(y0), q0)
            sc = sc + tl.dot(_lo16(y1), q1)
            sc = sc + tl.dot(_hi16(y0), q2)
            sc = sc + tl.dot(_hi16(y1), q3)
            sc = (sc - 1152.0 * qsum[None, :]) * sk[:, None]
        else:
            sc = tl.dot(_unp(kw, 0), q0)
            sc = sc + tl.dot(_unp(kw, 1), q1)
            sc = sc + tl.dot(_unp(kw, 2), q2)
            sc = sc + tl.dot(_unp(kw, 3), q3)
            sc = sc * sk[:, None]
        sc = tl.where(msk[:, None], sc * qk_scale, float("-inf"))
        if WRITE_SCORES:
            if do_write_sc:
                sc_v = tl.where(msk[:, None] & rmask[None, :], sc, float("-inf"))
                sc_tok = tl.max(sc_v, 1)
                pg_max = tl.max(tl.reshape(sc_tok, (NP, 16)), 1)
                p_offs = j + tl.arange(0, NP)
                tl.store(
                    PAGE_SCORES + (seq.to(tl.int64) * 2 + h) * MAX_PAGES_C + p_offs,
                    pg_max,
                    mask=(p_offs < hi) & (p_offs < MAX_PAGES_C),
                )
        m_new = tl.maximum(m_i, tl.max(sc, 0))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(sc - m_new[None, :])
        l_i = l_i * alpha + tl.sum(p, 0)
        ps = (p * (sv * 64.0)[:, None]).to(tl.float16)       # x64: the fp16 P . V product keeps its low bits; undone at the store
        if MAGIC:
            # the dots see code + 1152: remove 1152 * sum(ps) (the same fp16-rounded ps) per output dim at the store
            cv = cv * alpha + tl.sum(ps.to(tl.float32), 0)
            z0 = _pair8(vw, 0)
            z1 = _pair8(vw, 1)
            a0 = a0 * alpha[None, :] + tl.dot(_lo16(z0), ps)
            a1 = a1 * alpha[None, :] + tl.dot(_lo16(z1), ps)
            a2 = a2 * alpha[None, :] + tl.dot(_hi16(z0), ps)
            a3 = a3 * alpha[None, :] + tl.dot(_hi16(z1), ps)
        else:
            a0 = a0 * alpha[None, :] + tl.dot(_unp(vw, 0), ps)
            a1 = a1 * alpha[None, :] + tl.dot(_unp(vw, 1), ps)
            a2 = a2 * alpha[None, :] + tl.dot(_unp(vw, 2), ps)
            a3 = a3 * alpha[None, :] + tl.dot(_unp(vw, 3), ps)
        m_i = m_new
    pbase = (seq * S + sp) * HQ + h * QPK
    po = PO + (pbase + rows)[None, :] * D + (4 * c)[:, None]
    r = 1.0 / 64.0
    cvb = (cv * (1152.0 / 64.0))[None, :]
    tl.store(po, a0 * r - cvb, mask=rmask[None, :])
    tl.store(po + 1, a1 * r - cvb, mask=rmask[None, :])
    tl.store(po + 2, a2 * r - cvb, mask=rmask[None, :])
    tl.store(po + 3, a3 * r - cvb, mask=rmask[None, :])
    tl.store(PM + pbase + rows, m_i, mask=rmask)
    tl.store(PL + pbase + rows, l_i, mask=rmask)
    if FOLD:
        # the last split of this (row, KV head) to finish merges all splits (no separate reduce launch); CNT self-resets
        tick = tl.atomic_add(CNT + seq * (HQ // QPK) + h, 1, sem="acq_rel")
        if tick == S - 1:
            _fold_merge(PO, PM, PL, O, seq, qrow, h, S, s_o0, s_o1, 1, 0, HQ, D, QPK, SP, False)
            tl.atomic_xchg(CNT + seq * (HQ // QPK) + h, 0)


@triton.jit
def _fold_merge(PO, PM, PL, O, seq, qrow, h, S, s_o0, s_o1, NTOK: tl.constexpr, tok, HQ: tl.constexpr, D: tl.constexpr,
                QPK: tl.constexpr, SP: tl.constexpr, NORMED: tl.constexpr):
    """Merge the S split partials of query token `tok` of row `seq`, heads of KV head h (layout [seq, split, token, head]).
    NORMED: partials are a / l (verify / tail kernels, weight l 2^(m - mx)); else unnormalised a (decode kernel, weight 2^(m - mx))."""
    hr = tl.arange(0, 16)
    hm = hr < QPK
    sidx = tl.arange(0, SP)
    smask = sidx < S
    d = tl.arange(0, D)
    HS: tl.constexpr = NTOK * HQ
    base = seq * S * HS + tok * HQ + h * QPK
    ix = base + sidx[:, None] * HS + hr[None, :]
    m = tl.load(PM + ix, mask=smask[:, None] & hm[None, :], other=float("-inf"), volatile=True)
    lsum = tl.load(PL + ix, mask=smask[:, None] & hm[None, :], other=0.0, volatile=True)
    mx = tl.max(m, 0)
    mx = tl.where(mx > float("-inf"), mx, 0.0)
    w = tl.exp2(m - mx[None, :])
    denom = tl.sum(w * lsum, 0)
    denom = tl.where(denom > 0.0, denom, 1.0)
    acc = tl.zeros([16, D], tl.float32)
    for s in range(S):
        ws = tl.exp2(tl.load(PM + base + s * HS + hr, mask=hm, other=float("-inf"), volatile=True) - mx)
        if NORMED:
            ws = ws * tl.load(PL + base + s * HS + hr, mask=hm, other=0.0, volatile=True)
        acc += tl.load(PO + (base + s * HS + hr)[:, None] * D + d[None, :], mask=hm[:, None], other=0.0, volatile=True).to(tl.float32) * ws[:, None]
    out = acc / denom[:, None]
    tl.store(O + (qrow + tok).to(tl.int64) * s_o0 + (h * QPK + hr)[:, None] * s_o1 + d[None, :], out.to(O.dtype.element_ty),
             mask=hm[:, None])


@triton.jit(do_not_specialize=["S"])
def _verify_kernel_q2(Q, K, V, KS, VS, TABLE, SEQ_LENS, CU_Q, PO, PM, PL, qk_scale,
                      s_q0, s_q1, s_kb, s_kt, s_kh, s_sb, s_st, s_sh, s_tab, S, CNT, O, s_o0, s_o1,
                      HQ: tl.constexpr, D: tl.constexpr, QPK: tl.constexpr, NP: tl.constexpr,
                      FOLD: tl.constexpr = False, SP: tl.constexpr = 16):
    """Rows with query length 2 (speculative verify, gamma 1): both query tokens x 8 heads of a KV head in ONE 16-row group,
    so every K / V tile is decoded once for both tokens (E034 attempt 30); MAGIC decode as _decode_kernel_i8; token t of the
    row sees positions < L - 1 + t. fp16 normalised partials [seq, split, token, head, D]."""
    T: tl.constexpr = 16 * NP
    h = tl.program_id(0)
    seq = tl.program_id(1)
    sp = tl.program_id(2)
    qrow = tl.load(CU_Q + seq)
    if tl.load(CU_Q + seq + 1) - qrow != 2:
        return
    L = tl.load(SEQ_LENS + seq)
    npg = tl.cdiv(L, 16)
    per = tl.cdiv(npg, S)
    lo = sp * per
    hi = tl.maximum(tl.minimum(lo + per, npg), lo)
    rows = tl.arange(0, 16)
    hh = rows % QPK
    tk = rows // QPK
    lim = L - 1 + tk
    c = tl.arange(0, 32)
    t = tl.arange(0, T)
    pg = t // 16
    tt = t % 16
    qb = Q + (qrow + tk).to(tl.int64)[None, :] * s_q0 + (h * QPK + hh)[None, :] * s_q1 + (4 * c)[:, None]
    q0 = tl.load(qb).to(tl.float16)
    q1 = tl.load(qb + 1).to(tl.float16)
    q2 = tl.load(qb + 2).to(tl.float16)
    q3 = tl.load(qb + 3).to(tl.float16)
    qsum = tl.sum(q0.to(tl.float32) + q1.to(tl.float32) + q2.to(tl.float32) + q3.to(tl.float32), 0)
    cv = tl.zeros([16], tl.float32)
    m_i = tl.full([16], float("-inf"), tl.float32)
    l_i = tl.zeros([16], tl.float32)
    a0 = tl.zeros([32, 16], tl.float32)
    a1 = tl.zeros([32, 16], tl.float32)
    a2 = tl.zeros([32, 16], tl.float32)
    a3 = tl.zeros([32, 16], tl.float32)
    tab = TABLE + seq.to(tl.int64) * s_tab
    off = tt[:, None] * s_kt + h * s_kh + c[None, :]
    for j in range(lo, hi, NP):
        pj = j + pg
        pos = pj * 16 + tt
        msk = (pj < hi) & (pos < L)
        blk = tl.load(tab + pj, mask=pj < hi, other=0).to(tl.int64)
        kw = tl.load(K + blk[:, None] * s_kb + off, mask=msk[:, None], other=0)
        vw = tl.trans(tl.load(V + blk[:, None] * s_kb + off, mask=msk[:, None], other=0))
        sk = tl.load(KS + blk * s_sb + tt * s_st + h * s_sh, mask=msk, other=0.0)
        sv = tl.load(VS + blk * s_sb + tt * s_st + h * s_sh, mask=msk, other=0.0)
        y0 = _pair8(kw, 0)
        y1 = _pair8(kw, 1)
        sc = tl.dot(_lo16(y0), q0)
        sc = sc + tl.dot(_lo16(y1), q1)
        sc = sc + tl.dot(_hi16(y0), q2)
        sc = sc + tl.dot(_hi16(y1), q3)
        sc = (sc - 1152.0 * qsum[None, :]) * sk[:, None]
        vis = (pj < hi)[:, None] & (pos[:, None] < lim[None, :])
        sc = tl.where(vis, sc * qk_scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 0))
        m_use = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(sc - m_use[None, :])
        l_i = l_i * alpha + tl.sum(p, 0)
        ps = (p * (sv * 64.0)[:, None]).to(tl.float16)
        cv = cv * alpha + tl.sum(ps.to(tl.float32), 0)
        z0 = _pair8(vw, 0)
        z1 = _pair8(vw, 1)
        a0 = a0 * alpha[None, :] + tl.dot(_lo16(z0), ps)
        a1 = a1 * alpha[None, :] + tl.dot(_lo16(z1), ps)
        a2 = a2 * alpha[None, :] + tl.dot(_hi16(z0), ps)
        a3 = a3 * alpha[None, :] + tl.dot(_hi16(z1), ps)
        m_i = m_new
    pb = ((seq * S + sp) * 2 + tk) * HQ + h * QPK + hh
    scl = 1.0 / tl.where(l_i > 0.0, l_i, 1.0)
    cvb = (cv * (1152.0 / 64.0) * scl)[None, :]
    r = (scl * (1.0 / 64.0))[None, :]
    po = PO + pb[None, :] * D + (4 * c)[:, None]
    tl.store(po, (a0 * r - cvb).to(tl.float16))
    tl.store(po + 1, (a1 * r - cvb).to(tl.float16))
    tl.store(po + 2, (a2 * r - cvb).to(tl.float16))
    tl.store(po + 3, (a3 * r - cvb).to(tl.float16))
    tl.store(PM + pb, m_i)
    tl.store(PL + pb, l_i)
    if FOLD:
        tick = tl.atomic_add(CNT + seq * (HQ // QPK) + h, 1, sem="acq_rel")
        if tick == S - 1:
            _fold_merge(PO, PM, PL, O, seq, qrow, h, S, s_o0, s_o1, 2, 0, HQ, D, QPK, SP, True)
            _fold_merge(PO, PM, PL, O, seq, qrow, h, S, s_o0, s_o1, 2, 1, HQ, D, QPK, SP, True)
            tl.atomic_xchg(CNT + seq * (HQ // QPK) + h, 0)


@triton.jit(do_not_specialize=["S"])
def _reduce_q2(PO, PM, PL, O, CU_Q, s_o0, s_o1, S, SP: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr):
    seq = tl.program_id(0)
    tok = tl.program_id(1)
    qrow = tl.load(CU_Q + seq)
    if tl.load(CU_Q + seq + 1) - qrow != 2:
        return
    rows = tl.arange(0, HQ)
    sidx = tl.arange(0, SP)
    smask = sidx < S
    d = tl.arange(0, D)
    HS: tl.constexpr = 2 * HQ
    base = seq * S * HS + tok * HQ
    mm = tl.load(PM + base + sidx[:, None] * HS + rows[None, :], mask=smask[:, None], other=float("-inf"))
    lsum = tl.load(PL + base + sidx[:, None] * HS + rows[None, :], mask=smask[:, None], other=0.0)
    mx = tl.max(mm, 0)
    mx = tl.where(mx > float("-inf"), mx, 0.0)
    denom = tl.sum(tl.exp2(mm - mx[None, :]) * lsum, 0)
    denom = tl.where(denom > 0.0, denom, 1.0)
    acc = tl.zeros([HQ, D], tl.float32)
    for s in range(S):
        w = tl.exp2(tl.load(PM + base + s * HS + rows) - mx) * tl.load(PL + base + s * HS + rows)
        acc += tl.load(PO + (base + s * HS + rows)[:, None] * D + d[None, :]).to(tl.float32) * w[:, None]
    out = acc / denom[:, None]
    tl.store(O + (qrow + tok).to(tl.int64) * s_o0 + rows[:, None] * s_o1 + d[None, :], out.to(O.dtype.element_ty))


TAIL_QLEN = 16       # rows with VERIFY_QLEN < query length <= TAIL_QLEN go to the tail kernel (split-KV over the int8 cache)
_TAIL_G = TAIL_QLEN // 2


@triton.jit(do_not_specialize=["S"])
def _tail_kernel(Q, K, V, KS, VS, TABLE, SEQ_LENS, CU_Q, ROWS, PO, PM, PL, qk_scale,
                 s_q0, s_q1, s_kb, s_kt, s_kh, s_sb, s_st, s_sh, s_tab, S,
                 HQ: tl.constexpr, D: tl.constexpr, QPK: tl.constexpr, NP: tl.constexpr, G: tl.constexpr):
    """Rows with 2 < query length <= 2 G (a request joining with its prompt in the prefix cache recomputes the last block:
    query length 16 over the whole context). _verify_kernel_q2 per pair of query tokens: program (KV head, tail row x G +
    pair, split), the batch row read from ROWS; token t of the row sees positions < L - QL + 1 + t. fp16 normalised partials
    [tail row, split, token (2 G), head, D]."""
    T: tl.constexpr = 16 * NP
    h = tl.program_id(0)
    ti = tl.program_id(1) // G
    seq = tl.load(ROWS + ti)
    g = tl.program_id(1) % G
    sp = tl.program_id(2)
    qrow = tl.load(CU_Q + seq)
    ql = tl.load(CU_Q + seq + 1) - qrow
    if (ql <= 2) | (ql > 2 * G) | (2 * g >= ql):
        return
    L = tl.load(SEQ_LENS + seq)
    npg = tl.cdiv(L - ql + 2 * g + 2, 16)       # pages any token of this pair can see
    npg = tl.minimum(npg, tl.cdiv(L, 16))
    per = tl.cdiv(tl.cdiv(L, 16), S)
    lo = sp * per
    hi = tl.maximum(tl.minimum(lo + per, npg), lo)
    rows = tl.arange(0, 16)
    hh = rows % QPK
    tk = 2 * g + rows // QPK
    val = tk < ql
    lim = L - ql + 1 + tk
    c = tl.arange(0, 32)
    t = tl.arange(0, T)
    pg = t // 16
    tt = t % 16
    qb = Q + (qrow + tk).to(tl.int64)[None, :] * s_q0 + (h * QPK + hh)[None, :] * s_q1 + (4 * c)[:, None]
    q0 = tl.load(qb, mask=val[None, :], other=0.0).to(tl.float16)
    q1 = tl.load(qb + 1, mask=val[None, :], other=0.0).to(tl.float16)
    q2 = tl.load(qb + 2, mask=val[None, :], other=0.0).to(tl.float16)
    q3 = tl.load(qb + 3, mask=val[None, :], other=0.0).to(tl.float16)
    qsum = tl.sum(q0.to(tl.float32) + q1.to(tl.float32) + q2.to(tl.float32) + q3.to(tl.float32), 0)
    cv = tl.zeros([16], tl.float32)
    m_i = tl.full([16], float("-inf"), tl.float32)
    l_i = tl.zeros([16], tl.float32)
    a0 = tl.zeros([32, 16], tl.float32)
    a1 = tl.zeros([32, 16], tl.float32)
    a2 = tl.zeros([32, 16], tl.float32)
    a3 = tl.zeros([32, 16], tl.float32)
    tab = TABLE + seq.to(tl.int64) * s_tab
    off = tt[:, None] * s_kt + h * s_kh + c[None, :]
    for j in range(lo, hi, NP):
        pj = j + pg
        pos = pj * 16 + tt
        msk = (pj < hi) & (pos < L)
        blk = tl.load(tab + pj, mask=pj < hi, other=0).to(tl.int64)
        kw = tl.load(K + blk[:, None] * s_kb + off, mask=msk[:, None], other=0)
        vw = tl.trans(tl.load(V + blk[:, None] * s_kb + off, mask=msk[:, None], other=0))
        sk = tl.load(KS + blk * s_sb + tt * s_st + h * s_sh, mask=msk, other=0.0)
        sv = tl.load(VS + blk * s_sb + tt * s_st + h * s_sh, mask=msk, other=0.0)
        y0 = _pair8(kw, 0)
        y1 = _pair8(kw, 1)
        sc = tl.dot(_lo16(y0), q0)
        sc = sc + tl.dot(_lo16(y1), q1)
        sc = sc + tl.dot(_hi16(y0), q2)
        sc = sc + tl.dot(_hi16(y1), q3)
        sc = (sc - 1152.0 * qsum[None, :]) * sk[:, None]
        vis = msk[:, None] & (pos[:, None] < lim[None, :])
        sc = tl.where(vis, sc * qk_scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 0))
        m_use = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(sc - m_use[None, :])
        l_i = l_i * alpha + tl.sum(p, 0)
        ps = (p * (sv * 64.0)[:, None]).to(tl.float16)
        cv = cv * alpha + tl.sum(ps.to(tl.float32), 0)
        z0 = _pair8(vw, 0)
        z1 = _pair8(vw, 1)
        a0 = a0 * alpha[None, :] + tl.dot(_lo16(z0), ps)
        a1 = a1 * alpha[None, :] + tl.dot(_lo16(z1), ps)
        a2 = a2 * alpha[None, :] + tl.dot(_hi16(z0), ps)
        a3 = a3 * alpha[None, :] + tl.dot(_hi16(z1), ps)
        m_i = m_new
    pb = ((ti * S + sp) * (2 * G) + tk) * HQ + h * QPK + hh
    scl = 1.0 / tl.where(l_i > 0.0, l_i, 1.0)
    cvb = (cv * (1152.0 / 64.0) * scl)[None, :]
    r = (scl * (1.0 / 64.0))[None, :]
    po = PO + pb[None, :] * D + (4 * c)[:, None]
    tl.store(po, (a0 * r - cvb).to(tl.float16), mask=val[None, :])
    tl.store(po + 1, (a1 * r - cvb).to(tl.float16), mask=val[None, :])
    tl.store(po + 2, (a2 * r - cvb).to(tl.float16), mask=val[None, :])
    tl.store(po + 3, (a3 * r - cvb).to(tl.float16), mask=val[None, :])
    tl.store(PM + pb, m_i, mask=val)
    tl.store(PL + pb, l_i, mask=val)


@triton.jit(do_not_specialize=["S"])
def _reduce_tail(PO, PM, PL, O, CU_Q, ROWS, s_o0, s_o1, S, SP: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr, G: tl.constexpr):
    ti = tl.program_id(0)
    seq = tl.load(ROWS + ti)
    tok = tl.program_id(1)
    qrow = tl.load(CU_Q + seq)
    ql = tl.load(CU_Q + seq + 1) - qrow
    if (ql <= 2) | (ql > 2 * G) | (tok >= ql):
        return
    rows = tl.arange(0, HQ)
    sidx = tl.arange(0, SP)
    smask = sidx < S
    d = tl.arange(0, D)
    HS: tl.constexpr = 2 * G * HQ
    base = ti * S * HS + tok * HQ
    mm = tl.load(PM + base + sidx[:, None] * HS + rows[None, :], mask=smask[:, None], other=float("-inf"))
    lsum = tl.load(PL + base + sidx[:, None] * HS + rows[None, :], mask=smask[:, None], other=0.0)
    mx = tl.max(mm, 0)
    mx = tl.where(mx > float("-inf"), mx, 0.0)
    denom = tl.sum(tl.exp2(mm - mx[None, :]) * lsum, 0)
    denom = tl.where(denom > 0.0, denom, 1.0)
    acc = tl.zeros([HQ, D], tl.float32)
    for s in range(S):
        w = tl.exp2(tl.load(PM + base + s * HS + rows) - mx) * tl.load(PL + base + s * HS + rows)
        acc += tl.load(PO + (base + s * HS + rows)[:, None] * D + d[None, :]).to(tl.float32) * w[:, None]
    out = acc / denom[:, None]
    tl.store(O + (qrow + tok).to(tl.int64) * s_o0 + rows[:, None] * s_o1 + d[None, :], out.to(O.dtype.element_ty))


def make_tail(query_start_loc_cpu, num_reqs: int, device):
    """Host list of the batch rows with VERIFY_QLEN < query length <= TAIL_QLEN for one step: (rows int32 GPU, count), or None."""
    q = query_start_loc_cpu[:num_reqs + 1]
    ql = q[1:] - q[:-1]
    rows = ((ql > VERIFY_QLEN) & (ql <= TAIL_QLEN)).nonzero().flatten()
    if rows.numel() == 0:
        return None
    return rows.to(torch.int32).to(device, non_blocking=True), int(rows.numel())


def tail_splits(n_tail: int) -> int:
    """KV splits per tail row (E037 attempt 14, p0_tailS: one 16k join 239 us per layer at 12 splits, 138 at 48; two rows
    269 / 229; four or more rows best at 12; 4k rows flat up to 48)."""
    return 48 if n_tail <= 2 else 12


def run_tail(q, views: CacheViews, out, cu_q, seq_lens, table, scale, tail, **flags):
    """Rows listed in ``tail`` (make_tail). Empty splits (beyond what a token pair can see) store m = -inf, l = 0."""
    rows, nseq = tail
    S = flags.pop("S", tail_splits(nseq))
    npv = flags.pop("NP", _NP)
    xopt = flags.pop("xopt", _DEC_FLAGS)
    hq, hkv, d = q.shape[1], views.kc.shape[2], q.shape[2]
    po = torch.empty((nseq * S * TAIL_QLEN * hq, d), dtype=torch.float16, device=q.device)
    pm = torch.empty((nseq * S * TAIL_QLEN * hq,), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    k32, v32, ksc, vsc = views.k32, views.v32, views.ksc, views.vsc
    _tail_kernel[(hkv, nseq * _TAIL_G, S)](
        q, k32, v32, ksc, vsc, table, seq_lens, cu_q, rows, po, pm, pl, scale * _LOG2E,
        q.stride(0), q.stride(1), k32.stride(0), k32.stride(1), k32.stride(2),
        ksc.stride(0), ksc.stride(1), ksc.stride(2), table.stride(0), S,
        HQ=hq, D=d, QPK=hq // hkv, NP=npv, G=_TAIL_G, num_warps=1, num_stages=1, extra_options=xopt, **flags)
    _reduce_tail[(nseq, TAIL_QLEN)](po, pm, pl, out, cu_q, rows, out.stride(0), out.stride(1), S,
                                    SP=triton.next_power_of_2(S), HQ=hq, D=d, G=_TAIL_G, num_warps=4)


def run_verify2(q, views: CacheViews, out, cu_q, seq_lens, table, scale, **flags):
    """Rows with query length 2 (the other rows return at once)."""
    nseq = seq_lens.shape[0]
    S = flags.pop("S", decode_splits(nseq))
    npv = flags.pop("NP", _NP)
    xopt = flags.pop("xopt", _DEC_FLAGS)
    hq, hkv, d = q.shape[1], views.kc.shape[2], q.shape[2]
    po = torch.empty((nseq * S * 2 * hq, d), dtype=torch.float16, device=q.device)
    pm = torch.empty((nseq * S * 2 * hq,), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    k32, v32, ksc, vsc = views.k32, views.v32, views.ksc, views.vsc
    fold = flags.pop("FOLD", FOLD_REDUCE)
    _verify_kernel_q2[(hkv, nseq, S)](
        q, k32, v32, ksc, vsc, table, seq_lens, cu_q, po, pm, pl, scale * _LOG2E,
        q.stride(0), q.stride(1), k32.stride(0), k32.stride(1), k32.stride(2),
        ksc.stride(0), ksc.stride(1), ksc.stride(2), table.stride(0), S,
        _counters(q.device, nseq * hkv), out, out.stride(0), out.stride(1),
        HQ=hq, D=d, QPK=hq // hkv, NP=npv, FOLD=fold, SP=triton.next_power_of_2(S),
        num_warps=1, num_stages=1, extra_options=xopt, **flags)
    if not fold:
        _reduce_q2[(nseq, 2)](po, pm, pl, out, cu_q, out.stride(0), out.stride(1), S,
                              SP=triton.next_power_of_2(S), HQ=hq, D=d, num_warps=4)


@triton.jit(do_not_specialize=["S"])
def _reduce_kernel(PO, PM, PL, O, CU_Q, s_o0, s_o1, S, SP: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr):
    seq = tl.program_id(0)
    qrow = tl.load(CU_Q + seq)
    if tl.load(CU_Q + seq + 1) - qrow != 1:
        return
    rows = tl.arange(0, HQ)
    sidx = tl.arange(0, SP)
    smask = sidx < S
    d = tl.arange(0, D)
    base = seq * S * HQ
    m = tl.load(PM + base + sidx[:, None] * HQ + rows[None, :], mask=smask[:, None], other=float("-inf"))
    lsum = tl.load(PL + base + sidx[:, None] * HQ + rows[None, :], mask=smask[:, None], other=0.0)
    mx = tl.max(m, 0)
    mx = tl.where(mx > float("-inf"), mx, 0.0)      # empty sequence (graph padding): no NaN
    denom = tl.sum(tl.exp2(m - mx[None, :]) * lsum, 0)
    denom = tl.where(denom > 0.0, denom, 1.0)
    acc = tl.zeros([HQ, D], tl.float32)
    for s in range(S):
        w = tl.exp2(tl.load(PM + base + s * HQ + rows) - mx)
        acc += tl.load(PO + (base + s * HQ + rows)[:, None] * D + d[None, :]) * w[:, None]
    out = acc / denom[:, None]
    tl.store(O + qrow.to(tl.int64) * s_o0 + rows[:, None] * s_o1 + d[None, :], out.to(O.dtype.element_ty))


@triton.jit
def _ws_copy_kernel(KC, VC, KS, VS, TABLE, SEQ_LENS, CU_Q, CU_K, KW, VW,
                    s_kb, s_kt, s_kh, s_sb, s_st, s_sh, s_tab, ROW0,
                    HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr):
    """Dequantise one page of K and V of a row with query length > 1 into the bf16 workspace [tokens, HKV, D]
    at CU_K[row] + position."""
    row = tl.program_id(0) + ROW0
    pg = tl.program_id(1)
    if tl.load(CU_Q + row + 1) - tl.load(CU_Q + row) <= 1:
        return
    k0 = tl.load(CU_K + row)
    L = tl.minimum(tl.load(SEQ_LENS + row), tl.load(CU_K + row + 1) - k0)     # never past the row's workspace slice
    if pg * PAGE >= L:
        return
    blk = tl.load(TABLE + row.to(tl.int64) * s_tab + pg).to(tl.int64)
    off = k0.to(tl.int64) + pg * PAGE
    t = tl.arange(0, PAGE)
    d = tl.arange(0, D)
    ok = (pg * PAGE + t) < L
    for g in range(HKV):
        k = tl.load(KC + blk * s_kb + t[:, None] * s_kt + g * s_kh + d[None, :], mask=ok[:, None], other=0)
        v = tl.load(VC + blk * s_kb + t[:, None] * s_kt + g * s_kh + d[None, :], mask=ok[:, None], other=0)
        ks = tl.load(KS + blk * s_sb + t * s_st + g * s_sh, mask=ok, other=0.0)
        vs = tl.load(VS + blk * s_sb + t * s_st + g * s_sh, mask=ok, other=0.0)
        k = (k.to(tl.float32) * ks[:, None]).to(KW.dtype.element_ty)
        v = (v.to(tl.float32) * vs[:, None]).to(VW.dtype.element_ty)
        dst = ((off + t) * HKV + g)[:, None] * D + d[None, :]
        tl.store(KW + dst, k, mask=ok[:, None])
        tl.store(VW + dst, v, mask=ok[:, None])


@triton.jit
def _quant_rows(SRC, s_t, s_h, DST, SC, s_cb, s_cs, s_ch, s_sb, s_ss, s_sh, t, h, blk, pos, ok, D: tl.constexpr):
    d = tl.arange(0, D)
    x = tl.load(SRC + t.to(tl.int64)[:, None] * s_t + h * s_h + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 1) / 127.0, 1e-6)
    q = x * (1.0 / scale)[:, None]
    q = tl.where(q >= 0, q + 0.5, q - 0.5)
    q = tl.clamp(q, -128.0, 127.0)
    tl.store(SC + blk * s_sb + pos * s_ss + h * s_sh, scale, mask=ok)
    tl.store(DST + blk[:, None] * s_cb + pos[:, None] * s_cs + h * s_ch + d[None, :], q.to(tl.int8), mask=ok[:, None])


@triton.jit
def _write_kernel(KEY, VAL, KC, VC, KS, VS, SLOTS, s_kt, s_kh, s_vt, s_vh, s_cb, s_cs, s_ch, s_sb, s_ss, s_sh, NTOK,
                  PAGE: tl.constexpr, D: tl.constexpr, TB: tl.constexpr):
    """Per-(token, head) absmax int8 quantisation of TB tokens at a time into the head-contiguous cache. Same arithmetic as
    vLLM's `_reshape_cache_per_token_head` (scale = max(absmax / 127, 1e-6), round half away from zero, clamp)."""
    h = tl.program_id(1)
    t = tl.program_id(0) * TB + tl.arange(0, TB)
    ok = t < NTOK
    slot = tl.load(SLOTS + t, mask=ok, other=-1).to(tl.int64)
    ok = ok & (slot >= 0)
    blk = slot // PAGE
    pos = slot % PAGE
    _quant_rows(KEY, s_kt, s_kh, KC, KS, s_cb, s_cs, s_ch, s_sb, s_ss, s_sh, t, h, blk, pos, ok, D)
    _quant_rows(VAL, s_vt, s_vh, VC, VS, s_cb, s_cs, s_ch, s_sb, s_ss, s_sh, t, h, blk, pos, ok, D)


def write_kv(key, value, views: CacheViews, slot_mapping, tb: int = 32, num_warps: int = 4):
    """Quantise key / value [tokens, heads, 128] (bf16) into the cache; slot_mapping < 0 skips a token."""
    n, hkv, d = key.shape
    kc, ksc = views.kc, views.ksc
    _write_kernel[(triton.cdiv(n, tb), hkv)](
        key, value, kc, views.vc, ksc, views.vsc, slot_mapping, key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        kc.stride(0), kc.stride(1), kc.stride(2), ksc.stride(0), ksc.stride(1), ksc.stride(2), n,
        PAGE=_PAGE, D=d, TB=tb, num_warps=num_warps)


# --------------------------------------------------------------------------- launchers


def decode_splits(n_rows: int) -> int:
    """KV splits per decode row; a function of the row count only (graph capture). 12 at >= 28 rows with the MAGIC decode
    (E037 attempt 7: 64 x 4608 218.6 us, 64 x 17000 675.6 us vs 228.8 / 695.5 for the previous S 16 form)."""
    return 12 if n_rows >= 28 else 32


def run_decode(q, views: CacheViews, out, cu_q, seq_lens, table, scale, layer_idx: Optional[int] = None, **flags):
    """Rows of the batch with query length 1 (q: [tokens, heads, 128]; cu_q [rows + 1], seq_lens [rows], table [rows, pages])."""
    from . import sparse_decode as SD

    nseq = seq_lens.shape[0]
    flags.setdefault("MAGIC", True)
    S = flags.pop("S", decode_splits(nseq))
    npv = flags.pop("NP", _NP)
    xopt = flags.pop("xopt", _DEC_FLAGS)
    hq, hkv, d = q.shape[1], views.kc.shape[2], q.shape[2]
    po = torch.empty((nseq * S * hq, d), dtype=torch.float32, device=q.device)
    pm = torch.empty((nseq * S * hq,), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    k32, v32, ksc, vsc = views.k32, views.v32, views.ksc, views.vsc
    fold = flags.pop("FOLD", FOLD_REDUCE)

    bank_id, is_anchor, always_dense = SD.layer_role(layer_idx)
    sd = SD.get_state(q.device) if bank_id >= 0 else None
    use_sparse = False
    write_scores = False
    sp_table_ptr = table
    sp_seq_ptr = seq_lens
    row_slot_ptr = seq_lens
    need_ref_ptr = seq_lens
    page_sc_ptr = pm
    sp_slot_stride = 1

    if sd is not None and nseq <= sd.max_reqs:
        if is_anchor:
            sd.prepare_anchor(bank_id, always_dense, cu_q, seq_lens, table)
            write_scores = True
            use_sparse = not always_dense
        else:
            use_sparse = True
            if nseq >= 28:
                S = 6
        sp_table_ptr = sd.sp_table[bank_id]
        sp_seq_ptr = sd.bank_sp_seq[bank_id]
        row_slot_ptr = sd.row_slot
        need_ref_ptr = sd.need_refresh
        page_sc_ptr = sd.page_scores
        sp_slot_stride = SD.MAX_SEL_PAGES

    _decode_kernel_i8[(hkv, nseq, S)](
        q, k32, v32, ksc, vsc, table, seq_lens, cu_q, po, pm, pl, scale * _LOG2E,
        q.stride(0), q.stride(1), k32.stride(0), k32.stride(1), k32.stride(2),
        ksc.stride(0), ksc.stride(1), ksc.stride(2), table.stride(0), S,
        _counters(q.device, nseq * hkv), out, out.stride(0), out.stride(1),
        sp_table_ptr, sp_seq_ptr, row_slot_ptr, need_ref_ptr, page_sc_ptr, sp_slot_stride,
        HQ=hq, D=d, QPK=hq // hkv, NP=npv, FOLD=fold, SP=triton.next_power_of_2(S),
        SPARSE=use_sparse, WRITE_SCORES=write_scores, MAX_PAGES_C=SD.MAX_PAGES,
        num_warps=1, num_stages=1, extra_options=xopt, **flags)
    if not fold:
        _reduce_kernel[(nseq,)](po, pm, pl, out, cu_q, out.stride(0), out.stride(1), S,
                                SP=triton.next_power_of_2(S), HQ=hq, D=d, num_warps=4)
    if sd is not None and is_anchor and nseq <= sd.max_reqs:
        sd.commit_anchor(bank_id, seq_lens, table)


FOLD_REDUCE = True       # merge the KV splits in the last split's program (E037 attempt 19) instead of a reduce launch
_CNT = {}


def _counters(device, n: int) -> torch.Tensor:
    """Zeroed int32 split counters (one per row x KV head); the merging program resets its counter, so the buffer stays zero
    between launches and inside captured graphs. Grown (never shrunk) outside capture."""
    buf = _CNT.get(device)
    if buf is None or buf.numel() < n:
        buf = _CNT[device] = torch.zeros(max(n, 4096), dtype=torch.int32, device=device)
    return buf


@dataclass
class Plan:
    """Per-step prefill plan (rows with query length > 1), built on the host once per step."""
    cu_k: torch.Tensor      # int32 [rows + 1] GPU: workspace token offsets; rows with query length 1 contribute 0 keys
    max_k: int
    max_q: int
    max_pages: int
    n_dec: int              # rows with query length 1
    kw: torch.Tensor        # bf16 [tokens, heads, 128]
    vw: torch.Tensor
    row0: int               # first / one past the last row with query length > 1 (rows in between with length 1 have 0 keys)
    row1: int
    t0: int                 # query token range of those rows
    t1: int
    cu_qp: torch.Tensor     # int32 [row1 - row0 + 1] GPU: query offsets rebased to t0
    cu_kp: torch.Tensor     # int32 [row1 - row0 + 1] GPU: workspace offsets (cu_k[row0] is 0: rows before row0 have no keys)
    first: bool = False     # every row with query length > 1 starts its sequence (seq_len == query_len): its K / V are this call's
                            # own bf16 key / value, no cache read and no workspace (kw / vw are None, cu_kp == cu_qp)


VERIFY_QLEN = 2      # rows with this query length go to the verify kernel (speculative gamma 1), not to the prefill path


def make_plan(common, num_kv_heads: int, head_size: int, device, direct_max: int = VERIFY_QLEN) -> Optional[Plan]:
    """Rows with query length > direct_max (int8 path: TAIL_QLEN; the int4 path keeps VERIFY_QLEN) take the prefill path."""
    if common.max_query_len <= direct_max:
        return None
    n = common.num_reqs
    qsl = common.query_start_loc_cpu
    qlen = qsl[1:n + 1] - qsl[:n]
    # prefill-path rows: the host bound is exact for them (vLLM: optimistic only for async-spec decode rows) and needs no device
    # sync; seq_lens_cpu (a device copy) raced the async-spec input update (E037 attempt 12: _ws_copy_kernel ATU fault)
    ub = getattr(common, "seq_lens_cpu_upper_bound", None)
    slen = (ub if ub is not None else common.seq_lens_cpu)[:n].to(torch.int32)
    act = qlen > direct_max
    lens = torch.where(act, slen, torch.zeros_like(slen))
    cu = torch.zeros(n + 1, dtype=torch.int32)
    cu[1:] = lens.cumsum(0, dtype=torch.int32)
    total = int(cu[n])
    max_k = int(lens.max())
    idx = act.nonzero()
    r0, r1 = int(idx[0]), int(idx[-1]) + 1
    t0 = int(qsl[r0])
    cu_qp = (qsl[r0:r1 + 1] - t0).to(torch.int32).to(device, non_blocking=True)
    if bool((slen[act] == qlen[act].to(torch.int32)).all()):
        # first chunk of every prefilling sequence: attend over the bf16 key / value of this call (also saves the dequantise copy)
        return Plan(cu_k=cu_qp, max_k=int(qlen.max()), max_q=int(qlen.max()), max_pages=0, n_dec=int((~act).sum()), kw=None, vw=None,
                    row0=r0, row1=r1, t0=t0, t1=int(qsl[r1]), cu_qp=cu_qp, cu_kp=cu_qp, first=True)
    kw = torch.empty((total, num_kv_heads, head_size), dtype=torch.bfloat16, device=device)
    vw = torch.empty_like(kw)
    cu_dev = cu.to(device, non_blocking=True)
    return Plan(cu_k=cu_dev, max_k=max_k, max_q=int(qlen.max()),
                max_pages=-(-max_k // _PAGE), n_dec=int((~act).sum()), kw=kw, vw=vw, row0=r0, row1=r1, t0=t0, t1=int(qsl[r1]),
                cu_qp=cu_qp, cu_kp=cu_dev[r0:r1 + 1])


def run_prefill(q, views: CacheViews, out, cu_q, seq_lens, table, scale, plan: Plan, flash_attn_varlen_func, key=None, value=None):
    if plan.first and key is not None:
        out[plan.t0:plan.t1].copy_(flash_attn_varlen_func(
            q=q[plan.t0:plan.t1], k=key[plan.t0:plan.t1], v=value[plan.t0:plan.t1], cu_seqlens_q=plan.cu_qp, cu_seqlens_k=plan.cu_qp,
            max_seqlen_q=plan.max_q, max_seqlen_k=plan.max_q, softmax_scale=scale, causal=True))
        return
    if plan.first:      # no key / value handed in (kv sharing): read the cache
        raise RuntimeError("E034: first-chunk plan needs the layer's key / value")
    hkv, d = views.kc.shape[2], q.shape[2]
    kc, ksc = views.kc, views.ksc
    _ws_copy_kernel[(plan.row1 - plan.row0, plan.max_pages)](
        views.kc, views.vc, ksc, views.vsc, table, seq_lens, cu_q, plan.cu_k, plan.kw, plan.vw,
        kc.stride(0), kc.stride(1), kc.stride(2), ksc.stride(0), ksc.stride(1), ksc.stride(2), table.stride(0), plan.row0,
        HKV=hkv, D=d, PAGE=_PAGE, num_warps=4)
    # only the rows that need it: the decode rows in this call cost 1-2 ms per layer (E034 attempt 13)
    out[plan.t0:plan.t1].copy_(flash_attn_varlen_func(
        q=q[plan.t0:plan.t1], k=plan.kw, v=plan.vw, cu_seqlens_q=plan.cu_qp, cu_seqlens_k=plan.cu_kp,
        max_seqlen_q=plan.max_q, max_seqlen_k=plan.max_k, softmax_scale=scale, causal=True))


def attend(q, views: CacheViews, out, cu_q, seq_lens, table, scale, plan: Optional[Plan], flash_attn_varlen_func, key=None, value=None,
           max_query_len: int = 1, tail=None, layer_idx: Optional[int] = None):
    """q, out: [tokens, 16, 128]. Rows with query length > TAIL_QLEN: prefill path (plan); then the decode kernel (query length 1
    rows), the verify kernel (query length 2) and the tail kernel (3..TAIL_QLEN) write their rows."""
    if plan is not None:
        from . import pyramid_prefill as PP
        cap = PP.get_active_capture()
        if cap is not None and key is not None:
            if layer_idx == cap.target_layer:
                PP.capture_and_select(cap, q[plan.t0:plan.t1], key[plan.t0:plan.t1], scale, commit=True)
            elif layer_idx in cap.vote_layers:
                PP.capture_and_select(cap, q[plan.t0:plan.t1], key[plan.t0:plan.t1], scale, commit=False)
        if (
            plan.first
            and plan.max_q >= 4096
            and layer_idx is not None
            and layer_idx not in (0, 28, 41)
            and key is not None
            and value is not None
        ):
            from . import metax_causal_prefill as _CP
            _CP.causal_prefill_varlen(
                q[plan.t0:plan.t1], key[plan.t0:plan.t1], value[plan.t0:plan.t1],
                plan.cu_qp, plan.max_q, scale, out=out[plan.t0:plan.t1],
            )
        else:
            run_prefill(q, views, out, cu_q, seq_lens, table, scale, plan, flash_attn_varlen_func, key, value)
    if plan is None or plan.n_dec > 0:
        run_decode(q, views, out, cu_q, seq_lens, table, scale, layer_idx=layer_idx)
        if max_query_len >= VERIFY_QLEN:
            run_verify2(q, views, out, cu_q, seq_lens, table, scale)
        if tail is not None:
            run_tail(q, views, out, cu_q, seq_lens, table, scale, tail)


# --------------------------------------------------------------------------- start-up self-check


def _reference(q, kd, vd, cu, seq_lens, table, scale):
    """fp32 attention of every row; kd / vd: dequantised cache [blocks, 16, heads, 128] (CPU)."""
    outs = []
    for i in range(seq_lens.shape[0]):
        n, q0, q1 = int(seq_lens[i]), int(cu[i]), int(cu[i + 1])
        if q1 == q0:
            continue
        if n == 0:
            outs.append(torch.zeros_like(q[q0:q1], dtype=torch.float32))
            continue
        ids = table[i, :-(-n // _PAGE)].long()
        kk = kd[ids].reshape(-1, kd.shape[2], _HEAD)[:n].repeat_interleave(_QPK, 1)
        vv = vd[ids].reshape(-1, vd.shape[2], _HEAD)[:n].repeat_interleave(_QPK, 1)
        qs = q[q0:q1].float()
        sc = torch.einsum("qhd,khd->hqk", qs, kk) * scale
        qpos = torch.arange(n - qs.shape[0], n)[:, None]
        sc = sc.masked_fill(torch.arange(n)[None, :] > qpos, float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", sc.softmax(-1), vv))
    return torch.cat(outs)


def _chunk_kv(x, query_lens, seq_lens):
    """The key / value rows of this step's tokens (the last query_len tokens of each sequence), as the layer receives them."""
    parts, off = [], 0
    for ql, sl in zip(query_lens, seq_lens):
        parts.append(x[off + sl - ql:off + sl])
        off += sl
    return torch.cat(parts)


def _check_case(seq_lens, query_lens, device, gen, flash_attn_varlen_func) -> float:
    from types import SimpleNamespace

    hkv, scale = 2, _HEAD ** -0.5
    pages = [-(-n // _PAGE) for n in seq_lens]
    total = sum(pages)
    table = torch.zeros((len(seq_lens), max(max(pages), 1)), dtype=torch.int32)
    perm = torch.randperm(total, generator=gen).to(torch.int32) + 1
    pos = 0
    for i, n in enumerate(pages):
        table[i, :n] = perm[pos:pos + n]
        pos += n
    ntok = sum(seq_lens)
    key = torch.randn((ntok, hkv, _HEAD), generator=gen).to(device=device, dtype=torch.bfloat16)
    val = torch.randn((ntok, hkv, _HEAD), generator=gen).to(device=device, dtype=torch.bfloat16)
    slots = torch.cat([table[i, (torch.arange(n) // _PAGE)].long() * _PAGE + torch.arange(n) % _PAGE
                       for i, n in enumerate(seq_lens) if n > 0]).to(device=device, dtype=torch.int64)
    raw = torch.zeros((total + 1, 2, _PAGE, hkv, _HEAD + 4), dtype=torch.int8, device=device)
    views = make_views(raw, False)
    write_kv(key, val, views, slots)      # the real cache write path
    q = (torch.randn((sum(query_lens), _QPK * hkv, _HEAD), generator=gen) * 8).to(device=device, dtype=torch.bfloat16)
    cu_cpu = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    cu = cu_cpu.to(device)
    su_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    su = su_cpu.to(device)
    tab = table.to(device)
    out = torch.zeros_like(q)
    plan = make_plan(SimpleNamespace(max_query_len=max(query_lens), num_reqs=len(seq_lens), query_start_loc_cpu=cu_cpu,
                                     seq_lens_cpu=su_cpu), hkv, _HEAD, device, TAIL_QLEN)
    attend(q, views, out, cu, su, tab, scale, plan, flash_attn_varlen_func, _chunk_kv(key, query_lens, seq_lens), _chunk_kv(val, query_lens, seq_lens),
           max_query_len=max(query_lens), tail=make_tail(cu_cpu, len(seq_lens), device))
    torch.cuda.synchronize(device)
    if not torch.isfinite(out.float()).all():
        return float("inf")
    kd = (views.kc.float() * views.ksc[..., None]).cpu()
    vd = (views.vc.float() * views.vsc[..., None]).cpu()
    if _REF_ROUND_BF16:         # diagnostic: the prefill workspace holds bf16 values
        kd, vd = kd.to(torch.bfloat16).float(), vd.to(torch.bfloat16).float()
    off = 0
    first = plan is not None and plan.first
    for i, (sl, ql) in enumerate(zip(seq_lens, query_lens)):     # a first-chunk step attends over the exact bf16 key / value
        if first and ql > TAIL_QLEN:
            p = torch.arange(sl)
            kd[table[i, p // _PAGE].long(), p % _PAGE] = key[off:off + sl].float().cpu()
            vd[table[i, p // _PAGE].long(), p % _PAGE] = val[off:off + sl].float().cpu()
        off += sl
    ref = _reference(q.cpu(), kd, vd, cu_cpu, su_cpu, table, scale)
    return ((out.float().cpu() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()


def self_check(device, flash_attn_varlen_func) -> float:
    """Worst relative error (max abs / max |ref|) over small synthetic batches: pure decode (both split counts, page edges,
    padding rows), chunked prefill with context, mixed steps. The cache is written by the real write kernel."""
    gen = torch.Generator().manual_seed(0)
    cases = [
        ([1], [1]),
        ([17, 300, 4097, 16, 15, 33, 1600, 0], [1] * 8),
        ([700 + 97 * i for i in range(28)] + [0] * 4, [1] * 32),
        ([17, 4127, 300, 1], [17, 31, 1, 1]),
        ([2600] * 6 + [2048 + 300] + [900] * 25, [1] * 6 + [300] + [1] * 25),
        ([40, 40, 90], [40, 40, 90]),                                     # first chunks only (bf16 key / value path)
        ([700, 900, 300, 1500, 300], [1, 1, 300, 1500, 300]),             # decode rows + first chunks
        ([700, 900, 300, 2000, 300], [1, 1, 300, 1500, 300]),             # decode rows + first chunks + one continuing chunk (workspace)
        ([17, 300, 4097, 16, 33, 1600, 129, 8], [2] * 8),                 # speculative verify rows (query length 2)
        ([700 + 97 * i for i in range(28)] + [2] * 4, [2] * 32),          # verify rows, 28+ rows (S 12), short rows
        ([900, 4600, 300, 2000], [2, 1, 2, 300]),                         # verify + decode + first chunk in one step
        ([16384, 4096, 17, 3, 16, 40], [16, 16, 3, 3, 16, 7]),            # tail rows (prefix-cache joins, odd lengths, first chunks)
        ([4100] * 30 + [16400, 900], [2] * 30 + [16, 1]),                 # verify rows (S 12) + one join + decode
        ([2000, 700, 3000], [16, 17, 300]),                               # tail row next to prefill-path rows
    ]
    try:
        return max(_check_case(sl, ql, device, gen, flash_attn_varlen_func) for sl, ql in cases)
    finally:
        torch.cuda.empty_cache()

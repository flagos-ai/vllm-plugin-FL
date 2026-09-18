# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.

"""Exact per-chunk softmax state materialization, then two PV-half CTAs.
Chunk order and BF16 probability conversion match the serial attention path."""

import torch
import triton
import triton.language as tl


@triton.jit
def qk_states(
    Q,
    KV,
    INDICES,
    SINK,
    LENS,
    P,
    A,
    F,
    MAX,
    LSE,
    SQ,
    SKV,
    TOPK: tl.constexpr,
    NC: tl.constexpr,
    SINK_ON: tl.constexpr,
    LEN_ON: tl.constexpr,
    SCALE: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, 16)
    d = tl.arange(0, 256)
    k = tl.arange(0, 16)
    qp = Q + t * 16 * 512 + h[:, None] * 512 + d[None, :]
    q0 = tl.load(qp, eviction_policy="evict_first")
    q1 = tl.load(qp + 256, eviction_policy="evict_first")
    max_log = tl.full([16], -float("inf"), tl.float32)
    sum_exp = tl.full([16], 0.0, tl.float32)
    topk_len = tl.load(LENS + t) if LEN_ON else TOPK
    for ck in range(tl.cdiv(topk_len, 16)):
        ki = ck * 16 + k
        ids = tl.load(INDICES + t * TOPK + ki, ki < topk_len, other=-1)
        valid = (ids < SKV) & (ids >= 0)
        ids = tl.where(valid, ids, 0)
        kp = KV + d[:, None] + ids[None, :] * 512
        kv0 = tl.load(kp, cache_modifier=".cg")
        kv1 = tl.load(kp + 256, cache_modifier=".cg")
        qk = tl.dot(q0, kv0, out_dtype=tl.float32)
        qk = tl.dot(q1, kv1, qk, out_dtype=tl.float32)
        qk *= SCALE
        qk = tl.where(valid[None, :], qk, -float("inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        tl.store(
            P + ((t * NC + ck) * 16 + h[:, None]) * 16 + k[None, :],
            exp_qk.to(tl.bfloat16),
        )
        tl.store(A + (t * NC + ck) * 16 + h, alpha)
        max_log = new_max
    valid_mask = max_log != -float("inf")
    max_log = tl.where(valid_mask, max_log, -float("inf"))
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    if SINK_ON:
        sink = tl.load(SINK + h)
        factor = tl.math.exp(max_log) / (tl.math.exp(orig_lse) + tl.math.exp(sink))
    else:
        factor = 1.0 / sum_exp
    tl.store(F + t * 16 + h, factor)
    tl.store(MAX + t * 16 + h, max_log)
    tl.store(LSE + t * 16 + h, lse_out)


@triton.jit
def pv_replay(
    KV,
    INDICES,
    LENS,
    P,
    A,
    F,
    MAX,
    OUT,
    SQ,
    SKV,
    TOPK: tl.constexpr,
    NC: tl.constexpr,
    LEN_ON: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    vs = tl.program_id(1) * 256
    h = tl.arange(0, 16)
    d = tl.arange(0, 256)
    k = tl.arange(0, 16)
    acc = tl.zeros([16, 256], tl.float32)
    topk_len = tl.load(LENS + t) if LEN_ON else TOPK
    for ck in range(tl.cdiv(topk_len, 16)):
        ki = ck * 16 + k
        ids = tl.load(INDICES + t * TOPK + ki, ki < topk_len, other=-1)
        valid = (ids < SKV) & (ids >= 0)
        ids = tl.where(valid, ids, 0)
        vp = KV + (vs + d[:, None]) + ids[None, :] * 512
        v = tl.load(vp, cache_modifier=".cg")
        p = tl.load(P + ((t * NC + ck) * 16 + h[:, None]) * 16 + k[None, :])
        a = tl.load(A + (t * NC + ck) * 16 + h)
        acc = tl.dot(p, v.trans(), acc * a[:, None], out_dtype=tl.float32)
    factor = tl.load(F + t * 16 + h)
    valid_out = tl.load(MAX + t * 16 + h) != -float("inf")
    out = tl.where(valid_out[:, None], acc * factor[:, None], 0.0)
    tl.store(
        OUT + t * 16 * 512 + h[:, None] * 512 + (vs + d[None, :]), out.to(tl.bfloat16)
    )


def buffers(q, topk):
    t = q.shape[0]
    nc = triton.cdiv(topk, 16)
    return (
        torch.empty((t, nc, 16, 16), device=q.device, dtype=torch.bfloat16),
        torch.empty((t, nc, 16), device=q.device, dtype=torch.float32),
        torch.empty((t, 16), device=q.device, dtype=torch.float32),
    )


def run(q, kv, idx, sink, lens, out, ml, lse, work, qw=2, pw=2, sm_scale=512**-0.5):
    assert q.shape[1:] == (16, 512) and q.is_contiguous()
    assert kv.shape[1:] == (1, 512) and kv.is_contiguous()
    assert idx.is_contiguous() and out.is_contiguous()
    t = q.shape[0]
    n = idx.shape[-1]
    nc = triton.cdiv(n, 16)
    p, a, f = work
    k1 = qk_states[(t,)](
        q,
        kv,
        idx,
        sink,
        lens,
        p,
        a,
        f,
        ml,
        lse,
        t,
        kv.shape[0],
        n,
        nc,
        sink is not None,
        lens is not None,
        sm_scale,
        num_warps=qw,
        num_stages=1,
    )
    k2 = pv_replay[(t, 2)](
        kv,
        idx,
        lens,
        p,
        a,
        f,
        ml,
        out,
        t,
        kv.shape[0],
        n,
        nc,
        lens is not None,
        num_warps=pw,
        num_stages=1,
    )
    return [k1, k2]

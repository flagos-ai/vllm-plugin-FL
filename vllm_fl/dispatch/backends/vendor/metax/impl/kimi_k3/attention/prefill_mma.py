"""K3 MLA prefill: BF16 tiled QK/PV with short register lifetime and FP32 online state."""

from flag_gems import runtime


import torch, triton
import triton.language as tl


@triton.jit
def _prefill_mma(
    Q,
    K,
    V,
    CQ,
    CK,
    O,
    L,
    H: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    COMP: tl.constexpr,
    KT: tl.constexpr,
    VT: tl.constexpr,
    PM,
    PS,
    NQ,
    SPLITS: tl.constexpr,
    LSE_HEAD_MAJOR: tl.constexpr = False,
):
    block = tl.program_id(0) // SPLITS
    split = tl.program_id(0) % SPLITS
    head = tl.program_id(1) // (128 // VT)
    vblock = tl.program_id(1) % (128 // VT)
    seq = tl.program_id(2)
    q0 = tl.load(CQ + seq)
    q1 = tl.load(CQ + seq + 1)
    k0 = tl.load(CK + seq)
    k1 = tl.load(CK + seq + 1)
    nq = q1 - q0
    nk = k1 - k0
    first = block * BQ
    if first >= nq:
        return
    qr = first + tl.arange(0, BQ)
    dv = vblock * VT + tl.arange(0, VT)
    nn = tl.arange(0, BK)
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    denom = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, VT), tl.float32)
    limit = tl.maximum(0, tl.minimum(nk, nk - nq + first + BQ)) if CAUSAL else nk
    split_blocks = tl.cdiv(tl.cdiv(limit, BK), SPLITS)
    begin = split * split_blocks * BK
    end = tl.minimum((split + 1) * split_blocks * BK, limit)
    for start in range(begin, end, BK):
        n = start + nn
        scores = tl.zeros((BQ, BK), tl.float32)
        for kd in range(0, 192, KT):
            d = kd + tl.arange(0, KT)
            qa = tl.load(
                Q + ((q0 + qr[:, None]) * H + head) * 192 + d[None, :],
                (qr[:, None] < nq) & (d[None, :] < 192),
                0,
            )
            keys = tl.load(
                K + ((k0 + n[None, :]) * H + head) * 192 + d[:, None],
                (n[None, :] < nk) & (d[:, None] < 192),
                0,
            )
            scores = tl.dot(qa, keys, scores)
        scores = scores * SCALE
        mask = (n[None, :] < nk) & (qr[:, None] < nq)
        if CAUSAL:
            mask = mask & (n[None, :] <= nk - nq + qr[:, None])
        scores = tl.where(mask, scores, -float("inf"))
        cur = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.where(maximum != -float("inf"), tl.exp(maximum - cur), 0.0)
        prob = tl.where(mask, tl.exp(scores - cur[:, None]), 0.0)
        vals = tl.load(
            V + ((k0 + n[:, None]) * H + head) * 128 + dv[None, :], n[:, None] < nk, 0
        )
        hi = prob.to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc = tl.dot(hi, vals, acc)
        if COMP:
            lo = (prob - hi.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(lo, vals, acc)
            tail = (prob - hi.to(tl.float32) - lo.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(tail, vals, acc)
        denom = denom * alpha + tl.sum(prob, 1)
        maximum = cur
    if SPLITS > 1:
        tl.store(
            O + ((split * NQ + q0 + qr[:, None]) * H + head) * 128 + dv[None, :],
            acc,
            qr[:, None] < nq,
        )
        if vblock == 0:
            tl.store(PM + (split * NQ + q0 + qr) * H + head, maximum, qr < nq)
            tl.store(PS + (split * NQ + q0 + qr) * H + head, denom, qr < nq)
    else:
        out = tl.where(denom[:, None] > 0, acc / denom[:, None], 0.0)
        tl.store(
            O + ((q0 + qr[:, None]) * H + head) * 128 + dv[None, :],
            out,
            qr[:, None] < nq,
        )
        if vblock == 0:
            lse_offset = head * NQ + q0 + qr if LSE_HEAD_MAJOR else (q0 + qr) * H + head
            tl.store(
                L + lse_offset,
                tl.where(denom > 0, maximum + tl.log(denom), -float("inf")),
                qr < nq,
            )


@triton.jit
def _merge_prefill_splits(P, M, S, O, L, NQ, H: tl.constexpr, SPLITS: tl.constexpr):
    q = tl.program_id(0)
    h = tl.program_id(1)
    part = tl.arange(0, SPLITS)
    d = tl.arange(0, 128)
    idx = (part * NQ + q) * H + h
    maximum = tl.load(M + idx)
    denom = tl.load(S + idx)
    overall = tl.max(maximum, 0)
    weights = tl.where(denom > 0, tl.exp(maximum - overall), 0.0)
    values = tl.load(P + idx[:, None] * 128 + d[None, :])
    total = tl.sum(weights * denom, 0)
    result = tl.sum(values * weights[:, None], 0)
    tl.store(O + (q * H + h) * 128 + d, tl.where(total > 0, result / total, 0.0))
    tl.store(L + q * H + h, tl.where(total > 0, overall + tl.log(total), -float("inf")))


def can_use(q, k, v, cq, ck, maxq, scale):
    if runtime.device.vendor_name != "metax":
        return False
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v, cq, ck)):
        return False
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or cq.ndim != 1 or ck.ndim != 1:
        return False
    if (
        q.shape[1] != 12
        or q.shape[2] != 192
        or k.shape[1:] != q.shape[1:]
        or v.shape[1:] != (q.shape[1], 128)
        or k.shape[0] != v.shape[0]
        or k.shape[0] > 66048
    ):
        return False
    if not isinstance(maxq, int) or not (1 <= q.shape[0] <= 4096 and 1 <= maxq <= 4096):
        return False
    if cq.shape != ck.shape or cq.numel() < 2:
        return False
    if not q.is_cuda or any(
        x.device != q.device or not x.is_contiguous() for x in (q, k, v, cq, ck)
    ):
        return False
    if (
        not all(x.dtype == torch.bfloat16 for x in (q, k, v))
        or cq.dtype != torch.int32
        or ck.dtype != torch.int32
    ):
        return False
    return isinstance(scale, (float, int)) and scale == 192**-0.5


def configuration(totalq, maxq, causal):
    # Keep compile-time tiles independent of GPU sequence-length values.
    if totalq >= 2048:
        return 32, 32, 2, 32, 64
    if maxq <= 768:
        return 16, 64, 4, 64, 64
    return 32, 32, 4, 64, 64


def run(
    q,
    k,
    v,
    cq,
    ck,
    maxq,
    scale,
    causal=True,
    record=None,
    *,
    out=None,
    head_major_lse=False,
):
    h = q.shape[1]
    bq, bk, warps, kt, vt = configuration(q.shape[0], maxq, causal)
    # The real new-token caller shares the same cumulative lengths for Q/K.
    # Context fragments retain their previous configuration.
    if (
        h == 12
        and causal
        and 449 <= q.shape[0] <= 512
        and k.shape[0] == q.shape[0]
        and cq is ck
    ):
        bq, bk, warps, kt, vt = 64, 32, 4, 32, 64
    o = (
        torch.empty((q.shape[0], h, 128), dtype=torch.float32, device=q.device)
        if out is None
        else out
    )
    l = torch.empty(
        (h, q.shape[0]) if head_major_lse else q.shape[:2],
        dtype=torch.float32,
        device=q.device,
    )
    # Prefix fragments with very few queries otherwise launch too few blocks
    # and sum thousands of KV tiles serially. Split only that independent axis.
    splits = 16 if not causal and maxq <= 64 and k.shape[0] > 4096 else 1
    # TP8 context tiles reuse each KV load across more query rows. Long chunks
    # split the independent KV axis to bound serial FP32 denominator accumulation.
    if (
        h == 12
        and not causal
        and 65 <= q.shape[0] <= 512
        and k.shape[0] >= 2048
        and (128 <= maxq <= 512 or (maxq == 64 and q.shape[0] == 512))
    ):
        bq, bk, warps, kt, vt = 64, 32, 4, 32, 64
        splits = (
            1
            if k.shape[0] <= 4096
            else (16 if 128 <= maxq <= 256 and k.shape[0] > 32768 else 8)
        )

    # Large TP8 prefill query blocks reuse KV over64 queries and64 key rows.
    # Long context fragments use four independent KV partitions for stable FP32
    # online statistics. Selection uses CPU tensor geometry, neverGPU length reads.
    if h == 12 and 513 <= q.shape[0] <= 4096 and maxq >= 128:
        bq, bk, warps, kt, vt = 64, 64, 4, 64, 64
        splits = 4 if not causal and k.shape[0] > 32768 else 1

    if head_major_lse and splits > 1:
        raise ValueError(
            "Direct head-major output is only qualified for new-token attention"
        )
    partial = (
        torch.empty((splits, q.shape[0], h, 128), dtype=torch.float32, device=q.device)
        if splits > 1
        else o
    )
    pm = (
        torch.empty((splits, q.shape[0], h), dtype=torch.float32, device=q.device)
        if splits > 1
        else l
    )
    ps = torch.empty_like(pm) if splits > 1 else l
    kern = _prefill_mma[
        (triton.cdiv(maxq, bq) * splits, h * (128 // vt), cq.numel() - 1)
    ](
        q,
        k,
        v,
        cq,
        ck,
        partial,
        l,
        h,
        scale,
        causal,
        bq,
        bk,
        True,
        kt,
        vt,
        pm,
        ps,
        q.shape[0],
        splits,
        LSE_HEAD_MAJOR=head_major_lse,
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    merge = None
    if splits > 1:
        merge = _merge_prefill_splits[(q.shape[0], h)](
            partial,
            pm,
            ps,
            o,
            l,
            q.shape[0],
            h,
            splits,
            num_warps=4,
            enable_fp_fusion=False,
        )
    if record is not None:
        record.update(
            n_regs=kern.n_regs,
            n_spills=kern.n_spills,
            shared=kern.metadata.shared,
            BQ=bq,
            BK=bk,
            warps=warps,
            KT=kt,
            VT=vt,
            compensated=True,
            probability_bf16_parts=3,
            score_matrix_materialized=False,
            splits=splits,
            merge_spills=None if merge is None else merge.n_spills,
        )
    return o, l

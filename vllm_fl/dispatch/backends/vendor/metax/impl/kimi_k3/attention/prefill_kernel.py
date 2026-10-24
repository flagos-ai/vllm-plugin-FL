"""FP32 diagnostic attention with independent QK and value dimensions."""

import torch
import triton
import triton.language as tl
import vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.prefill_mma as prefill_mma


@triton.jit
def _attention(
    Q,
    K,
    V,
    CQ,
    CK,
    O,
    LSE,
    H: tl.constexpr,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
):
    qi = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    q0 = tl.load(CQ + batch)
    q1 = tl.load(CQ + batch + 1)
    k0 = tl.load(CK + batch)
    k1 = tl.load(CK + batch + 1)
    qlen = q1 - q0
    klen = k1 - k0
    if qi >= qlen:
        return
    d = tl.arange(0, BD)
    dv = tl.arange(0, BV)
    q = tl.load(Q + ((q0 + qi) * H + head) * DQ + d, d < DQ, other=0).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BV,), tl.float32)
    limit = tl.minimum(klen, klen - qlen + qi + 1) if CAUSAL else klen
    for start in range(0, limit, BK):
        n = start + tl.arange(0, BK)
        keys = tl.load(
            K + ((k0 + n[:, None]) * H + head) * DQ + d[None, :],
            (n[:, None] < limit) & (d[None, :] < DQ),
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1) * SCALE
        scores = tl.where(n < limit, scores, -float("inf"))
        current = tl.maximum(maximum, tl.max(scores, 0))
        alpha = tl.exp(maximum - current)
        p = tl.exp(scores - current)
        values = tl.load(
            V + ((k0 + n[:, None]) * H + head) * DV + dv[None, :],
            (n[:, None] < limit) & (dv[None, :] < DV),
            other=0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * values, axis=0)
        denom = denom * alpha + tl.sum(p, 0)
        maximum = current
    result = tl.where(denom > 0, acc / denom, 0.0)
    tl.store(O + ((q0 + qi) * H + head) * DV + dv, result, dv < DV)
    tl.store(
        LSE + (q0 + qi) * H + head,
        tl.where(denom > 0, maximum + tl.log(denom), -float("inf")),
    )


def attention(q, k, v, cq, ck, max_q, scale, causal=True):
    if prefill_mma.can_use(q, k, v, cq, ck, max_q, scale):
        return prefill_mma.run(q, k, v, cq, ck, max_q, scale, causal)
    if q.shape[-1] != 192 or k.shape[-1] != 192 or v.shape[-1] != 128:
        raise ValueError("K3 prefill requires QK192/V128")
    h = q.shape[1]
    out = torch.empty((q.shape[0], h, 128), dtype=torch.float32, device=q.device)
    lse = torch.empty((q.shape[0], h), dtype=torch.float32, device=q.device)
    _attention[(max_q, h, cq.numel() - 1)](
        q,
        k,
        v,
        cq,
        ck,
        out,
        lse,
        h,
        192,
        128,
        scale,
        causal,
        16,
        256,
        128,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out, lse

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Naive expert GEMV candidates. No route/activation/sum changes."""

import triton
import triton.language as tl


@triton.jit
def naive_gemv(
    A,
    W,
    C,
    RW,
    IDS,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    nb = tl.program_id(1)
    ns = nb * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    expert = tl.load(IDS + row).to(tl.int64)
    acc = tl.full((BN, BK), 0, tl.float32)
    for off in range(tl.cdiv(K, BK)):
        kk = off * BK + ks
        x = tl.load(A + (row // TOPK) * K + kk, kk < K, 0).to(tl.float32)
        w = tl.load(
            W + expert * N * K + ns[:, None] * K + kk[None, :],
            (ns[:, None] < N) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        acc += x[None, :] * w
    value = tl.sum(acc, 1)
    if WEIGHTED:
        value *= tl.load(RW + row)
    tl.store(C + row * N + ns, value, ns < N)


def expert_gemv(a, weight, out, routed_weights, expert_ids, top_k, weighted, config):
    """Consume the M3 expert layout validated by ExpertLaunchContext."""
    return naive_gemv[
        (expert_ids.numel(), triton.cdiv(weight.shape[1], config["BN"]))
    ](
        a, weight, out, routed_weights, expert_ids,
        weight.shape[1], weight.shape[2], top_k, weighted,
        config["BN"], config["BK"], num_warps=config["num_warps"],
        num_stages=1, enable_fp_fusion=False,
    )

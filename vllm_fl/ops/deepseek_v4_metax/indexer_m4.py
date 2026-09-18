# SPDX-License-Identifier: Apache-2.0
"""Shard complete query rows, then gather exact top-512 indices (never KV columns)."""

import datetime
import os
import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from .indexer_score import mqa_hoist_skip_kernel

_GROUPS = {}


def row_range(m, owner, degree=4):
    if degree != 4 or not 0 <= owner < degree:
        raise ValueError("Only the validated four-rank row partition is supported")
    chunk = (m + degree - 1) // degree
    return chunk, owner * chunk, min(m, (owner + 1) * chunk)


def _group():
    tp = get_tp_group()
    if tp.world_size != 8 or dist.get_world_size() != 8:
        raise RuntimeError("M/4 requires a single TP8 group, no DP/PP")
    key = (id(tp.device_group), tuple(tp.ranks))
    if key not in _GROUPS:
        own = None
        # ALL eight ranks create BOTH groups in the same order.
        for start in (0, 4):
            group = dist.new_group(
                ranks=tp.ranks[start : start + 4],
                backend="nccl",
                timeout=datetime.timedelta(seconds=180),
            )
            if start <= tp.rank_in_group < start + 4:
                own = (group, tp.rank_in_group - start)
        _GROUPS[key] = own
    return _GROUPS[key]


def try_prefill_indexer_m4(
    q, kv, weights, ks, ke, out, topk_tokens, topk_into, reference_logits
):
    if os.getenv("VLLM_FL_METAX_INDEXER_M84_MODE", "off") != "on":
        return False
    m, h, d = q.shape
    n = kv.shape[0]
    tp = get_tp_group()
    eligible = (
        tp.world_size == 8
        and dist.get_world_size() == 8
        and m >= 2048
        and n >= 4096
        and h == 64
        and d == 128
        and topk_tokens == 512
        and q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and weights.dtype == torch.float32
        and ks.dtype == ke.dtype == out.dtype == torch.int32
        and all(x.is_contiguous() for x in (q, kv, weights, ks, ke, out))
        and weights.shape == (m, h)
        and kv.shape == (n, d)
        and ks.numel() == ke.numel() == m
        and out.shape == (m, 512)
    )
    if not eligible:
        return False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("M/4 prefill must not be captured; use FULL_DECODE_ONLY")
    group, owner = _group()
    chunk, lo, hi = row_range(m, owner)
    pitch = (n + 63) // 64 * 64
    qq, ww = q[lo:hi], weights[lo:hi]
    kk, ee = ks.reshape(-1)[lo:hi], ke.reshape(-1)[lo:hi]
    storage = torch.zeros((chunk, pitch), device=q.device, dtype=torch.float32)
    scores = storage[:, :n]
    mqa_hoist_skip_kernel[((hi - lo + 63) // 64, (n + 63) // 64)](
        qq,
        kv,
        ww,
        kk,
        ee,
        scores,
        hi - lo,
        n,
        h,
        d,
        qq.stride(0),
        qq.stride(1),
        qq.stride(2),
        kv.stride(0),
        kv.stride(1),
        ww.stride(0),
        ww.stride(1),
        scores.stride(0),
        scores.stride(1),
        True,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=1,
        pipeline="basic",
        scenario="",
    )
    ids = torch.full((chunk, 512), -1, device=q.device, dtype=torch.int32)
    # Reuse the same top-k implementation as the non-sharded path. Its row
    # offsets, short-row padding and tie ordering are part of the contract.
    topk_into(scores[: hi - lo], kk, ee, ids[: hi - lo])
    gathered = torch.empty((4 * chunk, 512), device=q.device, dtype=torch.int32)
    dist.all_gather_into_tensor(gathered, ids, group=group)
    candidate = gathered[:m]
    if os.getenv("VLLM_FL_DSV4_METAX_VERIFY_M4", "1") == "1":
        scores_reference = reference_logits(q, kv, weights, ks, ke)
        expected = torch.empty_like(out)
        topk_into(scores_reference, ks, ke, expected)
        checks = [None] * tp.world_size
        dist.all_gather_object(
            checks, torch.equal(candidate, expected), group=tp.cpu_group
        )
        if not all(checks):
            raise RuntimeError("M/4 changes top-k indices/order on at least one rank")
    out.copy_(candidate)
    return True

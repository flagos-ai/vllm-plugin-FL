# SPDX-License-Identifier: Apache-2.0
"""M/2 prefill row ownership and full-logits all-gather."""

import datetime
import torch
import torch.distributed as dist
from vllm_fl.dispatch.backends.vendor.metax.impl.dsv4.indexer.deep_gemm import (
    bf16_mqa_logits as _reference,
)

from vllm_fl import dsv4_config as _dsv4
from vllm_fl.dispatch.backends.vendor.metax.impl.dsv4.indexer.row_score import (
    bf16_mqa_logits_rowshard_kernel,
)

_PAIR = None


def _tp_info():
    from vllm.distributed import get_tp_group

    tp = get_tp_group()
    return tp.world_size, list(tp.ranks), tp.rank_in_group


def _pair():
    global _PAIR
    if _PAIR is None:
        size, ranks, local_rank = _tp_info()
        if size != 8 or dist.get_world_size() != 8:
            raise RuntimeError("Query-row sharding requires single-node TP8/DP1/PP1")
        for i in range(0, 8, 2):
            pg = dist.new_group(
                ranks=ranks[i : i + 2],
                backend="nccl",
                timeout=datetime.timedelta(seconds=180),
            )
            if local_rank in (i, i + 1):
                _PAIR = (pg, local_rank % 2)
    return _PAIR


def bf16_mqa_logits_prefill(q, kv, weights, ks, ke):
    (m, h, d) = q.shape
    n = kv.shape[0]
    eligible = (
        m >= 2048
        and n >= 1024
        and (h == 64)
        and (d == 128)
        and (q.dtype == torch.bfloat16)
        and (kv.dtype == torch.bfloat16)
        and (weights.dtype == torch.float32)
        and (ks.dtype == torch.int32)
        and (ke.dtype == torch.int32)
        and q.is_contiguous()
        and kv.is_contiguous()
        and weights.is_contiguous()
        and ks.is_contiguous()
        and ke.is_contiguous()
        and (weights.shape == (m, h))
        and (kv.shape == (n, d))
        and (ks.numel() == m)
        and (ke.numel() == m)
    )
    if not _dsv4.current.indexer_row_sharding or not eligible:
        return _reference(q, kv, weights, ks, ke)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "rowshard eligible prefill unexpectedly inside CUDA graph capture"
        )
    (group, rank) = _pair()
    chunk = (m + 127) // 128 * 64
    lo = rank * chunk
    hi = min(m, lo + chunk)
    qq = q[lo:hi]
    ww = weights[lo:hi]
    kk = ks.reshape(-1)[lo:hi]
    ee = ke.reshape(-1)[lo:hi]
    local = torch.zeros((chunk, n), device=q.device, dtype=torch.float32)
    result = torch.empty((2 * chunk, n), device=q.device, dtype=torch.float32)
    bf16_mqa_logits_rowshard_kernel[(hi - lo + 63) // 64, (n + 63) // 64](
        qq,
        kv,
        ww,
        kk,
        ee,
        local,
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
        local.stride(0),
        local.stride(1),
        True,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=4,
        pipeline="basic",
        scenario="",
    )
    dist.all_gather_into_tensor(result, local, group=group)
    output = result[:m]
    return output

"""Complete-query-row sharding: local exact top-k, then gather indices."""

import datetime
import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm_fl import dsv4_config as _dsv4
from vllm_fl.dispatch.backends.vendor.metax.impl.dsv4.indexer.query_score import (
    mqa_hoist_skip_kernel,
)

_GROUPS = {}


def _group(p):
    tp = get_tp_group()
    assert tp.world_size == 8 and dist.get_world_size() == 8
    if p == 8:
        return (tp.device_group, tp.rank_in_group)
    if p not in _GROUPS:
        own = None
        for start in (0, 4):
            group = dist.new_group(
                ranks=tp.ranks[start : start + 4],
                backend="nccl",
                timeout=datetime.timedelta(seconds=180),
            )
            if start <= tp.rank_in_group < start + 4:
                own = (group, tp.rank_in_group - start)
        assert own is not None
        _GROUPS[p] = own
    return _GROUPS[p]


def try_prefill_query_sharding(q, kv, weights, ks, ke, out, topk_tokens, topk_into):
    if not _dsv4.current.indexer_row_sharding:
        return False
    (m, h, d) = q.shape
    n = kv.shape[0]
    tp = get_tp_group()
    eligible = (
        tp.world_size == 8
        and m >= 2048
        and (n >= _dsv4.current.indexer_min_keys)
        and (h == 64)
        and (d == 128)
        and (topk_tokens == 512)
        and (q.dtype == torch.bfloat16)
        and (kv.dtype == torch.bfloat16)
        and (weights.dtype == torch.float32)
        and (ks.dtype == torch.int32)
        and (ke.dtype == torch.int32)
        and (out.dtype == torch.int32)
        and all((x.is_contiguous() for x in (q, kv, weights, ks, ke, out)))
        and (weights.shape == (m, h))
        and (kv.shape == (n, d))
        and (ks.numel() == m)
        and (ke.numel() == m)
        and (out.shape == (m, topk_tokens))
    )
    p = _dsv4.current.indexer_query_shards
    if not eligible:
        return False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("INDEXER_M84 eligible prefill capture")
    (group, owner) = _group(p)
    chunk = (m + p - 1) // p
    lo = owner * chunk
    hi = min(m, lo + chunk)
    assert hi - lo > 1
    pitch = (n + 63) // 64 * 64
    qq = q[lo:hi]
    ww = weights[lo:hi]
    kk = ks.reshape(-1)[lo:hi]
    ee = ke.reshape(-1)[lo:hi]
    storage = torch.zeros((chunk, pitch), device=q.device, dtype=torch.float32)
    scores = storage[:, :n]
    mqa_hoist_skip_kernel[(hi - lo + 63) // 64, (n + 63) // 64](
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
    ids = torch.full((chunk, topk_tokens), -1, device=q.device, dtype=torch.int32)
    topk_into(scores[: hi - lo], kk, ee, ids[: hi - lo])
    gathered = torch.empty((p * chunk, topk_tokens), device=q.device, dtype=torch.int32)
    dist.all_gather_into_tensor(gathered, ids, group=group)
    candidate = gathered[:m]
    out.copy_(candidate)
    return True

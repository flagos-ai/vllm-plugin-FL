# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Static TP/shape selection, owned by FL; local kernels receive explicit metadata."""

import os
import weakref

import torch

from . import attention_launch as kernels, attention_policy as policy
from .base_index import index_score as oldi
from .base_sparse import sparse_attn as oldp, sparse_decode as oldd
from .native_index import (
    minimax_m3_index_decode as native_index_decode,
    minimax_m3_index_topk as index_topk,
)
from vllm_fl.minimax_m3_config import load

_TP_OK = None
_STATE = None
_OWNER_MAP = {}
_OWNER_CONTEXT = None
_STATS = {
    "index_q1": 0,
    "index_prefill": 0,
    "sparse_prefill": 0,
    "sparse_decode": 0,
    "fallback": 0,
    "prefill_consumer_rejected": 0,
}
_HITS = {}


def _tp16():
    global _TP_OK
    if _TP_OK is None:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

        try:
            _TP_OK = get_tensor_model_parallel_world_size() == 16
        except AssertionError:
            return False
    return _TP_OK


def _same_device(q, *xs):
    return q.is_cuda and all(x.device == q.device for x in xs)


def _int_vector(x):
    return x.dtype == torch.int32 and x.ndim == 1 and (x.stride(0) == 1)


def _bt(x):
    return x.dtype == torch.int32 and x.ndim == 2 and (x.stride(1) == 1)


def _hit(kind, shape):
    _STATS[kind] += 1
    key = kind + ":" + str(shape)
    if key not in _HITS:
        _HITS[key] = 0
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else None
        print(
            "M3_LONG_ATTENTION_HIT rank="
            + str(rank)
            + " pid="
            + str(os.getpid())
            + " "
            + key
            + " capture="
            + str(torch.cuda.is_current_stream_capturing()),
            flush=True,
        )
    _HITS[key] += 1


def _prefill_consumer_ok(index_cache):
    global _OWNER_MAP, _OWNER_CONTEXT
    st = _STATE
    if st is None:
        return False
    caller = st["caller"]
    if caller.minimax_m3_sparse_attn is not sparse_attn:
        return False
    if (
        st["model_cls"]._run_attention is not st["model_run"]
        or st["impl_cls"].forward is not st["impl_forward"]
    ):
        return False
    from vllm.forward_context import get_forward_context

    try:
        layers = get_forward_context().no_compile_layers
    except (AssertionError, AttributeError):
        return False
    ptr = index_cache.data_ptr()

    def rebuild():
        global _OWNER_MAP, _OWNER_CONTEXT
        _OWNER_MAP = {}
        _OWNER_CONTEXT = id(layers)
        for layer in layers.values():
            if not isinstance(layer, st["model_cls"]):
                continue
            try:
                ik = layer.indexer.impl.index_cache.kv_cache
                if isinstance(ik, torch.Tensor) and ik.numel():
                    _OWNER_MAP[ik.data_ptr()] = weakref.ref(layer)
            except AttributeError:
                continue

    ref = _OWNER_MAP.get(ptr) if id(layers) == _OWNER_CONTEXT else None
    layer = ref() if ref is not None else None
    if layer is None:
        rebuild()
        ref = _OWNER_MAP.get(ptr)
        layer = ref() if ref else None
    if layer is None:
        return False
    try:
        ii = layer.indexer.impl
        kv = layer.kv_cache
        return (
            type(layer.impl) is st["impl_cls"]
            and type(ii) is st["index_impl_cls"]
            and (ii.index_cache.kv_cache.data_ptr() == ptr)
            and (ii.topk_blocks == 16)
            and (ii.block_size == 128)
            and (ii.init_blocks == 0)
            and (ii.local_blocks == 1)
            and (layer.num_heads == 4)
            and (layer.num_kv_heads == 1)
            and (layer.head_dim == 128)
            and (layer.qkv_proj.weight.dtype == torch.bfloat16)
            and (kv.dtype == torch.bfloat16)
            and (kv.ndim == 5)
            and (tuple(kv.shape[1:]) == (2, 128, 1, 128))
            and (kv.device == index_cache.device)
        )
    except AttributeError:
        return False


def index_score(q, kv, bt, cu, lens, prefix, max_query_len, max_seq_len, num_kv_heads):
    (maxq, maxseq, nkv) = (max_query_len, max_seq_len, num_kv_heads)
    ok = (
        _tp16()
        and q.ndim == 3
        and (tuple(q.shape[1:]) == (1, 128))
        and (nkv == 1)
        and (q.dtype == kv.dtype == torch.bfloat16)
        and (kv.ndim == 3)
        and (tuple(kv.shape[1:]) == (128, 128))
        and _bt(bt)
        and all(_int_vector(x) for x in [cu, lens, prefix])
        and (cu.numel() == lens.numel() + 1)
        and (lens.numel() == prefix.numel())
        and (bt.shape[0] >= lens.numel())
        and _same_device(q, kv, bt, cu, lens, prefix)
    )
    cfg = policy.choose_index(q.shape[0], lens.numel(), maxq, maxseq) if ok else None
    if cfg is not None and cfg["mode"] == "dot" and (not _prefill_consumer_ok(kv)):
        _STATS["prefill_consumer_rejected"] += 1
        cfg = None
    if cfg is None:
        _STATS["fallback"] += 1
        return oldi(q, kv, bt, cu, lens, prefix, maxq, maxseq, nkv)
    _hit("index_q1" if maxq == 1 else "index_prefill", (q.shape[0], maxq, maxseq))
    return kernels.index_score(q, kv, bt, cu, lens, prefix, maxq, maxseq, nkv, cfg)


def sparse_ok(q, kv, ids, bt, lens, nkv, out):
    return (
        _tp16()
        and q.ndim == 3
        and (tuple(q.shape[1:]) == (4, 128))
        and (nkv == 1)
        and (q.dtype == kv.dtype == out.dtype == torch.bfloat16)
        and (kv.ndim == 5)
        and (tuple(kv.shape[1:]) == (2, 128, 1, 128))
        and (out.shape == q.shape)
        and (ids.dtype == torch.int32)
        and (tuple(ids.shape) == (1, q.shape[0], 16))
        and _bt(bt)
        and _int_vector(lens)
        and _same_device(q, kv, ids, bt, lens, out)
    )


def sparse_attn(
    q,
    kv_cache,
    topk_idx,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
):
    (kv, ids, bt, cu, lens, prefix, maxq, nkv, scale, out) = (
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
    )
    ok = (
        sparse_ok(q, kv, ids, bt, lens, nkv, out)
        and _int_vector(cu)
        and _int_vector(prefix)
        and (cu.numel() == lens.numel() + 1)
        and (prefix.numel() == lens.numel())
        and (bt.shape[0] >= lens.numel())
        and _same_device(q, cu, prefix)
    )
    cfg = policy.choose_prefill(q.shape[0], maxq) if ok else None
    if cfg is None:
        _STATS["fallback"] += 1
        return oldp(q, kv, ids, bt, cu, lens, prefix, maxq, nkv, scale, out)
    _hit("sparse_prefill", (q.shape[0], maxq))
    kernels.prefill(q, kv, ids, bt, cu, lens, prefix, maxq, nkv, scale, out, cfg)


def sparse_decode(
    q,
    kv_cache,
    topk_idx,
    block_table,
    seq_lens,
    num_kv_heads,
    sm_scale,
    output,
    decode_query_len,
):
    (kv, ids, bt, lens, nkv, scale, out) = (
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
    )
    ok = (
        sparse_ok(q, kv, ids, bt, lens, nkv, out)
        and decode_query_len == 1
        and (q.shape[0] == lens.numel())
        and (bt.shape[0] >= lens.numel())
    )
    cfg = policy.choose_decode(q.shape[0]) if ok else None
    if cfg is None:
        _STATS["fallback"] += 1
        return oldd(q, kv, ids, bt, lens, nkv, scale, out, decode_query_len)
    _hit("sparse_decode", (q.shape[0], cfg["page_parts"]))
    kernels.decode(q, kv, ids, bt, lens, nkv, scale, out, decode_query_len, cfg)


def initialize_consumer_contract():
    """Capture the approved caller identities once; kernels never inspect this context."""
    global _STATE
    from vllm_fl.models.minimax_m3.bindings import install

    install()
    from vllm.models.minimax_m3.common import indexer, sparse_attention as caller

    from vllm_fl.models.minimax_m3.model import MiniMaxM3SparseAttention

    wanted = dict(
        caller=caller,
        model_cls=MiniMaxM3SparseAttention,
        model_run=MiniMaxM3SparseAttention._run_attention,
        impl_cls=caller.MiniMaxM3SparseTritonImpl,
        impl_forward=caller.MiniMaxM3SparseTritonImpl.forward,
        index_impl_cls=indexer.MiniMaxM3IndexerTritonImpl,
    )
    if _STATE is not None and wanted != _STATE:
        raise RuntimeError("Attention consumer changed after initialization")
    assert caller.minimax_m3_sparse_attn is sparse_attn
    _STATE = wanted


def snapshot():
    return dict(
        counts=dict(_STATS),
        hits=dict(_HITS),
        consumer_initialized=_STATE is not None,
        scope="Python eager/capture calls; graph replays do not increment",
    )


def index_decode(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk,
    init_blocks,
    local_blocks,
    num_kv_heads,
    decode_query_len,
    max_decode_query_len,
    out=None,
):
    if load().framework.exact_decode_fallback and decode_query_len == 1:
        cu = torch.arange(seq_lens.shape[0] + 1, device=idx_q.device, dtype=torch.int32)
        prefix = seq_lens - 1
        score = index_score(
            idx_q,
            index_kv_cache,
            block_table,
            cu,
            seq_lens,
            prefix,
            1,
            max_seq_len,
            num_kv_heads,
        )
        return index_topk(
            score, cu, prefix, 1, topk, init_blocks, local_blocks, out=out
        )

    return native_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        out=out,
    )

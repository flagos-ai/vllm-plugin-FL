# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import json

import torch

from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

from . import index_policy, index_score as kernels
from .native_index import minimax_m3_index_score as original

_TP_OK = None
COUNTS = {"optimized": 0, "fallback": 0}
HITS = {}


def index_score(q, kv, bt, cu, lens, prefix, max_query_len, max_seq_len, num_kv_heads):
    global _TP_OK
    if _TP_OK is None:
        try:
            _TP_OK = get_tensor_model_parallel_world_size() == 16
        except AssertionError:
            _TP_OK = None
    ok = (
        _TP_OK is True
        and q.ndim == 3
        and (q.shape[1:] == (1, 128))
        and (num_kv_heads == 1)
        and (q.dtype == kv.dtype == torch.bfloat16)
        and (kv.ndim == 3)
        and (kv.shape[1:] == (128, 128))
        and (bt.dtype == cu.dtype == lens.dtype == prefix.dtype == torch.int32)
        and (
            q.device
            == kv.device
            == bt.device
            == cu.device
            == lens.device
            == prefix.device
        )
        and (cu.numel() == lens.numel() + 1)
        and (lens.numel() == prefix.numel())
        and (q.shape[0] > 0)
        and (max_query_len > 0)
        and (max_seq_len > 0)
    )
    if not ok:
        COUNTS["fallback"] += 1
        return original(
            q, kv, bt, cu, lens, prefix, max_query_len, max_seq_len, num_kv_heads
        )
    cfg = index_policy.choose(q.shape[0], max_query_len, max_seq_len)
    key = json.dumps(
        dict(
            total_q=q.shape[0],
            max_query=max_query_len,
            max_seq=max_seq_len,
            batch=lens.numel(),
            config=cfg,
        ),
        sort_keys=True,
    )
    if key not in HITS:
        print("M3_INDEX_OPT_HIT " + key, flush=True)
    HITS[key] = HITS.get(key, 0) + 1
    COUNTS["optimized"] += 1
    return kernels.score(
        q, kv, bt, cu, lens, prefix, max_query_len, max_seq_len, num_kv_heads, **cfg
    )

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch

from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

from . import index_policy
from .operators import index_score as score, native_index_score as original

_TP_OK = None


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
        return original(
            q, kv, bt, cu, lens, prefix, max_query_len, max_seq_len, num_kv_heads
        )
    cfg = index_policy.choose(q.shape[0], max_query_len, max_seq_len)
    return score(
        q, kv, bt, cu, lens, prefix, max_query_len, max_seq_len, num_kv_heads, cfg
    )

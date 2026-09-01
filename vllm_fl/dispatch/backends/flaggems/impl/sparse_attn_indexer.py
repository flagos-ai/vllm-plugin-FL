# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems wrappers for sparse indexer helper ops."""

from __future__ import annotations

import torch
from flag_gems.runtime import torch_device_fn


def _native(op_name, *args, **kwargs):
    from vllm_fl.dispatch.backends.reference.impl import sparse_attn_indexer

    fn = getattr(sparse_attn_indexer, f"{op_name}_torch")
    tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if tensor is None:
        tensor = next(
            (value for value in kwargs.values() if isinstance(value, torch.Tensor)),
            None,
        )
    if tensor is None:
        return fn(*args, **kwargs)
    with torch_device_fn.device(tensor.device):
        return fn(*args, **kwargs)


def indexer_k_quant_and_cache_flaggems(*args, **kwargs):
    return _native("indexer_k_quant_and_cache", *args, **kwargs)


def cp_gather_indexer_k_quant_cache_flaggems(*args, **kwargs):
    return _native("cp_gather_indexer_k_quant_cache", *args, **kwargs)


def top_k_per_row_prefill_flaggems(*args, **kwargs):
    return _native("top_k_per_row_prefill", *args, **kwargs)


def top_k_per_row_decode_flaggems(*args, **kwargs):
    return _native("top_k_per_row_decode", *args, **kwargs)


def pack_seq_triton_flaggems(*args, **kwargs):
    return _native("pack_seq_triton", *args, **kwargs)


def unpack_seq_triton_flaggems(*args, **kwargs):
    return _native("unpack_seq_triton", *args, **kwargs)

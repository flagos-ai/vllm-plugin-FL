# Copyright (c) 2026 BAAI. All rights reserved.

"""CUDA vendor implementations for sparse indexer helper ops."""

from __future__ import annotations


def _native(op_name, *args, **kwargs):
    from vllm_fl.dispatch.backends.reference.impl import sparse_attn_indexer

    fn = getattr(sparse_attn_indexer, f"{op_name}_torch")
    return fn(*args, **kwargs)


def indexer_k_quant_and_cache_cuda(*args, **kwargs):
    return _native("indexer_k_quant_and_cache", *args, **kwargs)


def cp_gather_indexer_k_quant_cache_cuda(*args, **kwargs):
    return _native("cp_gather_indexer_k_quant_cache", *args, **kwargs)


def top_k_per_row_prefill_cuda(*args, **kwargs):
    return _native("top_k_per_row_prefill", *args, **kwargs)


def top_k_per_row_decode_cuda(*args, **kwargs):
    return _native("top_k_per_row_decode", *args, **kwargs)


def pack_seq_triton_cuda(*args, **kwargs):
    return _native("pack_seq_triton", *args, **kwargs)


def unpack_seq_triton_cuda(*args, **kwargs):
    return _native("unpack_seq_triton", *args, **kwargs)

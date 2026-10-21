# Copyright (c) 2026 BAAI. All rights reserved.

"""vLLM-native fallback implementations for sparse indexer helper ops."""

from __future__ import annotations


def indexer_k_quant_and_cache_torch(*args, **kwargs):
    from vllm import _custom_ops as ops

    return ops.indexer_k_quant_and_cache(*args, **kwargs)


def cp_gather_indexer_k_quant_cache_torch(*args, **kwargs):
    from vllm import _custom_ops as ops

    return ops.cp_gather_indexer_k_quant_cache(*args, **kwargs)


def top_k_per_row_prefill_torch(*args, **kwargs):
    from vllm import _custom_ops as ops

    return ops.top_k_per_row_prefill(*args, **kwargs)


def top_k_per_row_decode_torch(*args, **kwargs):
    from vllm import _custom_ops as ops

    return ops.top_k_per_row_decode(*args, **kwargs)


def pack_seq_triton_torch(*args, **kwargs):
    from vllm.v1.attention.ops.common import pack_seq_triton

    return pack_seq_triton(*args, **kwargs)


def unpack_seq_triton_torch(*args, **kwargs):
    from vllm.v1.attention.ops.common import unpack_seq_triton

    return unpack_seq_triton(*args, **kwargs)

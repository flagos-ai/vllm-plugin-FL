# SPDX-License-Identifier: Apache-2.0
"""Bounded KDA library calls with framework-owned request metadata."""
from .indexer_backend import INDEXER_BACKEND


def fused_safe_kda_gate(*args, **kwargs):
    return INDEXER_BACKEND._kpool("safe_kda_gate", *args, **kwargs)


def fused_recurrent_kda(*args, **kwargs):
    return INDEXER_BACKEND._kpool("fused_recurrent_kda", *args, **kwargs)


def chunk_kda_with_safe_gate(*args, **kwargs):
    cu_seqlens = kwargs.get("cu_seqlens")
    if cu_seqlens is not None:
        from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
            prepare_chunk_offsets,
        )
        from vllm.model_executor.layers.fla.ops.kda import prepare_chunk_indices

        kwargs["chunk_indices"] = prepare_chunk_indices(cu_seqlens, 64)
        kwargs["chunk_offsets"] = prepare_chunk_offsets(cu_seqlens, 64)
    return INDEXER_BACKEND._kpool("chunk_kda_with_safe_gate", *args, **kwargs)

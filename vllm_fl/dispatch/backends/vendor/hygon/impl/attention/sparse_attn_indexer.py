# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon operator adapters for vLLM's native ROCm sparse indexer.

The sparse-indexer control flow stays in
``vllm.model_executor.layers.sparse_attn_indexer`` and cache
quantization/gather reuses the Triton kernels already shipped in
``vllm.v1.attention.ops.rocm_aiter_mla_sparse``.  This module only replaces the
CUDA-extension leaves that are unavailable on Hygon with the LightOp APIs used
by the Hygon vLLM implementation, retaining vLLM's PyTorch references as
correctness fallbacks.
"""

from __future__ import annotations

import functools
import logging
from types import SimpleNamespace
from typing import Any

import torch

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_lightop_attention() -> Any | None:
    """Resolve both the current and legacy LightOp attention APIs."""
    try:
        from lightop import attention

        return attention
    except (ImportError, AttributeError):
        try:
            from lightop import gemmopt, mqa_logits, op

            return SimpleNamespace(
                mqa_logits=mqa_logits,
                paged_mqa_logits=gemmopt.paged_mqa_logits,
                top_k_per_row_prefill=op.top_k_per_row_prefill,
                top_k_per_row_decode=op.top_k_per_row_decode,
            )
        except (ImportError, AttributeError):
            return None


def fp8_mqa_logits_hygon(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Run the Hygon LightOp prefill MQA kernel."""
    attention = _get_lightop_attention()
    if attention is None:
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            fp8_mqa_logits_torch,
        )

        logger.warning(
            "LightOp attention is unavailable; using vLLM's PyTorch FP8 "
            "prefill MQA reference."
        )
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

    k_fp8, scale = kv
    return attention.mqa_logits(
        q,
        k_fp8,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        scale,
    )


def fp8_paged_mqa_logits_hygon(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    max_model_len: int,
) -> torch.Tensor:
    """Run the Hygon LightOp decode MQA kernel."""
    del schedule_metadata  # LightOp does not consume DeepGEMM scheduling data.
    attention = _get_lightop_attention()
    if attention is None:
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            fp8_paged_mqa_logits_torch,
        )

        logger.warning(
            "LightOp attention is unavailable; using vLLM's PyTorch FP8 "
            "paged MQA reference."
        )
        return fp8_paged_mqa_logits_torch(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )

    return attention.paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        None,
        max_model_len,
        False,
    )


def _topk_indices_torch(
    logits: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference top-k matching vLLM's sparse-indexer output contract."""
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.topk(logits, k=k, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1),
        indices,
    )
    if row_starts is not None:
        starts = row_starts.to(device=indices.device, dtype=torch.int32).view(-1, 1)
        indices = torch.where(indices < 0, indices, indices - starts)
    if k == topk_tokens:
        return indices

    padded = torch.full(
        (logits.shape[0], topk_tokens),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    padded[:, :k] = indices
    return padded


def top_k_per_row_prefill_hygon_out(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """LightOp-compatible replacement for ``_C::top_k_per_row_prefill``."""
    del stride0, stride1
    attention = _get_lightop_attention()
    if attention is None:
        topk_indices.copy_(
            _topk_indices_torch(logits, topk_tokens, row_starts)[:num_rows]
        )
        return

    if logits.ndim != 2:
        raise RuntimeError(f"Prefill top-k expects 2D logits, got {logits.shape}")
    if logits.shape[0] != num_rows:
        if logits.shape[1] == num_rows:
            logits = logits.transpose(0, 1)
        else:
            raise RuntimeError(
                "Prefill top-k logits/query row mismatch: "
                f"logits={tuple(logits.shape)}, num_rows={num_rows}"
            )

    logits = logits.contiguous()
    max_seq_len = logits.shape[1]
    starts = (
        row_starts.to(device=logits.device, dtype=torch.int32)
        .reshape(-1)[:num_rows]
        .clamp(0, max_seq_len)
    )
    ends = (
        row_ends.to(device=logits.device, dtype=torch.int32)
        .reshape(-1)[:num_rows]
        .clamp(0, max_seq_len)
    )
    ends = torch.maximum(ends, starts)
    output = (
        topk_indices
        if topk_indices.is_contiguous()
        else torch.empty_like(topk_indices)
    )
    attention.top_k_per_row_prefill(
        logits,
        starts,
        ends,
        output,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )
    if output is not topk_indices:
        topk_indices.copy_(output)


def _decode_row_ends(
    seq_lens: torch.Tensor,
    next_n: int,
    num_rows: int,
) -> torch.Tensor:
    if seq_lens.ndim == 2:
        return seq_lens.reshape(-1)[:num_rows].clamp(min=0)

    flat = seq_lens.reshape(-1)
    if next_n > 1 and flat.numel() * next_n == num_rows:
        offsets = torch.arange(next_n, dtype=flat.dtype, device=flat.device)
        return (flat.unsqueeze(1) - next_n + offsets + 1).reshape(-1).clamp(min=0)
    return flat[:num_rows].clamp(min=0)


def top_k_per_row_decode_hygon_out(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """LightOp-compatible replacement for ``_C::top_k_per_row_decode``."""
    del stride0, stride1
    attention = _get_lightop_attention()
    if attention is None:
        row_ends = _decode_row_ends(seq_lens, next_n, num_rows)
        masked = logits.clone()
        columns = torch.arange(logits.shape[1], device=logits.device)
        masked.masked_fill_(columns.unsqueeze(0) >= row_ends.unsqueeze(1), float("-inf"))
        topk_indices.copy_(_topk_indices_torch(masked, topk_tokens))
        return

    row_ends = _decode_row_ends(seq_lens, next_n, num_rows)
    attention.top_k_per_row_decode(
        logits,
        1,
        row_ends.to(device=logits.device, dtype=torch.int32),
        topk_indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )


def fp8_fp4_mqa_logits_hygon(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Adapt vLLM's DeepGEMM ABI to the Hygon FP8 LightOp ABI."""
    del clean_logits
    q_values, q_scale = q
    if q_scale is not None:
        raise AssertionError("Hygon sparse indexer does not support FP4 Q")
    return fp8_mqa_logits_hygon(
        q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke
    )


def fp8_fp4_paged_mqa_logits_hygon(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    *,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Adapt vLLM's DeepGEMM paged ABI to the Hygon LightOp ABI."""
    del clean_logits
    q_values, q_scale = q
    if q_scale is not None:
        raise AssertionError("Hygon sparse indexer does not support FP4 Q")
    return fp8_paged_mqa_logits_hygon(
        q_values,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
    )


def indexer_k_quant_and_cache_hygon(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str,
) -> None:
    """Use the Triton cache-insert kernel already shipped by vLLM 0.24.0."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        indexer_k_quant_and_cache_triton,
    )

    indexer_k_quant_and_cache_triton(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt
    )


def cp_gather_indexer_k_quant_cache_hygon(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """Use vLLM's Triton gather kernel with its required token-to-seq map."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_k_quant_cache_triton,
    )

    seq_lengths = (cu_seq_lens[1:] - cu_seq_lens[:-1]).to(torch.int64)
    seq_ids = torch.arange(
        seq_lengths.numel(),
        device=cu_seq_lens.device,
        dtype=torch.int32,
    )
    token_to_seq = torch.repeat_interleave(
        seq_ids,
        seq_lengths,
        output_size=dst_k.shape[0],
    )
    cp_gather_indexer_k_quant_cache_triton(
        kv_cache,
        dst_k,
        dst_scale,
        block_table,
        cu_seq_lens,
        token_to_seq,
    )


def install_sparse_indexer_hygon_ops(sparse_indexer: Any) -> None:
    """Install FL-only Hygon leaves into vLLM's current indexer flow."""
    sparse_indexer.fp8_fp4_mqa_logits = fp8_fp4_mqa_logits_hygon
    sparse_indexer.fp8_fp4_paged_mqa_logits = fp8_fp4_paged_mqa_logits_hygon

    # ``ops`` is vllm._custom_ops.  Its stock functions enter CUDA extension
    # namespaces unavailable on Hygon, so replace only these four public Python
    # entry points in the Hygon process.
    sparse_indexer.ops.indexer_k_quant_and_cache = (
        indexer_k_quant_and_cache_hygon
    )
    sparse_indexer.ops.cp_gather_indexer_k_quant_cache = (
        cp_gather_indexer_k_quant_cache_hygon
    )
    sparse_indexer.ops.top_k_per_row_prefill = top_k_per_row_prefill_hygon_out
    sparse_indexer.ops.top_k_per_row_decode = top_k_per_row_decode_hygon_out

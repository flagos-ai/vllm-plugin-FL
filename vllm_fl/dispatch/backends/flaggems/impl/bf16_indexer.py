# SPDX-License-Identifier: Apache-2.0
"""FlagGems BF16 indexer operator adapters."""

from __future__ import annotations

import torch

from .top_k_per_row import top_k_per_row_decode

_NATIVE_ATEN_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _validate_cache_write(keys, cache, slot_mapping):
    if keys.ndim != 2 or cache.ndim != 3 or slot_mapping.ndim != 1:
        raise ValueError("Expected keys [T,D], cache [B,S,D], slots [T]")
    if keys.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("BF16 indexer cache writer requires bfloat16 tensors")
    if keys.shape[1] != cache.shape[2] or keys.shape[1] == 0:
        raise ValueError("Indexer key/cache head dimensions must match and be positive")
    if keys.shape[0] < slot_mapping.shape[0]:
        raise ValueError("slot_mapping cannot contain more tokens than keys")
    if keys.device != cache.device or keys.device != slot_mapping.device:
        raise ValueError("keys, cache and slot_mapping must share a device")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("slot_mapping must be int32 or int64")
    if slot_mapping.stride(0) != 1:
        raise ValueError("slot_mapping must be contiguous")
    if cache.stride(0) != cache.shape[1] * cache.stride(1):
        raise ValueError("cache blocks must be token-major")


def bf16_indexer_cache_write_flaggems(keys, cache, slot_mapping) -> None:
    from flag_gems import concat_and_cache_mla

    _validate_cache_write(keys, cache, slot_mapping)
    if slot_mapping.numel() == 0:
        return
    if cache.stride(-1) != 1:
        raise NotImplementedError("FlagGems MLA cache requires unit inner stride")
    block = min(keys.shape[-1], 512)
    if block & (block - 1):
        raise NotImplementedError("FlagGems MLA cache tile must be a power of two")
    keys = keys[: slot_mapping.shape[0]].contiguous()
    slots = torch.where(
        slot_mapping < cache.shape[0] * cache.shape[1], slot_mapping, -1
    )
    concat_and_cache_mla(
        keys, keys[:, :0], cache, slots, kv_cache_dtype="auto", scale=keys[:1, :1]
    )


def _bf16_paged_mqa_logits_flaggems(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    *,
    max_context_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Use FlagGems' BF16 paged-MQA logits kernel."""
    from flag_gems.fused import bf16_paged_mqa_logits

    if q.ndim != 4 or cache.ndim != 4 or cache.shape[-2] != 1:
        raise ValueError(
            "BF16 paged-MQA logits expects q [B,N,H,D] and "
            "cache [blocks,S,1,D]"
        )
    if q.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("BF16 paged-MQA logits requires BF16 query and cache")
    if weights.dtype != torch.float32:
        raise ValueError("BF16 paged-MQA logits requires FP32 per-head weights")
    if q.shape[-2] not in (32, 64) or q.shape[-1] != 128:
        raise ValueError("FlagGems BF16 paged-MQA supports H=32/64 and D=128")
    if cache.shape[1] != 64 or cache.shape[-1] != q.shape[-1]:
        raise ValueError(
            "FlagGems BF16 paged-MQA requires block_size=64 and matching head_dim"
        )

    return bf16_paged_mqa_logits(
        q,
        cache,
        weights,
        seq_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_context_len,
        clean_logits=clean_logits,
    )


def _native_candidate_order(candidate_logits: torch.Tensor) -> torch.Tensor:
    """Sort candidates without re-entering FlagGems' global ATen topk.

    The candidate width follows the decode length.  FlagGems specializes its
    generic topk kernel on both ``N`` and ``k``, so dispatching through
    ``torch.topk`` here compiles a new full-sort kernel for every decode step.
    Redispatch to ATen's native implementation for this final ordering only;
    the FlagGems fused candidate-selection kernel remains in use.
    """
    return torch.ops.aten.topk.default.redispatch(
        _NATIVE_ATEN_KEYSET,
        candidate_logits,
        candidate_logits.shape[1],
        -1,
        True,
        True,
    )[1]


def bf16_indexer_decode_flaggems(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    indices: torch.Tensor,
    *,
    next_n: int,
    max_context_len: int,
    clean_logits: bool = False,
) -> None:
    """Produce sorted sparse-attention candidates for BF16 Indexer decode.

    This is the public backend boundary.  Paged-logit materialization, valid
    range top-k, and the final ordering are one implementation concern and must
    not leak into the model adapter.
    """
    logits = _bf16_paged_mqa_logits_flaggems(
        q,
        cache,
        weights,
        seq_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_context_len,
        clean_logits=clean_logits,
    )
    top_k_per_row_decode(logits, seq_lens, indices, next_n=next_n)

    candidate_positions = indices.clamp_min(0).to(torch.int64)
    candidate_logits = torch.gather(logits, 1, candidate_positions)
    candidate_logits.masked_fill_(indices < 0, float("-inf"))
    candidate_order = _native_candidate_order(candidate_logits)
    indices.copy_(torch.gather(indices, 1, candidate_order))


__all__ = [
    "bf16_indexer_cache_write_flaggems",
    "bf16_indexer_decode_flaggems",
]

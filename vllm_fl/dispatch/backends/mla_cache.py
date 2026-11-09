# Copyright (c) 2026 BAAI. All rights reserved.
"""Portable non-quantized MLA cache gather for platforms without the CUDA ABI."""

from __future__ import annotations

import torch


def torch_gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """BF16/FP16 PyTorch fallback for ``_C_cache_ops.gather_and_maybe_dequant_cache``.

    Matches vLLM ``cache_kernels.cu::gather_and_maybe_dequant_cache`` for the
    non-FP8 path used by MLA chunked-prefill / prefix-cache context gather.
    """
    if kv_cache_dtype.startswith("fp8"):
        raise NotImplementedError(
            "MLA: FP8 gather_and_maybe_dequant_cache not supported"
        )

    if num_tokens <= 0:
        return

    # [num_blocks, block_size, D] or [num_blocks, block_size, 1, D]
    src = src_cache.squeeze(2) if src_cache.ndim == 4 else src_cache
    assert src.ndim == 3, f"unexpected MLA cache shape {tuple(src_cache.shape)}"
    block_size = src.size(1)
    hidden = src.size(-1)
    flat = src.reshape(-1, hidden)

    token_ids = torch.arange(num_tokens, device=src.device, dtype=torch.long)
    batch_ids = token_to_seq[:num_tokens].to(dtype=torch.long)
    batch_start = cu_seq_lens[batch_ids].to(dtype=torch.long)
    batch_offset = token_ids - batch_start
    if seq_starts is not None:
        batch_offset = batch_offset + seq_starts[batch_ids].to(dtype=torch.long)

    block_table_ids = torch.div(batch_offset, block_size, rounding_mode="floor")
    slot_ids = torch.remainder(batch_offset, block_size)
    # block_table: [batch, max_blocks]; stride may exceed size(1)
    block_ids = block_table[batch_ids, block_table_ids].to(dtype=torch.long)
    flat_idx = block_ids * block_size + slot_ids
    dst[:num_tokens].copy_(flat[flat_idx])

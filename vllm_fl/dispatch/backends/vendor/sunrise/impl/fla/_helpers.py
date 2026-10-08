# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared helpers for PTPU FLA wrappers."""

from __future__ import annotations

from typing import Optional

import torch


def ensure_chunk_indices(
    cu_seqlens: Optional[torch.Tensor],
    chunk_size: int,
    chunk_indices: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Compute the FLA ``chunk_indices`` table when not supplied.

    PTPU's per-stage kernels accept the same ``chunk_indices`` layout that
    FLA emits via ``prepare_chunk_indices``; we only need to fall back to
    computing it when the caller (typically FLA's own
    ``chunk_gated_delta_rule_fwd`` orchestration) hasn't done so.
    """
    if chunk_indices is not None or cu_seqlens is None:
        return chunk_indices
    # Import lazily so this module can be loaded before the patch fires.
    from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices

    return prepare_chunk_indices(cu_seqlens, chunk_size)


def ensure_chunk_offsets(
    cu_seqlens: Optional[torch.Tensor],
    chunk_size: int,
    chunk_offsets: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Compute FLA's ``chunk_offsets`` table when not supplied."""
    if chunk_offsets is not None or cu_seqlens is None:
        return chunk_offsets
    from vllm.model_executor.layers.fla.ops.index import prepare_chunk_offsets

    return prepare_chunk_offsets(cu_seqlens, chunk_size)

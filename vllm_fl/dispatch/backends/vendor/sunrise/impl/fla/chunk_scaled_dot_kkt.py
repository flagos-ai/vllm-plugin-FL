# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU drop-in for ``fla.ops.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd``."""

from __future__ import annotations

from typing import Optional

import torch

from ._helpers import ensure_chunk_indices


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute ``beta * K @ K^T`` decayed by ``g`` on PTPU.

    The PTPU sgl wrapper expects all of ``(k, beta, g_cumsum, cu_seqlens,
    chunk_indices)``; fall back to FLA when ``beta`` / ``cu_seqlens`` are
    missing (the GDN prefill orchestrator always supplies both, so this is
    a safety net for unusual callers).
    """
    if beta is None or cu_seqlens is None:
        from ...patches.patch_fla_ops import get_orig_chunk_scaled_dot_kkt_fwd

        _fla_chunk_scaled_dot_kkt_fwd = get_orig_chunk_scaled_dot_kkt_fwd()
        if _fla_chunk_scaled_dot_kkt_fwd is None:
            from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
                chunk_scaled_dot_kkt_fwd as _fla_chunk_scaled_dot_kkt_fwd,
            )

        return _fla_chunk_scaled_dot_kkt_fwd(
            k,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            output_dtype=output_dtype,
        )

    chunk_indices = ensure_chunk_indices(cu_seqlens, chunk_size, chunk_indices)

    from torch_ptpu.sgl_kernel import (
        chunk_scaled_dot_kkt as _ptpu_chunk_scaled_dot_kkt,
    )

    A = _ptpu_chunk_scaled_dot_kkt(k, beta, g, cu_seqlens, chunk_indices)
    # FLA pins this to float32 downstream; PTPU should already emit float32,
    # but cast defensively to keep the contract identical.
    if A.dtype != output_dtype:
        A = A.to(output_dtype)
    return A

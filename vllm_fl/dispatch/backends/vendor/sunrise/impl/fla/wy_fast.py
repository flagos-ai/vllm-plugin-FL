# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU drop-in for ``fla.ops.wy_fast.recompute_w_u_fwd``."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._helpers import ensure_chunk_indices


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute (w, u) given (k, v, beta, g_cumsum, A) on PTPU.

    Note the parameter-order subtlety: FLA passes ``(g_cumsum, A)`` while
    PTPU's wrapper takes ``(A, g_cumsum)`` — easy to get wrong.
    """
    # PTPU's recompute_w_u_fwd needs both ``A`` (chunk-tril decomposition)
    # and the chunk index table to navigate variable-length sequences.
    chunk_indices = ensure_chunk_indices(
        cu_seqlens,
        # ``A.shape[-1]`` is the chunk size FLA used (typically 64).
        A.shape[-1],
        chunk_indices,
    )

    from torch_ptpu.sgl_kernel import recompute_w_u_fwd as _ptpu_recompute_w_u_fwd

    return _ptpu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

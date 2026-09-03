# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU drop-in for ``fla.ops.solve_tril.solve_tril``."""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# When PTPU's ``solve_tril`` rejects our index tables for any reason
# (e.g. an unexpected layout requirement we can't reverse-engineer), we
# permanently fall back to FLA Triton in this process to avoid retrying
# on every chunk. A single warning is emitted on first failure.
_PTPU_SOLVE_TRIL_DISABLED = False


def _fla_fallback(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[torch.Tensor],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # Use the snapshot saved by ``patch_fla_ops.apply_patch``. Re-importing
    # ``vllm.model_executor.layers.fla.ops.solve_tril.solve_tril`` after the
    # patch resolves to this wrapper and recurses.
    from ...patches.patch_fla_ops import get_orig_solve_tril

    _fla_solve_tril = get_orig_solve_tril()
    if _fla_solve_tril is None:
        from vllm.model_executor.layers.fla.ops.solve_tril import (
            solve_tril as _fla_solve_tril,
        )

    return _fla_solve_tril(
        A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=output_dtype,
    )


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Solve ``(I + A)^-1`` for strictly lower-triangular ``A`` on PTPU.

    PTPU's ``solve_tril`` (for ``BT > 16``) is a two-phase algorithm:

    1. **phase 1** — solve each 16×16 sub-block independently. Driven
       by a chunk-index table built at ``chunk_size=16``.
    2. **merge** — recursively combine the 16×16 inverses into the
       full ``BT×BT`` inverse. Driven by a chunk-index table built at
       ``chunk_size=BT`` (typically 64 for GDN), which happens to be
       the same layout FLA passes for its single-phase kernel.

    The ``BT == 16`` fast path collapses into a single phase and PTPU
    accepts ``chunk_indices_merge=None`` for that case.

    Any unexpected rejection by the C++ kernel triggers a one-shot fall
    back to FLA Triton (process-wide) so correctness is never traded
    for performance.
    """
    global _PTPU_SOLVE_TRIL_DISABLED

    if A.dtype != torch.float32:
        # PTPU strictly requires float32 input; FLA pipeline always
        # supplies float32, but defer if a caller somehow violates it.
        return _fla_fallback(A, cu_seqlens, chunk_indices, output_dtype)

    if _PTPU_SOLVE_TRIL_DISABLED:
        return _fla_fallback(A, cu_seqlens, chunk_indices, output_dtype)

    BT = A.shape[-1]

    # PTPU requires explicit ``chunk_indices_*`` tables whenever
    # ``cu_seqlens`` is provided AND ``BT > 16``. Build them from
    # FLA's helper so the layout matches what the rest of the pipeline
    # expects.
    chunk_indices_phase1 = None
    chunk_indices_merge = None
    if cu_seqlens is not None:
        from vllm.model_executor.layers.fla.ops.index import (
            prepare_chunk_indices,
        )

        if BT > 16:
            chunk_indices_phase1 = prepare_chunk_indices(cu_seqlens, 16)
            chunk_indices_merge = (
                chunk_indices
                if chunk_indices is not None
                else prepare_chunk_indices(cu_seqlens, BT)
            )
        else:
            # BT == 16: single-phase path, only phase1 indices needed.
            chunk_indices_phase1 = (
                chunk_indices
                if chunk_indices is not None
                else prepare_chunk_indices(cu_seqlens, BT)
            )

    from torch_ptpu.sgl_kernel import solve_tril as _ptpu_solve_tril

    try:
        return _ptpu_solve_tril(
            A,
            cu_seqlens=cu_seqlens,
            output_dtype=output_dtype,
            chunk_indices_phase1=chunk_indices_phase1,
            chunk_indices_merge=chunk_indices_merge,
        )
    except (RuntimeError, ValueError) as exc:
        # One-shot fallback: PTPU rejected our index layout. Log once
        # with the relevant shapes so we can iterate offline, then
        # permanently route this op through FLA Triton.
        _PTPU_SOLVE_TRIL_DISABLED = True
        logger.warning(
            "PTPU solve_tril rejected our index tables (BT=%d, "
            "cu_seqlens=%s, phase1=%s, merge=%s): %s. Falling back to "
            "FLA Triton for solve_tril for the rest of this process.",
            BT,
            None if cu_seqlens is None else tuple(cu_seqlens.shape),
            None if chunk_indices_phase1 is None else tuple(chunk_indices_phase1.shape),
            None if chunk_indices_merge is None else tuple(chunk_indices_merge.shape),
            exc,
        )
        return _fla_fallback(A, cu_seqlens, chunk_indices, output_dtype)

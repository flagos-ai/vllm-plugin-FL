# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU drop-in for ``vllm.model_executor.layers.fla.ops.cumsum.chunk_local_cumsum``."""

from __future__ import annotations

from typing import Optional

import torch

from ._helpers import ensure_chunk_indices


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    **kwargs,
) -> torch.Tensor:
    """Per-chunk cumulative-sum of the gating tensor on PTPU.

    Mirrors the FLA signature (scalar / vector / head_first variants). PTPU's
    ``chunk_local_cumsum`` always operates ``reverse=False`` / ``head_first=
    False``, which is exactly what GDN prefill emits. For any non-default
    knob we defer to FLA to keep semantics identical and avoid surprising
    out-of-band callers.

    PTPU also requires ``cu_seqlens`` (and a matching ``chunk_indices``
    table); when ``cu_seqlens`` is ``None`` (fixed-length batch) we fall
    back to FLA. The GDN prefill orchestrator always runs varlen, so the
    fallback is purely a safety net.
    """
    if reverse or head_first or cu_seqlens is None:
        from ...patches.patch_fla_ops import get_orig_chunk_local_cumsum

        _fla_chunk_local_cumsum = get_orig_chunk_local_cumsum()
        if _fla_chunk_local_cumsum is None:
            from vllm.model_executor.layers.fla.ops.cumsum import (
                chunk_local_cumsum as _fla_chunk_local_cumsum,
            )

        return _fla_chunk_local_cumsum(
            g,
            chunk_size,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            head_first=head_first,
            output_dtype=output_dtype,
            **kwargs,
        )

    chunk_indices = ensure_chunk_indices(cu_seqlens, chunk_size, chunk_indices)

    from torch_ptpu.sgl_kernel import (
        chunk_local_cumsum as _ptpu_chunk_local_cumsum,
    )

    return _ptpu_chunk_local_cumsum(g, cu_seqlens, chunk_indices, output_dtype)

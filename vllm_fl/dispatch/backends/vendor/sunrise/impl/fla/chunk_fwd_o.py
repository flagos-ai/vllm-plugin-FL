# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU drop-in for ``fla.ops.chunk_o.chunk_fwd_o``."""

from __future__ import annotations

from typing import Optional

import torch

from ._helpers import ensure_chunk_indices


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute the chunked attention output ``o`` on PTPU.

    The PTPU sgl wrapper writes into a pre-allocated ``o`` and takes a
    ``B_times_H`` hint for grid sizing. FLA, by contrast, allocates ``o``
    internally and returns it; we replicate the FLA contract here.

    Falls back to FLA when ``cu_seqlens`` is None (fixed-length / batched
    layout), since PTPU's variant requires both ``cu_seqlens`` and
    ``chunk_indices`` to be populated.
    """
    if cu_seqlens is None:
        from ...patches.patch_fla_ops import get_orig_chunk_fwd_o

        _fla_chunk_fwd_o = get_orig_chunk_fwd_o()
        if _fla_chunk_fwd_o is None:
            from vllm.model_executor.layers.fla.ops.chunk_o import (
                chunk_fwd_o as _fla_chunk_fwd_o,
            )

        return _fla_chunk_fwd_o(
            q,
            k,
            v,
            h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )

    chunk_indices = ensure_chunk_indices(cu_seqlens, chunk_size, chunk_indices)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)

    # FLA shapes: q [B, T, Hg, K], v [B, T, H, V]; B_times_H == B * H.
    # ``q.shape[0]`` is always 1 in the varlen prefill path, but compute it
    # generically to remain correct for any future caller.
    B = q.shape[0]
    H = v.shape[-2]
    B_times_H = B * H

    from torch_ptpu.sgl_kernel import chunk_fwd_o as _ptpu_chunk_fwd_o

    _ptpu_chunk_fwd_o(q, k, v, h, g, o, cu_seqlens, chunk_indices, scale, B_times_H)
    return o

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
    core_attn_out: Optional[torch.Tensor] = None,
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

        # 0.24 upstream chunk_fwd_o accepts a trailing ``core_attn_out``
        # buffer; only forward it when the resolved FLA impl actually
        # declares the parameter (0.20.2's does not), so this stays
        # version-agnostic across both vLLMs.
        import inspect

        _fla_kwargs = dict(
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )
        if "core_attn_out" in inspect.signature(_fla_chunk_fwd_o).parameters:
            _fla_kwargs["core_attn_out"] = core_attn_out
        return _fla_chunk_fwd_o(q, k, v, h, **_fla_kwargs)

    chunk_indices = ensure_chunk_indices(cu_seqlens, chunk_size, chunk_indices)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # 0.24 GDN passes a pre-allocated ``core_attn_out`` buffer to avoid a
    # fresh alloc per call (upstream internalised the zero-buffer reuse that
    # patch_gdn_core_attn_buf provides on 0.20.2). Honour it when present,
    # matching upstream semantics: o = core_attn_out[:v.numel()].view(*v.shape).
    if core_attn_out is not None:
        assert core_attn_out.numel() >= v.numel()
        o = core_attn_out[: v.numel()].view(*v.shape)
    else:
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

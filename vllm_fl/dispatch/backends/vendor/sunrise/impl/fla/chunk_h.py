# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU drop-in for ``fla.ops.chunk_delta_h.chunk_gated_delta_rule_fwd_h``.

This is the sixth (and previously last-remaining) GDN prefill stage. It
computes the per-chunk hidden state ``h`` and (optionally) the recomputed
``v_new`` and a packed ``final_state`` tensor that the orchestrator hands
back to the SSM cache.

Contract translation
--------------------
FLA's stage returns ``(h, v_new, final_state)``:

* ``h``         : ``(B, NT, H, V, K)`` per-chunk recurrent state, dtype = ``k.dtype``
* ``v_new``     : ``(B, T, H, V)`` (same as ``u``) – recomputed values, or ``None``
* ``final_state``: ``(N, H, V, K)`` float32, or ``None`` when
                  ``output_final_state=False``

PTPU's ``chunk_delta_h_fwd`` writes ``h`` (and optionally ``v_new``) in
place and treats ``initial_state`` as a *single* read-modify-write buffer
that contains the per-sequence prior state on entry and the per-sequence
final state on return. To preserve FLA's "fresh ``final_state`` output"
semantics without mutating the caller's ``initial_state``, we allocate
our own float32 buffer, seed it with the caller's initial state (or zero
when none was supplied), pass it to PTPU as ``initial_state``, and hand
it back as ``final_state``.

State layout transpose (critical correctness note)
--------------------------------------------------
The PTPU SGL ``chunk_delta_h_fwd`` / ``fused_sigmoid_gating_delta_rule_update``
kernels store the recurrent state as ``[K_row, V_col]`` — V is the
inner contiguous dim (``offset = k*V + v``). FLA / vLLM allocate it as
``[V_row, K_col]`` — K is the inner dim (``offset = v*K + k``). When
``head_k_dim == head_v_dim`` (the Qwen3.5-4B case) both layouts share
the same memory size and outer strides, so the kernel does **not**
crash on a layout mismatch — it just produces a transposed result.

To guarantee the ``final_state`` we hand back to vLLM follows the
documented FLA contract (``[N, H, V, K]`` with ``offset = v*K + k``),
we allocate a PTPU-native ``[N, H, K, V]`` working buffer here, copy
the caller's initial state into it with ``transpose(-1, -2)`` first,
let PTPU do its in-place update, then transpose the buffer back into
the FLA-shaped ``final_state`` tensor before returning.

Fallback paths
--------------
* ``gk is not None`` (KDA-style gating)            -> FLA Triton.
* ``cu_seqlens is None`` (fixed-length / batched)  -> FLA Triton.
  vLLM's varlen prefill always supplies ``cu_seqlens``; this branch only
  guards offline test paths and warmup runs that exercise the batched
  shape.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._helpers import ensure_chunk_offsets


def _fla_fallback(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    chunk_size: int,
    save_new_value: bool,
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[torch.Tensor],
    chunk_offsets: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    from ...patches.patch_fla_ops import get_orig_chunk_gated_delta_rule_fwd_h

    _fla_chunk_gated_delta_rule_fwd_h = get_orig_chunk_gated_delta_rule_fwd_h()
    if _fla_chunk_gated_delta_rule_fwd_h is None:
        from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h as _fla_chunk_gated_delta_rule_fwd_h,
        )

    return _fla_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    # PTPU does not surface a ``gk`` path, and ``cu_seqlens`` is mandatory
    # for the varlen kernel — bail to FLA for anything else.
    if gk is not None or cu_seqlens is None:
        return _fla_fallback(
            k, w, u, g, gk, initial_state, output_final_state,
            chunk_size, save_new_value,
            cu_seqlens, chunk_indices, chunk_offsets,
        )

    B, T, Hg, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    BT = chunk_size

    chunk_offsets = ensure_chunk_offsets(cu_seqlens, BT, chunk_offsets)
    # PTPU's ``chunk_delta_h_fwd`` is strict about its index tensor dtype
    # (must be int32). FLA's ``prepare_chunk_offsets`` inherits the dtype
    # of ``cu_seqlens``, which is int64 on vLLM v1's attention metadata,
    # so we cast here. Cheap: this tensor is tiny (one int per chunk).
    if chunk_offsets.dtype != torch.int32:
        chunk_offsets = chunk_offsets.to(torch.int32)

    # NT must match FLA's allocation: ``len(chunk_indices)`` if cu_seqlens
    # is provided. We derive it from ``chunk_offsets[-1]`` to avoid forcing
    # a chunk_indices materialization when the caller didn't pass one.
    NT = int(chunk_offsets[-1].item())
    N = cu_seqlens.numel() - 1

    h = k.new_empty(B, NT, H, V, K)
    v_new = torch.empty_like(u) if save_new_value else None

    # PTPU writes the per-sequence final state back into the same buffer it
    # reads ``initial_state`` from. To match FLA's contract ("return a
    # fresh ``final_state`` tensor, leave the caller's ``initial_state``
    # untouched") we own that buffer here, AND we own the layout: PTPU
    # speaks ``[K, V]`` (V inner) while FLA / vLLM speak ``[V, K]`` (K
    # inner). See the module docstring for the full background.
    #
    # Critical: PTPU's kernel only reads / writes ``initial_state`` when
    # BOTH ``initial_state`` and ``initial_state_indices`` are non-null
    # (see ``chunk_delta_h_fwd.cc`` ``use_initial_state``). When we want
    # the kernel to populate ``final_state`` we MUST pass a slot-id table,
    # otherwise the kernel runs with ``USE_INITIAL_STATE=false`` and
    # ``INPLACE_UPDATE=false`` -> ``ptpu_state_buf`` stays at its
    # zero-initialised contents and we silently return the wrong final
    # state to vLLM. That zero ``final_state`` becomes the seed of the
    # next decode step's recurrent state, breaking GDN-derived context for
    # subsequent tokens. (Manifests dramatically on Qwen3.5-9B with very
    # short prompts; on Qwen3.5-4B the per-chunk ``h`` carries enough
    # local context that the regression was below the smoke-test floor.)
    if output_final_state or initial_state is not None:
        # Working buffer in PTPU's native [N, H, K, V] layout (V inner).
        ptpu_state_buf: Optional[torch.Tensor] = k.new_empty(
            N, H, K, V, dtype=torch.float32
        )
        if initial_state is not None:
            # vLLM hands us [N, H, V, K]. Transpose into PTPU's [N, H, K, V]
            # convention while seeding the buffer.
            ptpu_state_buf.copy_(initial_state.transpose(-1, -2))
        else:
            ptpu_state_buf.zero_()
        # One slot per sequence; layout matches the buffer above.
        initial_state_indices = torch.arange(
            N, dtype=torch.int32, device=k.device
        )
    else:
        ptpu_state_buf = None
        initial_state_indices = None

    from torch_ptpu.sgl_kernel import chunk_delta_h_fwd as _ptpu_chunk_delta_h_fwd

    _ptpu_chunk_delta_h_fwd(
        k=k,
        v=u,
        w=w,
        h=h,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        v_new=v_new,
        initial_state_indices=initial_state_indices,
        initial_state=ptpu_state_buf,
        g=g,
        gk=gk,
    )

    # Convert PTPU's [N, H, K, V] result back into FLA's [N, H, V, K]
    # contract before returning.
    if output_final_state:
        assert ptpu_state_buf is not None
        final_state = ptpu_state_buf.transpose(-1, -2).contiguous()
    else:
        final_state = None
    return h, v_new, final_state

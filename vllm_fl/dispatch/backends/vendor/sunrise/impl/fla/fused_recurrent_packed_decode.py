# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU wrapper for FLA packed-decode GDN; transposes state layout and calls native kernel."""

from __future__ import annotations

from typing import Optional

import torch

from ._gdn_state_transpose import (
    gather_transpose_to_scratch,
    transpose_scatter_to_pool,
)


# Pre-allocated scratch buffers for cudagraph-safe decode.

_DEFAULT_HEADROOM: int = 256

_PTPU_STATE_SCRATCH: "dict[tuple, torch.Tensor]" = {}
_ARANGE_BUF: "dict[tuple, torch.Tensor]" = {}


def _next_capacity(needed: int, current: int) -> int:
    """Doubling-grow capacity, never below ``_DEFAULT_HEADROOM``."""
    target = max(needed, _DEFAULT_HEADROOM)
    if current > 0:
        target = max(target, current * 2)
    return target


def _ensure_state_scratch(
    B: int,
    HV: int,
    K: int,
    V: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Lazy-init/return scratch of shape ``(>=B, HV, K, V)`` in PTPU layout.

    Re-allocates (and drops the previous buffer) only when ``B`` exceeds the
    current capacity. Once allocated, the buffer's ``data_ptr()`` is stable
    -- captured graphs hold this pointer; replays reuse the same memory.
    """
    key = (device, dtype, HV, K, V)
    buf = _PTPU_STATE_SCRATCH.get(key)
    cur = 0 if buf is None else buf.shape[0]
    if buf is None or cur < B:
        new_capacity = _next_capacity(B, cur)
        buf = torch.empty(new_capacity, HV, K, V, device=device, dtype=dtype)
        _PTPU_STATE_SCRATCH[key] = buf
    return buf


def _ensure_arange_buf(
    min_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Lazy-init/return an ``arange(>=min_size)`` buffer.

    Used for both ``cu_seqlens`` (size ``B+1``) and ``local_indices``
    (size ``B``). Both are int32 contiguous integer sequences, so a single
    ``arange`` of ``B+1`` elements covers both via slicing.
    """
    key = (device, dtype)
    buf = _ARANGE_BUF.get(key)
    cur = 0 if buf is None else buf.numel()
    if buf is None or cur < min_size:
        new_capacity = _next_capacity(min_size, cur)
        buf = torch.arange(new_capacity, device=device, dtype=dtype)
        _ARANGE_BUF[key] = buf
    return buf


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PTPU-native re-implementation of FLA's packed-decode GDN fast path.

    Parameters mirror
    ``vllm.model_executor.layers.fla.ops.fused_recurrent_gated_delta_rule_packed_decode``
    1:1. The function writes its result into ``out`` (in place) and into
    ``initial_state`` (via ``ssm_state_indices`` slots), then returns the
    same ``(out, initial_state)`` tuple the upstream wrapper would.
    """
    # Any shape outside what PTPU's fused_sigmoid_gating_delta_rule_update
    # supports falls through to the saved-original FLA Triton callable.
    if initial_state is None or ssm_state_indices is None:
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    if mixed_qkv.ndim != 2:
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    if ssm_state_indices.ndim != 1:
        # PTPU's per-call slot indexing assumes a 1-D ``initial_state_indices``;
        # the packed_decode contract guarantees this but be defensive.
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    B = mixed_qkv.shape[0]
    if initial_state.ndim != 4:
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    HV, V, K = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    qk_dim_total = qkv_dim - HV * V
    if qk_dim_total <= 0 or qk_dim_total % 2 != 0:
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    q_dim = qk_dim_total // 2
    if q_dim % K != 0:
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    H = q_dim // K

    q_flat = mixed_qkv[:, 0:H * K]
    k_flat = mixed_qkv[:, H * K:2 * H * K]
    v_flat = mixed_qkv[:, 2 * H * K:]
    q = q_flat.view(1, B, H, K)
    k = k_flat.view(1, B, H, K)
    v = v_flat.view(1, B, HV, V)

    device = mixed_qkv.device
    state_dtype = initial_state.dtype

    state_scratch_full = _ensure_state_scratch(B, HV, K, V, device, state_dtype)
    ptpu_state_buf = state_scratch_full[:B]  # view, no allocation

    arange_full = _ensure_arange_buf(B + 1, device, torch.int32)
    cu_seqlens = arange_full[:B + 1]      # view
    local_indices = arange_full[:B]        # view

    # Gather/transpose state into PTPU layout scratch.
    gather_transpose_to_scratch(initial_state, ssm_state_indices, ptpu_state_buf)

    from torch_ptpu.sgl_kernel import (
        fused_sigmoid_gating_delta_rule_update as _ptpu_fused_sigmoid_gating,
    )

    o = _ptpu_fused_sigmoid_gating(
        A_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b,
        ptpu_state_buf,
        local_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        is_kda=False,
    )
    # Copy kernel output into caller buffer.
    out.squeeze(1).copy_(o.squeeze(0))

    # Scatter updated state back to vLLM layout.
    transpose_scatter_to_pool(ptpu_state_buf, ssm_state_indices, initial_state)

    return out, initial_state


def _fallback(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    out: torch.Tensor,
    ssm_state_indices: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback to the original FLA Triton packed_decode implementation.

    Uses the snapshot saved by ``patch_fla_ops.apply_patch`` so we don't
    recurse through the module-level rebind that installed this wrapper.
    """
    from ...patches.patch_fla_ops import (
        get_orig_fused_recurrent_gated_delta_rule_packed_decode,
    )

    orig = get_orig_fused_recurrent_gated_delta_rule_packed_decode()
    if orig is None:
        from vllm.model_executor.layers.fla.ops.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode as orig,
        )

    return orig(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

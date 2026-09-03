# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU wrapper for fused_sigmoid_gating_delta_rule_update with state layout transpose."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-step fused GDN decode kernel on PTPU.

    PTPU's variant always writes ``final_state`` in-place into
    ``initial_state_source``; FLA returns it as a separate tensor when
    ``inplace_final_state=False``. The hot path in
    ``GatedDeltaNetAttention._forward_core`` always sets
    ``inplace_final_state=True`` with ``initial_state=ssm_state``, so the
    PTPU contract matches directly: we return ``(o, ssm_state)``.

    The ``num_accepted_tokens`` argument exists only for speculative
    decoding and is not used by Qwen3.5 today; PTPU does not expose an
    equivalent. We fall back to FLA whenever ``num_accepted_tokens`` is
    set or when ``initial_state`` / ``ssm_state_indices`` are absent (the
    paged-decode contract PTPU expects).
    """
    if (
        not inplace_final_state
        or initial_state is None
        or ssm_state_indices is None
        or num_accepted_tokens is not None
    ):
        # Use the saved un-patched FLA Triton original to avoid
        # infinite recursion: the module-level name on
        # ``vllm.model_executor.layers.fla.ops.fused_sigmoid_gating``
        # has been rebound to point to *this* function by
        # ``patch_fla_ops.apply_patch``.
        from ...patches.patch_fla_ops import (
            get_orig_fused_sigmoid_gating_delta_rule_update,
        )

        _fla_fused_sigmoid_gating = (
            get_orig_fused_sigmoid_gating_delta_rule_update()
        )
        if _fla_fused_sigmoid_gating is None:
            from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
                fused_sigmoid_gating_delta_rule_update as _fla_fused_sigmoid_gating,  # noqa: E501
            )

        return _fla_fused_sigmoid_gating(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            beta=beta,
            threshold=threshold,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_kda=is_kda,
        )

    from torch_ptpu.sgl_kernel import (
        fused_sigmoid_gating_delta_rule_update as _ptpu_fused_sigmoid_gating,
    )

    # Layout transpose: vLLM's ``ssm_state`` is laid out [num_blocks, HV, V, K]
    # but PTPU's kernel expects [slot, HV, K, V]. Build a small per-call
    # working buffer that contains only the slots referenced by
    # ``ssm_state_indices``, transposed into PTPU's layout. After the
    # kernel runs we transpose the buffer back and scatter into the
    # caller's ``initial_state`` tensor (= ``ssm_state``) so vLLM keeps
    # its expected on-disk layout.
    #
    # This is also strictly cheaper than transposing the full state pool
    # in/out per call: only B (not num_blocks) slots are touched.
    slot_idx = ssm_state_indices.to(torch.int64)
    state_slots_fla = initial_state.index_select(0, slot_idx)  # [B, HV, V, K]
    ptpu_state_buf = state_slots_fla.transpose(-1, -2).contiguous()  # [B, HV, K, V]
    # The kernel keys slots through ``ssm_state_indices`` as well, so we
    # need to remap indices to ``arange(B)`` to address ``ptpu_state_buf``.
    local_indices = torch.arange(
        slot_idx.numel(), device=slot_idx.device, dtype=ssm_state_indices.dtype
    )

    o = _ptpu_fused_sigmoid_gating(
        A_log,
        a,
        dt_bias,
        beta,
        threshold,
        q,
        k,
        v,
        b,
        ptpu_state_buf,
        local_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        is_kda=is_kda,
    )

    # Scatter updated state slots back into the pool.
    src_back = ptpu_state_buf.transpose(-1, -2).contiguous().to(initial_state.dtype)
    initial_state[slot_idx] = src_back
    return o, initial_state

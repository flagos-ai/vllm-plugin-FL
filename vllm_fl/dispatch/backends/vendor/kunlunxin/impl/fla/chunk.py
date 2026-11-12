# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
#
# 2026 - Modified by Kunlunxin, Inc. All Rights Reserved.

"""
Kunlunxin implementation of chunk_gated_delta_rule.

Contains both the low-level kernel wrapper (chunk_gated_delta_rule_fwd) and
the top-level entry (chunk_gated_delta_rule).

The Kunlunxin native kernel supports use_qk_l2norm_in_kernel natively,
so normalization is delegated to the kernel when enabled, rather than
being done in Python like other backends.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from einops import rearrange

import xtorch_ops


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple:
    """
    Kunlunxin chunked gated delta rule forward pass.

    Args:
        q: queries [B, T, H, K]
        k: keys [B, T, H, K]
        v: values [B, T, H, V]
        g: decay gates (in log space)
        beta: betas
        scale: scale factor
        initial_state: [N, H, V, K] -- the layout of vLLM's paged SSM cache
        output_final_state: whether to output final state
        cu_seqlens: cumulative sequence lengths

    Returns:
        7-tuple: (g, o, A, final_state, w, h, v_new)
        to match the dispatch interface expected by vllm_fl
    """
    input_dtype = q.dtype

    # vLLM's paged SSM cache stores one state block as [H, V, K]
    # (MambaStateShapeCalculator.gated_delta_net_state_shape), while the xtorch
    # chunk kernel follows the FLA convention [N, H, K, V] for h0 / ht
    # (see xtorch_ops.chunk_gated_delta_rule: initial_state/ht are
    # (batch, num_heads, head_dim, v_head_dim)).  Convert on the way in and on
    # the way out; handing the [V, K] cache slice straight to the kernel is
    # silently wrong whenever K == V (Qwen3.6), and it transposes the recurrent
    # state that prefill writes back, which corrupts every decode step after the
    # first token.
    initial_state = initial_state.transpose(2, 3).contiguous()

    # check dtype
    if input_dtype == torch.bfloat16:
        g_input = g.float()
        beta_input = beta.float()
        state_input = initial_state.float()
    else:
        g_input = g
        beta_input = beta
        state_input = initial_state

    cu_seqlens_cpu = cu_seqlens.cpu() if cu_seqlens is not None else None

    # Use -1 for automatic scale (1/sqrt(K)) only when caller does not provide one
    kernel_scale = scale if scale is not None else -1

    final_state = torch.empty_like(state_input)
    o = torch.empty_like(v)
    xtorch_ops.chunk_gated_delta_rule(
        q,
        k,
        v,
        g_input,
        beta_input,
        kernel_scale,
        state_input,
        o,
        final_state,
        cu_seqlens_cpu,
        head_first=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Kernel returns ht as [N, H, K, V]; convert back to the cache layout [N, H, V, K]
    final_state = final_state.transpose(2, 3).contiguous()

    # Return 7-tuple to match the dispatch interface:
    # (g, o, A, final_state, w, h, v_new)
    return g, o, None, final_state, None, None, None


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        # When use_qk_l2norm_in_kernel=True, the native kernel handles L2-norm
        # internally. When False, the caller has already normalized q/k externally.

        # initial_state arrives straight out of vLLM's paged cache, whose
        # per-block shape is [H, V, K]
        # (MambaStateShapeCalculator.gated_delta_net_state_shape).
        # chunk_gated_delta_rule_fwd owns the [V, K] <-> [K, V] conversion for
        # both the incoming state and the returned final state, so no transpose
        # belongs here.
        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: Optional[torch.Tensor] = None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences
            (the layout of vLLM's paged SSM cache).
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, (
        "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
    )

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
            stacklevel=2,
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            stacklevel=2,
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    if core_attn_out is not None:
        output_flat = o.reshape(-1)
        core_attn_out.reshape(-1)[: output_flat.numel()].copy_(output_flat)
    return o, final_state

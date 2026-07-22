# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/sigmoid_gating.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_delta_rule_update_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Recurrent delta-rule decode update: q/k L2 norm + state update + output
    in a single launch. The sigmoid gating (g/beta) is computed outside by
    the AscendC ``npu_fused_gdn_gating`` op — the in-kernel gating section
    of the upstream fused_sigmoid_gating_delta_rule_update kernel is
    miscompiled by the Ascend Triton pipeline in this environment.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV - 1, i_nh % HV
    # The grid carries one dummy group of HV programs up front: the first
    # scheduled program on this Ascend Triton pipeline sporadically
    # produces a corrupted state/output tile (verified on 910B4-1 with the
    # CANN 8.5.0 bishengir pipeline; every other program is exact), so it
    # is made to return immediately and all real sequences are shifted to
    # healthy programs.
    if i_n < 0:
        return
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    p_beta = beta + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V

    # NOTE: the FL ssm state cache is laid out per slot as (HV, V, K)
    # (v-major), as required by the AscendC recurrent_gated_delta_rule
    # kernel, while the upstream kernel this is adapted from assumes the
    # FLA (HV, K, V) layout. The state tile is therefore kept in the
    # native (BV, BK) v-major orientation end to end.
    mask_h_t = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        # if idx >= 0:
        tmp0 = tl.where(idx < 0, 0, idx)
        p_h0 = h0_source + tmp0 * HV * K * V + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
        temp1 = tl.load(p_h0, mask=mask_h_t, other=0).to(tl.float32)
        temp2 = tl.zeros_like(temp1)
        b_h += tl.where(idx < 0, temp2, temp1)

    for i in range(0, T):
        # Load inputs
        b_q = tl.load(p_q + i * H * K, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k + i * H * K, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v + i * HV * V, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g + i * HV).to(tl.float32)
        b_beta = tl.load(p_beta + i * HV).to(tl.float32)

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=1)
        b_v -= tl.sum(b_h * b_k[None, :], 1)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += v[:, None] * k[None, :]
        b_h += b_v[:, None] * b_k[None, :]

        # Compute output: o = sum(h * q, dim=1)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o + i * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h_t)


def fused_recurrent_delta_rule_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor = None,
):
    """
    Fused triton implementation of the recurrent gated delta-rule decode
    update: q/k L2 norm + state update + output in a single kernel launch.

    ``g`` (log-space forget gate) and ``beta`` are precomputed (by the
    AscendC ``npu_fused_gdn_gating`` op) with shape ``[B * T, HV]``.

    Note: ``initial_state_source`` must use the FL per-slot (HV, V, K) state
    layout (v-major), consistent with the AscendC GDN kernels.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    if cu_seqlens is not None:
        # One extra group of dummy programs is prepended in the grid (the
        # kernel returns immediately for them) to absorb the corrupted
        # first scheduled program; no tensor padding needed.
        N += 1

    o = q.new_empty(NK, *v.shape)
    grid = (NK, NV, N * HV)

    if not initial_state_indices.is_contiguous():
        initial_state_indices = initial_state_indices.contiguous()
    if not initial_state_source.is_contiguous():
        initial_state_source = initial_state_source.contiguous()
    if not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()

    fused_recurrent_delta_rule_update_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o

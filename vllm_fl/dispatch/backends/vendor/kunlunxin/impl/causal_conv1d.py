# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
#
# 2026 - Modified by Kunlunxin, Inc. All Rights Reserved.

"""
Kunlunxin implementation of causal_conv1d operators.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from vllm.attention.backends.utils import PAD_SLOT_ID

import xtorch_ops


def causal_conv1d_fn_kunlunxin(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
) -> torch.Tensor:
    """
    Kunlunxin causal conv1d forward (prefill path).

    Args:
        x: input tensor (dim, cu_seqlen) for varlen
        weight: (dim, width)
        bias: (dim,) or None
        activation: "silu", "swish", or None
        conv_states: (num_cache_lines, dim, state_len)
        has_initial_state: (batch,) bool
        cache_indices: (batch,) int
        query_start_loc: (batch+1,) int
        metadata: optional metadata
        pad_slot_id: padding slot id

    Returns:
        output tensor (same shape as x, modified in-place)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None
    query_start_loc_cpu = query_start_loc.cpu().tolist()

    xtorch_ops.causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation in ["silu", "swish"],
        False,          # is_ncw: x is (cu_seqlen, dim), conv_states is (N, state_len, dim)
        query_start_loc_cpu,
        pad_slot_id,
    )

    return x


def causal_conv1d_update_kunlunxin(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
) -> torch.Tensor:
    """
    Kunlunxin causal conv1d update (decode path).

    Args:
        x: input tensor [batch, dim] or [batch, dim, seqlen]
        conv_state: (num_cache_lines, dim, state_len)
        weight: (dim, width)
        bias: (dim,) or None
        activation: "silu", "swish", or None
        conv_state_indices: (batch,) state indices
        num_accepted_tokens: (batch,) for speculative decoding
        query_start_loc: (batch+1,) for varlen
        max_query_len: max query length
        pad_slot_id: padding slot id
        block_idx_last_scheduled_token: (batch,) for APC
        initial_state_idx: (batch,) for APC
        validate_data: whether to validate inputs

    Returns:
        output tensor (same shape as input x)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(
            f"activation must be None, silu, or swish, actual: {activation}"
        )

    activation_val = activation in ["silu", "swish"]

    # check layout
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(1)
    else:
        x = x.transpose(1, 2)
    x = x.contiguous()

    xtorch_ops.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation_val,
        cache_seqlens,
        conv_state_indices,
        intermediate_conv_window,
        False,       # is_ncw
        pad_slot_id,
    )

    # reverse layout
    if unsqueeze:
        x = x.squeeze(1)
    else:
        x = x.transpose(1, 2)

    return x

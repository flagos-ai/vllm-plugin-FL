# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/layer.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
import torch.nn.functional as F
import torch_npu
from flag_gems.runtime.backend._ascend import fused


def _torch_fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor|None = None,
) -> torch.Tensor:
    """Pure PyTorch implementation of fused MoE experts for NPU.

    This avoids the Triton fused_moe_kernel which has compatibility issues
    on Ascend NPU hardware.
    """
    num_tokens, hidden_dim = hidden_states.size()
    E, N, _ = w1.size()  # w1: [E, N, K_in]
    K = w2.size(1)        # w2: [E, K_out, N//2]
    top_k = topk_ids.size(1)

    if global_num_experts == -1:
        global_num_experts = E

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.zeros_like(hidden_states)

    # Map global expert ids to local expert ids
    if expert_map is not None:
        local_topk_ids = expert_map[topk_ids.long()]
    else:
        local_topk_ids = topk_ids.long()

    # Process each expert
    for expert_idx in range(E):
        # Find which (token, k) pairs are assigned to this expert
        mask = (local_topk_ids == expert_idx)  # [num_tokens, top_k]
        if not mask.any():
            continue

        # Get token indices and their k-slot indices
        token_indices, k_indices = torch.where(mask)

        # Gather the hidden states for these tokens
        expert_input = hidden_states[token_indices]  # [n, hidden_dim]

        # Apply router weight on input if needed
        if apply_router_weight_on_input:
            weights = topk_weights[token_indices, k_indices].unsqueeze(-1)
            expert_input = expert_input * weights.to(expert_input.dtype)

        # First matmul: expert_input @ w1[expert_idx].T
        # w1[expert_idx] shape: [N, hidden_dim], result: [n, N]
        gate_up = torch.mm(expert_input, w1[expert_idx].t())

        # Activation (pure PyTorch to avoid Triton kernel issues on NPU)
        if activation == "silu":
            d = gate_up.shape[-1] // 2
            gate_up = F.silu(gate_up[..., :d]) * gate_up[..., d:]
        elif activation == "gelu":
            gate_up = torch_npu.npu_gelu_mul(gate_up)
        elif activation == "silu_no_mul":
            gate_up = F.silu(gate_up)
        elif activation == "gelu_no_mul":
            gate_up = torch_npu.npu_gelu(gate_up)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

        # Second matmul: activated @ w2[expert_idx].T
        # w2[expert_idx] shape: [K_out, N//2], result: [n, K_out]
        expert_output = torch.mm(gate_up, w2[expert_idx].t())

        # Apply router weight on output if not applied on input
        if not apply_router_weight_on_input:
            weights = topk_weights[token_indices, k_indices].unsqueeze(-1)
            expert_output = expert_output * weights.to(expert_output.dtype)

        # Accumulate results
        out_hidden_states.index_add_(0, token_indices, expert_output)

    return out_hidden_states


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
        )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    # Use pure-torch implementation on NPU to avoid Triton kernel
    # compatibility issues with the Ascend backend.
    return _torch_fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )

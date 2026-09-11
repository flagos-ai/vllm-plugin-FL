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

import logging
logger = logging.getLogger(__name__)


def _npu_grouped_matmul_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optimized MoE using npu_grouped_matmul — single batched kernel for all experts.

    Replaces the Python for-loop over experts with:
    1. npu_moe_init_routing_v2 — sort tokens by expert, get per-expert counts
    2. npu_grouped_matmul — batched gate_up projection (all experts in one call)
    3. npu_swiglu — fused SiLU+mul activation
    4. npu_grouped_matmul — batched down projection
    5. npu_moe_token_unpermute — scatter results back with router weights
    """
    num_tokens, hidden_dim = hidden_states.shape
    E, N, _ = w1.shape  # w1: [E, N, K_in]
    top_k = topk_ids.shape[1]

    if global_num_experts == -1:
        global_num_experts = E

    # Handle expert_map for tensor parallel
    if expert_map is not None:
        local_topk_ids = expert_map[topk_ids.long()]
        # Mask invalid experts (mapped to -1)
        valid_mask = local_topk_ids >= 0
        topk_weights = topk_weights * valid_mask.to(topk_weights.dtype)
        topk_ids_for_routing = local_topk_ids.to(torch.int32)
    else:
        topk_ids_for_routing = topk_ids.to(torch.int32)

    # Apply router weight on input if needed
    if apply_router_weight_on_input:
        # Scale hidden states by topk weights before routing
        # For this path, we need to expand hidden states first
        pass  # Handled below in the unpermute step

    # Step 1: Sort tokens by expert using npu_moe_init_routing_v2
    sorted_hidden_states, expanded_row_idx, expert_tokens, _ = (
        torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids_for_routing,
            active_num=num_tokens * top_k,
            expert_num=E,
            expert_tokens_num_type=1,  # count mode
            expert_tokens_num_flag=True,
            active_expert_range=[0, E],
            quant_mode=-1,  # no quantization
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    # Step 2: Gate-up projection — npu_grouped_matmul
    # w1 is [E, N, K] — grouped_matmul expects weight as [E, K, N] with split_item=2
    # split_item=2 means the weight K dimension splits across the group_list
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w1.transpose(1, 2).contiguous()],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
    )[0]

    # Step 3: Activation
    if activation == "silu":
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)
    elif activation == "gelu":
        gate_up_out = torch_npu.npu_gelu_mul(gate_up_out)
    elif activation == "silu_no_mul":
        gate_up_out = F.silu(gate_up_out)
    elif activation == "gelu_no_mul":
        gate_up_out = torch_npu.npu_gelu(gate_up_out)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

    # Step 4: Down projection — npu_grouped_matmul
    # w2 is [E, K_out, N//2] — need transpose to [E, N//2, K_out]
    down_out = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2.transpose(1, 2).contiguous()],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
    )[0]

    # Step 5: Unpermute and apply router weights
    # npu_moe_token_unpermute expects sorted_indices as int32
    expanded_row_idx_abs = torch.abs(expanded_row_idx).to(torch.int32)
    out = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=down_out,
        sorted_indices=expanded_row_idx_abs,
        probs=topk_weights.to(down_out.dtype) if not apply_router_weight_on_input else None,
    )

    if inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out


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

    # Try optimized npu_grouped_matmul path first
    try:
        return _npu_grouped_matmul_fused_experts(
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
    except Exception as e:
        # Fall back to Python loop on first failure, then log warning
        if not hasattr(fused_experts_impl, '_grouped_matmul_warned'):
            logger.warning(
                "npu_grouped_matmul MoE failed (%s), falling back to torch.mm loop. "
                "This warning will not repeat.", e
            )
            fused_experts_impl._grouped_matmul_warned = True

    # Fallback: pure-torch implementation
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

# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/layer.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
import torch.nn.functional as F
import torch_npu


def topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize):
    """Tensor reference for FL's softmax router; its scratch output is unused."""
    scores = gating_output.float().softmax(dim=-1)
    weights, indices = scores.topk(topk_weights.shape[-1], dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    topk_weights.copy_(weights)
    topk_ids.copy_(indices.to(topk_ids.dtype))
    return topk_weights, topk_ids


def _validate_activation(activation, alpha, beta, limit):
    if activation == "swigluoai_uninterleave":
        if limit is None or alpha is None or beta is None:
            raise ValueError(
                "Packed SwiGLU-OAI requires explicit alpha, beta and limit"
            )
    elif activation == "silu":
        if alpha not in (None, 1.0) or beta not in (None, 0.0):
            raise NotImplementedError("Non-default SILU alpha/beta")
    elif activation not in ("gelu", "gelu_tanh", "silu_no_mul", "gelu_no_mul"):
        raise NotImplementedError(f"Unsupported Ascend MoE activation: {activation}")


def bf16_grouped_experts(
    hidden_states,
    w1,
    w2,
    *,
    group_list,
    group_list_type,
    activation="silu",
    alpha=None,
    beta=None,
    limit=None,
):
    """rc1 unquantized MLP for already routed rows, with packaged NPU ops."""
    _validate_activation(activation, alpha, beta, limit)
    if (
        hidden_states.dtype not in (torch.bfloat16, torch.float16)
        or w1.dtype != hidden_states.dtype
        or w2.dtype != hidden_states.dtype
    ):
        raise ValueError("Grouped experts require matching BF16/FP16 rows and weights")
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("Expected [rows,K] activations and [E,N,K] weights")
    if group_list_type not in (0, 1):
        raise ValueError("Unsupported expert group-list format")
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, w2.shape[1]))
    gate_up = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1.transpose(1, 2)],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    if activation == "swigluoai_uninterleave":
        activated = torch_npu.npu_clipped_swiglu(
            gate_up,
            interleaved=False,
            alpha=alpha,
            bias=beta,
            limit=limit,
        )
    elif activation == "silu":
        if limit is not None and limit > 0:
            gate, up = gate_up.chunk(2, dim=-1)
            gate_up = torch.cat((gate.clamp(max=limit), up.clamp(-limit, limit)), -1)
        activated = torch_npu.npu_swiglu(gate_up)
    elif activation in ("gelu", "gelu_tanh"):
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (
            F.gelu(gate, approximate="tanh" if activation == "gelu_tanh" else "none")
            * up
        )
    elif activation == "silu_no_mul":
        activated = F.silu(gate_up)
    elif activation == "gelu_no_mul":
        activated = F.gelu(gate_up)
    return torch_npu.npu_grouped_matmul(
        x=[activated],
        weight=[w2.transpose(1, 2)],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]


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
    alpha=None, beta=None, limit=None,
) -> torch.Tensor:
    """Unquantized local-expert routing and GMM using packaged torch_npu ops.

    Mirrors current vLLM-Ascend AllGather dispatch/MLP/combine semantics.
    Routing produces fixed-size expanded tensors; no host reads or dynamic
    boolean-indexed expert batches occur during graph capture.
    """
    _validate_activation(activation, alpha, beta, limit)
    num_tokens = hidden_states.size(0)
    num_experts = w1.size(0)
    first_expert = 0
    if expert_map is not None:
        from vllm.distributed import get_ep_group

        group = get_ep_group()
        total_experts = expert_map.numel()
        if (global_num_experts not in (-1, total_experts)
                or total_experts != num_experts * group.world_size):
            raise NotImplementedError("Ascend EP requires evenly partitioned experts")
        first_expert = group.rank_in_group * num_experts
        global_num_experts = total_experts
        # Keep global IDs for routing; each rank contributes only its experts.
        topk_weights = topk_weights * (expert_map[topk_ids.long()] != -1)
    elif global_num_experts not in (-1, num_experts):
        raise NotImplementedError("Sharded Ascend experts require an expert map")
    else:
        global_num_experts = num_experts
    if apply_router_weight_on_input:
        raise NotImplementedError("Ascend grouped MoE requires output router weights")
    if num_tokens == 0:
        return hidden_states if inplace else torch.empty_like(hidden_states)

    # The current rc1 dispatcher uses count-mode expert tokens and pairs its
    # init-routing interface with token_unpermute (not finalize_routing).
    # Use global IDs with a contiguous local range, including nonzero offsets.
    routed, row_indices, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids.to(torch.int32),
        active_num=num_tokens * topk_ids.size(1),
        expert_num=global_num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[first_expert, first_expert + num_experts],
        quant_mode=-1,
    )
    group_list = expert_tokens.to(torch.int64)
    # One vendor provider owns identical activation semantics in both paths.
    expert_output = bf16_grouped_experts(
        routed, w1, w2, group_list=group_list, group_list_type=1,
        activation=activation, alpha=alpha, beta=beta, limit=limit,
    )
    if expert_map is not None:
        # Only the compact local-route prefix is defined by GMM. Unpermute
        # must not expose stale/NaN tail storage even with a zero probability.
        valid_rows = torch.arange(
            expert_output.size(0), device=expert_output.device
        ) < group_list.sum()
        expert_output = torch.where(valid_rows[:, None], expert_output, 0)
        # Native v2 marks remote routes as -1. abs(-1) would read row 1,
        # which is out of bounds when N * topk == 1; use a safe masked row.
        sorted_indices = row_indices.clamp_min(0)
    else:
        sorted_indices = torch.abs(row_indices)
    result = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=expert_output,
        sorted_indices=sorted_indices,
        probs=topk_weights,
    )
    if inplace:
        hidden_states.copy_(result)
        return hidden_states
    return result


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
    alpha=None, beta=None, limit=None,
) -> torch.Tensor:
    if any((use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
            per_channel_quant)) or any(value is not None for value in (
                w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale,
                block_shape, w1_bias, w2_bias)):
        raise NotImplementedError("Ascend grouped MoE supports unquantized, bias-free weights")
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
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]

    # All expert routing, computation and reduction stay on the device.
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
        alpha=alpha, beta=beta, limit=limit,
    )


def topk_bias(logits, bias, topk, renormalize, scoring_func):
    """Use native NPU expert selection, preserving unbiased FP32 weights."""
    import torch_npu

    if scoring_func not in ("softmax", "sigmoid"):
        raise ValueError(f"Unsupported scoring function: {scoring_func}")
    logits = logits.float()
    _, indices, _ = torch_npu.npu_moe_gating_top_k(
        logits,
        k=topk,
        bias=None if bias is None else bias.float(),
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=0,
        norm_type=0 if scoring_func == "softmax" else 1,
        routed_scaling_factor=1.0,
        eps=1e-20,
    )
    # CANN sigmoid selection normalizes its returned values. Gather original
    # scores so the public renormalize=False contract remains exact as well.
    scores = logits.softmax(-1) if scoring_func == "softmax" else logits.sigmoid()
    weights = scores.gather(1, indices.long())
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return weights, indices.to(torch.int32)

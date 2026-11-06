# SPDX-License-Identifier: Apache-2.0
"""ModelSlim dynamic INT8 experts using packaged torch_npu operators.

Routing/count metadata follows rc1 ops/fused_moe/token_dispatcher.py and
quantized GMM follows rc1 ops/fused_moe/moe_mlp.py's unfused W8A8 path.
Weights retain the loader's logical [E,N,K] shape; post-load preparation
makes their [E,K,N] runtime transpose contiguous without copying per step.
"""

import math

import torch
import torch.nn.functional as F
import torch_npu


def modelslim_w8a8_prepare_experts(w1, w2, s1, s2):
    for weight, scale in ((w1, s1), (w2, s2)):
        if weight.ndim != 3 or weight.dtype != torch.int8:
            raise ValueError("Expert weights must be int8 [experts, output, input]")
        if scale.shape != (*weight.shape[:2], 1):
            raise ValueError("Expert scales must be [experts, output, 1]")
    if (
        w1.shape[0] != w2.shape[0]
        or w1.shape[1] != 2 * w2.shape[2]
        or w1.shape[2] != w2.shape[1]
    ):
        raise ValueError("Incompatible gated expert weight shapes")
    return (
        w1.transpose(1, 2).contiguous().transpose(1, 2),
        w2.transpose(1, 2).contiguous().transpose(1, 2),
        s1.to(torch.float32 if s1.dtype == torch.float16 else s1.dtype).contiguous(),
        s2.to(torch.float32 if s2.dtype == torch.float16 else s2.dtype).contiguous(),
    )


def _activate(gate_up, activation, limit):
    if activation == "silu" and limit is None:
        return torch_npu.npu_swiglu(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "swiglustep":
        return F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
    # SiluAndMulWithClamp: clamp gate before the sigmoid, not afterwards.
    gate = gate.clamp(max=limit)
    return gate * torch.sigmoid(gate) * up.clamp(-limit, limit)


def modelslim_w8a8_experts(
    x,
    w1,
    w2,
    s1,
    s2,
    topk_weights,
    topk_ids,
    first_expert=0,
    global_num_experts=-1,
    activation="silu",
    limit=None,
):
    """Compute this contiguous local expert range; no communication/groups."""
    if activation not in ("silu", "swiglustep"):
        raise NotImplementedError(f"Unsupported ModelSlim activation: {activation}")
    if activation == "swiglustep" and limit is None:
        limit = 7.0
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError("Activation limit must be finite and positive")
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Expert input must be BF16/FP16 [tokens, hidden]")
    if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
        raise ValueError(
            "Routing IDs and weights must have matching [tokens, topk] shape"
        )
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("Routing shape does not match tokens or has empty topk")
    num_experts = w1.shape[0]
    if global_num_experts == -1:
        global_num_experts = first_expert + num_experts
    if first_expert < 0 or first_expert + num_experts > global_num_experts:
        raise ValueError("Local expert range is outside global experts")
    if x.shape[0] == 0:
        return torch.empty_like(x)

    local_routes = (topk_ids >= first_expert) & (topk_ids < first_expert + num_experts)
    probs = torch.where(local_routes, topk_weights, 0)
    routed, row_indices, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        topk_ids.to(torch.int32),
        active_num=x.shape[0] * topk_ids.shape[1],
        expert_num=global_num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[first_expert, first_expert + num_experts],
        quant_mode=-1,
    )
    group_list = expert_tokens.to(torch.int64)
    valid_rows = torch.arange(routed.shape[0], device=x.device) < group_list.sum()
    # Routing/GMM define only the compact local prefix. Sanitize tails before
    # quantization as well as unpermute: a remote row may contain stale NaNs.
    routed = torch.where(valid_rows[:, None], routed, 0)
    quantized, token_scale = torch_npu.npu_dynamic_quant(routed, dst_type=torch.int8)
    gate_up = torch_npu.npu_grouped_matmul(
        x=[quantized],
        weight=[w1.transpose(1, 2)],
        scale=[s1.squeeze(-1)],
        per_token_scale=[token_scale],
        split_item=2,
        group_type=0,
        group_list_type=1,
        group_list=group_list,
        output_dtype=x.dtype,
    )[0]
    gate_up = torch.where(valid_rows[:, None], gate_up, 0)
    activated = _activate(gate_up, activation, limit)
    quantized, token_scale = torch_npu.npu_dynamic_quant(activated, dst_type=torch.int8)
    output = torch_npu.npu_grouped_matmul(
        x=[quantized],
        weight=[w2.transpose(1, 2)],
        scale=[s2.squeeze(-1)],
        per_token_scale=[token_scale],
        split_item=2,
        group_type=0,
        group_list_type=1,
        group_list=group_list,
        output_dtype=x.dtype,
    )[0]
    output = torch.where(valid_rows[:, None], output, 0)
    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=output,
        sorted_indices=row_indices.clamp_min(0),
        probs=probs,
    )

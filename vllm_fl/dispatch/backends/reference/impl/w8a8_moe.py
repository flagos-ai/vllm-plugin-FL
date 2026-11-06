# SPDX-License-Identifier: Apache-2.0
"""Small eager oracle, including dynamic activation QDQ at both matmuls.

This reference uses float32 accumulation for integer products. It is not a
graph/performance backend and deliberately does not read distributed state.
"""

import math

import torch
import torch.nn.functional as F

from vllm_fl.quantization.w8a8.reference import dynamic_per_token_quant_int8


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
        w1.contiguous(),
        w2.contiguous(),
        s1.float().contiguous(),
        s2.float().contiguous(),
    )


def _linear(x, weight, scale):
    quantized, token_scale = dynamic_per_token_quant_int8(x)
    product = quantized.float() @ weight.float().t()
    return (product * token_scale * scale.float().reshape(1, -1)).to(x.dtype)


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
    if global_num_experts == -1:
        global_num_experts = first_expert + w1.shape[0]
    if first_expert < 0 or first_expert + w1.shape[0] > global_num_experts:
        raise ValueError("Local expert range is outside global experts")
    output = torch.zeros_like(x, dtype=torch.float32)
    # Boolean-selected expert batches and Python loops are intentional here:
    # this independent oracle is restricted to bounded eager component tests.
    for expert in range(w1.shape[0]):
        token, route = torch.where(topk_ids == first_expert + expert)
        if token.numel() == 0:
            continue
        gate_up = _linear(x[token], w1[expert], s1[expert])
        gate, up = gate_up.chunk(2, dim=-1)
        if activation == "swiglustep":
            activated = F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
        elif limit is not None:
            gate = gate.clamp(max=limit)
            activated = gate * torch.sigmoid(gate) * up.clamp(-limit, limit)
        else:
            # Fused SwiGLU rounds once on output, not between SiLU and mul.
            activated = (F.silu(gate.float()) * up.float()).to(gate_up.dtype)
        local_output = _linear(activated, w2[expert], s2[expert])
        weighted = local_output.float() * topk_weights[token, route, None].float()
        output.index_add_(0, token, weighted)
    return output.to(x.dtype)

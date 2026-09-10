# Copyright (c) 2026 BAAI. All rights reserved.

"""Unquantized Ascend experts for vLLM's modular MoE interface."""

import torch
import torch.nn.functional as F
import torch_npu


def grouped_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    clamp_limit: float | None = None,
) -> torch.Tensor:
    """Group routed rows for two native NPU matmuls, then reduce per token.

    Inputs follow FusedMoEExpertsModular.apply: the prepare stage has already
    applied input-side router weights. Missing local experts contribute zero.
    Weight tensors remain in vLLM's [expert, output, input] storage layout;
    grouped matmul accepts transposed weight views without copying the model.
    """
    if activation not in (
        "silu",
        "gelu",
        "gelu_tanh",
        "silu_no_mul",
        "gelu_no_mul",
        "gelu_tanh_no_mul",
    ):
        raise NotImplementedError(f"Ascend grouped MoE activation: {activation}")
    if clamp_limit is not None and activation != "silu":
        raise NotImplementedError("Ascend grouped MoE clamp requires silu")

    tokens, top_k = topk_ids.shape
    if tokens == 0:
        return hidden_states.new_empty((0, w2.shape[1]))
    local_ids = topk_ids.long()
    if expert_map is not None:
        local_ids = torch.where(
            local_ids >= 0,
            expert_map[local_ids.clamp(min=0)],
            -1,
        )
    local_ids = local_ids.flatten()
    # Dropping nonlocal routes also handles EP ranks receiving no local work.
    routed_rows = torch.where((local_ids >= 0) & (local_ids < w1.shape[0]))[0]
    routed = hidden_states.new_zeros((tokens * top_k, w2.shape[1]))
    if routed_rows.numel() == 0:
        return routed.view(tokens, top_k, w2.shape[1]).sum(dim=1)
    expert_ids, order = torch.sort(local_ids[routed_rows])
    routed_rows = routed_rows[order]
    counts = torch.bincount(expert_ids.long(), minlength=w1.shape[0]).to(torch.int64)
    expanded = hidden_states[routed_rows // top_k].contiguous()

    def matmul(x, weight, bias):
        return torch_npu.npu_grouped_matmul(
            [x],
            [weight.transpose(1, 2)],
            bias=None if bias is None else [bias.float()],
            group_list=counts,
            group_type=0,
            group_list_type=1,
            split_item=3,
        )[0]

    gate_up = matmul(expanded, w1, w1_bias)
    if activation == "silu":
        if clamp_limit is not None:
            gate, up = gate_up.float().chunk(2, dim=-1)
            activated = (
                F.silu(gate.clamp(max=clamp_limit))
                * up.clamp(-clamp_limit, clamp_limit)
            ).to(gate_up.dtype)
        else:
            activated = torch_npu.npu_swiglu(gate_up, dim=-1)
    elif activation.endswith("_no_mul"):
        activated = (
            F.silu(gate_up)
            if activation == "silu_no_mul"
            else F.gelu(
                gate_up,
                approximate="tanh" if activation == "gelu_tanh_no_mul" else "none",
            )
        )
    else:
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (
            F.gelu(
                gate,
                approximate="tanh" if activation == "gelu_tanh" else "none",
            )
            * up
        )
    expert_output = matmul(activated.contiguous(), w2, w2_bias)
    if not apply_router_weight_on_input:
        # Keep router multiplication in FP32, matching fused_moe's accumulator.
        expert_output = (
            expert_output.float() * topk_weights.flatten()[routed_rows, None]
        ).to(hidden_states.dtype)
    routed[routed_rows] = expert_output
    return routed.view(tokens, top_k, w2.shape[1]).sum(dim=1)

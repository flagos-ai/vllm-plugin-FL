# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/layer.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch_npu

logger = logging.getLogger(__name__)

_ASCENDC_MOE_AVAILABLE: bool | None = None


def ascendc_moe_available() -> bool:
    """Whether the AscendC fused-MoE custom ops can be used.

    Requires the packaged CANN custom-op package (same one the GDN patch
    bootstraps) and the ``_C_ascend`` torch bindings.  Note: the custom
    ``moe_grouped_matmul`` kernel is *not* used on purpose — it crashes the
    aicore on ascend910b even for trivial single-expert inputs (CCU
    instruction address check error), so the grouped matmuls stay on
    ``torch_npu.npu_grouped_matmul``.  Set ``VLLM_FL_DISABLE_ASCENDC_MOE=1``
    to keep the FlagGems/torch_npu path.
    """
    global _ASCENDC_MOE_AVAILABLE
    if _ASCENDC_MOE_AVAILABLE is not None:
        return _ASCENDC_MOE_AVAILABLE
    if os.environ.get("VLLM_FL_DISABLE_ASCENDC_MOE", "0") == "1":
        logger.info("VLLM_FL_DISABLE_ASCENDC_MOE=1, keep FlagGems/torch_npu MoE path")
        _ASCENDC_MOE_AVAILABLE = False
        return False
    try:
        from ..patches.patch_qwen3_6_gdn import _bootstrap_custom_op_env

        if not _bootstrap_custom_op_env():
            _ASCENDC_MOE_AVAILABLE = False
            return False
    except Exception as e:
        logger.warning("CANN custom op bootstrap failed: %s", e)
        _ASCENDC_MOE_AVAILABLE = False
        return False
    missing = [
        name
        for name in ("moe_gating_top_k", "npu_moe_init_routing_custom")
        if not hasattr(torch.ops._C_ascend, name)
    ]
    if missing:
        logger.warning("torch.ops._C_ascend missing ops %s; keep torch_npu MoE path", missing)
        _ASCENDC_MOE_AVAILABLE = False
        return False
    _ASCENDC_MOE_AVAILABLE = True
    return True


def fused_topk_ascend(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Drop-in replacement for vllm_fl ``fused_topk`` (softmax routing only).

    Uses the fused AscendC ``moe_gating_top_k`` kernel (softmax + top-k +
    optional L1 renorm in one launch) instead of the FlagGems Triton
    ``topk_softmax``.  Applies to the plain ``RoutingMethodType.Renormalize``
    routing used by Qwen3.5/Qwen3.6 (no grouped top-k, no correction bias);
    other routing modes never reach ``fused_topk``.
    """
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    # fp32 input keeps the softmax/renorm precision identical to the
    # FlagGems path and makes the returned topk_weights fp32 as before.
    topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
        gating_output.to(torch.float32),
        topk,
        1,  # k_group: no grouped top-k
        1,  # group_count: no grouped top-k
        0,  # group_select_mode
        int(renormalize),  # renorm: 1 = L1 renorm of the top-k weights
        0,  # norm_type: softmax
        False,  # out_flag
        1.0,  # routed_scaling_factor
        1e-20,  # eps
        None,  # bias_opt
    )
    if indices_type is not None:
        topk_ids = topk_ids.to(indices_type)
    return topk_weights, topk_ids, None


def convert_moe_weights_pretransposed(layer: torch.nn.Module) -> None:
    """Transpose FusedMoE expert weights once at load time.

    The legacy path pays a ``w.transpose(1, 2).contiguous()`` copy for both
    grouped matmuls on every forward (e.g. 2 x 128MB per layer per step for
    Qwen3.6-35B-A3B TP=4).  Store them pre-transposed instead:

        w13_weight: [E, 2*intermediate, hidden] -> [E, hidden, 2*intermediate]
        w2_weight:  [E, hidden, intermediate]   -> [E, intermediate, hidden]

    Idempotent; only meaningful together with ``_ascendc_fused_experts_impl``
    (which detects the layout from the weight shapes).
    """
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    # After conversion w13.size(1) == hidden == w2.size(2); before, they are
    # 2*intermediate and intermediate respectively.
    if w13.size(1) == w2.size(2):
        return
    layer.w13_weight.data = w13.transpose(1, 2).contiguous()
    layer.w2_weight.data = w2.transpose(1, 2).contiguous()


def _ascendc_fused_experts_impl(
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
    """AscendC fused MoE experts implementation.

    Same structure as ``_torch_fused_experts_impl`` but the token
    routing/permute runs on the AscendC ``npu_moe_init_routing_custom``
    kernel and the weights are pre-transposed (see
    ``convert_moe_weights_pretransposed``), so the grouped matmuls consume
    them directly:

        w1: [E, hidden, 2*intermediate]  (pre-transposed)
        w2: [E, intermediate, hidden]    (pre-transposed)

    ``expanded_row_idx`` (row_idx_type=0) is the gather map expected by
    ``npu_moe_token_unpermute`` (``|.|`` guards inactive rows), so the
    scatter-based inverse permutation of the legacy path is not needed.
    All operations are graph-safe.
    """
    num_tokens, hidden_dim = hidden_states.size()
    E = w1.size(0)
    top_k = topk_ids.size(1)

    if global_num_experts == -1:
        global_num_experts = len(expert_map) if expert_map is not None else E

    # Map global expert ids to local expert ids.  Out-of-range entries are
    # clamped to 0 and masked to zero weight so the graph shape stays static.
    if expert_map is not None:
        mask = expert_map[topk_ids.long()] != -1
        local_topk_ids = expert_map[topk_ids.long()].clamp(min=0)
        topk_weights = topk_weights * mask.to(topk_weights.dtype)
    else:
        local_topk_ids = topk_ids.long()

    # Expand tokens according to top-k expert assignment and sort them by
    # expert.  expanded_row_idx maps each sorted row back to the original
    # flat (token*topk + k) position (row_idx_type=0).
    expanded_x, expanded_row_idx, expert_token_count, _ = (
        torch.ops._C_ascend.npu_moe_init_routing_custom(
            hidden_states,
            local_topk_ids.to(torch.int32),
            active_num=num_tokens * top_k,
            expert_num=global_num_experts,
            drop_pad_mode=0,
            expert_tokens_num_type=1,  # count mode
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[0, E],
            row_idx_type=0,
        )
    )

    # Apply router weight on the expanded tokens if requested.
    if apply_router_weight_on_input:
        expanded_weights = (
            topk_weights.view(-1)[expanded_row_idx.abs().long()]
            .unsqueeze(-1)
            .to(expanded_x.dtype)
        )
        expanded_x = expanded_x * expanded_weights
        probs = None
    else:
        probs = topk_weights

    # Gate-up grouped matmul on the pre-transposed weight:
    # x [total_tokens*top_k, hidden], weight [E, hidden, 2*intermediate].
    gate_up = torch_npu.npu_grouped_matmul(
        [expanded_x],
        [w1],
        group_list=expert_token_count,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]

    # Activation.
    if activation == "silu":
        gate_up = torch_npu.npu_swiglu(gate_up)
    elif activation == "gelu":
        d = gate_up.shape[-1] // 2
        gate_up = F.gelu(gate_up[..., :d]) * gate_up[..., d:]
    elif activation == "silu_no_mul":
        gate_up = F.silu(gate_up)
    elif activation == "gelu_no_mul":
        gate_up = F.gelu(gate_up)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

    # Down grouped matmul on the pre-transposed weight:
    # x [total_tokens*top_k, intermediate], weight [E, intermediate, hidden].
    down = torch_npu.npu_grouped_matmul(
        [gate_up],
        [w2],
        group_list=expert_token_count,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]

    # Scatter/sum the expert outputs back to the token dimension.
    out = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=down,
        sorted_indices=expanded_row_idx.abs(),
        probs=probs,
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
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ascend NPU native fused MoE experts implementation.

    This implementation replaces the previous pure-PyTorch loop with
    ``npu_moe_init_routing_v2`` + ``npu_grouped_matmul`` +
    ``npu_moe_token_unpermute``.  All operations are graph-safe (no host-side
    synchronization or data-dependent control flow), so the whole MoE layer can
    be captured inside ``torch.npu.NPUGraph`` / ``torch.compile(fullgraph=True)``.

    Weight layout expected from vLLM ``FusedMoE``:
        - w1: [E, 2*intermediate, hidden]
        - w2: [E, hidden, intermediate]
    For grouped matmul we transpose the last two dims so each expert weight
    becomes [hidden, out_features].
    """
    num_tokens, hidden_dim = hidden_states.size()
    E = w1.size(0)
    top_k = topk_ids.size(1)

    if global_num_experts == -1:
        global_num_experts = len(expert_map) if expert_map is not None else E

    # Map global expert ids to local expert ids.  Out-of-range entries are
    # clamped to 0 and masked to zero weight so the graph shape stays static.
    if expert_map is not None:
        mask = expert_map[topk_ids.long()] != -1
        local_topk_ids = expert_map[topk_ids.long()].clamp(min=0)
        topk_weights = topk_weights * mask.to(topk_weights.dtype)
    else:
        local_topk_ids = topk_ids.long()

    # Expand tokens according to top-k expert assignment and sort them by
    # expert.  row_idx maps each sorted row back to the original flat
    # (token*topk + k) position.
    expanded_x, row_idx, expert_token_count, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        local_topk_ids.to(torch.int32),
        active_num=num_tokens * top_k,
        expert_num=global_num_experts,
        expert_tokens_num_type=1,  # count mode
        expert_tokens_num_flag=True,
        quant_mode=-1,
        active_expert_range=[0, E],
        row_idx_type=1,
    )

    # Apply router weight on the expanded tokens if requested.
    if apply_router_weight_on_input:
        expanded_weights = (
            topk_weights.view(-1)[row_idx.long()]
            .unsqueeze(-1)
            .to(expanded_x.dtype)
        )
        expanded_x = expanded_x * expanded_weights
        probs = None
    else:
        probs = topk_weights

    # npu_moe_token_unpermute expects ``sorted_indices`` as the *gather* index
    # (sorted_position -> original flat position).  Compute the inverse of
    # row_idx on-device.
    sorted_indices = torch.empty_like(row_idx)
    sorted_indices.scatter_(
        0,
        row_idx.long(),
        torch.arange(row_idx.numel(), device=row_idx.device, dtype=torch.int32),
    )

    # Gate-up grouped matmul: x shape [total_tokens*top_k, hidden]
    # weight shape [E, hidden, 2*intermediate].
    w1_t = w1.transpose(1, 2).contiguous()
    gate_up = torch_npu.npu_grouped_matmul(
        [expanded_x],
        [w1_t],
        group_list=expert_token_count,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]

    # Activation.
    if activation == "silu":
        gate_up = torch_npu.npu_swiglu(gate_up)
    elif activation == "gelu":
        d = gate_up.shape[-1] // 2
        gate_up = F.gelu(gate_up[..., :d]) * gate_up[..., d:]
    elif activation == "silu_no_mul":
        gate_up = F.silu(gate_up)
    elif activation == "gelu_no_mul":
        gate_up = F.gelu(gate_up)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

    # Down grouped matmul: x shape [total_tokens*top_k, intermediate]
    # weight shape [E, intermediate, hidden].
    w2_t = w2.transpose(1, 2).contiguous()
    down = torch_npu.npu_grouped_matmul(
        [gate_up],
        [w2_t],
        group_list=expert_token_count,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]

    # Scatter/sum the expert outputs back to the token dimension.
    out = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=down,
        sorted_indices=sorted_indices,
        probs=probs,
    )

    if inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out


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
    # Unquantized weights that went through convert_moe_weights_pretransposed
    # are stored as [E, hidden, ...] (hidden_states.size(1) == w1.size(1));
    # the AscendC custom-op path consumes them directly.
    is_unquantized = not (
        use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16
    )
    if (
        is_unquantized
        and hidden_states.size(1) == w1.size(1)
        and ascendc_moe_available()
    ):
        return _ascendc_fused_experts_impl(
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

    # Quantized MoE is not handled here; fall back would require a different
    # code path. For the graph-mode Qwen3.6-35B-A3B serving scenario we only
    # need the unquantized path.
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

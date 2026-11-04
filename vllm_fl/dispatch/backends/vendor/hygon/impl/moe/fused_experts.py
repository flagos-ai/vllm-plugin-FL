# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon functional fused-MoE pipeline with per-stage GEMM configs."""

from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _get_config_quant_dtype,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.triton_utils import tl

from vllm_fl.dispatch import CachedOp

from .bf16_moe_fusions import (
    invoke_hygon_bf16_gemm1_silu,
    supports_hygon_bf16_gemm1_silu,
    try_hygon_fixed_topk8_reduce,
)
from .stage_config import (
    requires_separate_expert_assignment,
    resolve_moe_stage_configs,
)

logger = init_logger(__name__)

_moe_align_block_size = CachedOp("moe_align_block_size")
_invoke_fused_moe_triton_kernel = CachedOp("invoke_fused_moe_triton_kernel")
_silu_and_mul = CachedOp("silu_and_mul")
_gelu_and_mul = CachedOp("gelu_and_mul")
_moe_sum = CachedOp("moe_sum")


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
    """Run functional fused MoE with independent GEMM1/GEMM2 configs."""
    del w1_zp, w2_zp

    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(
            2
        ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    num_local_experts, intermediate_size, _ = w1.size()
    hidden_size = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = num_local_experts
    top_k_num = topk_ids.size(1)

    chunk_size = 65536
    max_chunk_tokens = min(num_tokens, chunk_size)
    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=None,
        dtype=hidden_states.dtype,
    )
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        ocp_mx_scheme=None,
    )
    get_config = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    raw_config = get_config(max_chunk_tokens)
    gemm1_config, gemm2_config = resolve_moe_stage_configs(raw_config)
    if "gemm1" in raw_config:
        logger.info_once(
            f"Using Hygon stage-specific MoE configs: "
            f"GEMM1={gemm1_config}, GEMM2={gemm2_config}"
        )

    cache13 = torch.empty(
        max_chunk_tokens * top_k_num * max(intermediate_size, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[
        : max_chunk_tokens * top_k_num * intermediate_size
    ].view(max_chunk_tokens, top_k_num, intermediate_size)
    intermediate_cache3 = cache13[: max_chunk_tokens * top_k_num * hidden_size].view(
        max_chunk_tokens, top_k_num, hidden_size
    )
    intermediate_cache2 = torch.empty(
        (max_chunk_tokens * top_k_num, intermediate_size // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    for begin_chunk_idx in range(0, num_tokens, chunk_size):
        end_chunk_idx = min(begin_chunk_idx + chunk_size, num_tokens)
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk = curr_hidden_states.size(0)

        curr_cache1 = intermediate_cache1[:tokens_in_chunk]
        curr_cache2 = intermediate_cache2[: tokens_in_chunk * top_k_num]
        curr_cache3 = intermediate_cache3[:tokens_in_chunk]

        if tokens_in_chunk != max_chunk_tokens:
            raw_config = get_config(tokens_in_chunk)
            gemm1_config, gemm2_config = resolve_moe_stage_configs(raw_config)
            if "gemm1" in raw_config:
                logger.info_once(
                    f"Using Hygon stage-specific MoE configs: "
                    f"GEMM1={gemm1_config}, GEMM2={gemm2_config}"
                )

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        activation_name = (
            activation.value if hasattr(activation, "value") else activation
        )
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
            curr_topk_ids,
            gemm1_config["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )
        use_fused_gemm1_silu = supports_hygon_bf16_gemm1_silu(
            qcurr_hidden_states,
            w1,
            curr_cache2,
            sorted_token_ids,
            expert_ids,
            gemm1_config,
            top_k=top_k_num,
            activation_name=activation_name,
            apply_router_weight_on_input=apply_router_weight_on_input,
            has_quantization=any(
                (
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    a1q_scale is not None,
                    w1_scale is not None,
                    block_shape is not None,
                )
            ),
            has_bias=w1_bias is not None,
            has_expert_map=expert_map is not None,
            has_lora=False,
        )
        if use_fused_gemm1_silu:
            invoke_hygon_bf16_gemm1_silu(
                qcurr_hidden_states,
                w1,
                curr_cache2,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k=top_k_num,
            )
        else:
            _invoke_fused_moe_triton_kernel(
                qcurr_hidden_states,
                w1,
                curr_cache1,
                a1q_scale,
                w1_scale,
                curr_topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                apply_router_weight_on_input,
                top_k_num,
                gemm1_config,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
                B_bias=w1_bias,
            )
            if activation_name == "silu":
                curr_cache2 = _silu_and_mul(
                    None, curr_cache1.view(-1, intermediate_size)
                )
            elif activation_name == "gelu":
                curr_cache2 = _gelu_and_mul(
                    None, curr_cache1.view(-1, intermediate_size)
                )
            elif activation_name == "silu_no_mul":
                curr_cache2 = F.silu(curr_cache1.view(-1, intermediate_size))
            elif activation_name == "gelu_no_mul":
                curr_cache2 = F.gelu(curr_cache1.view(-1, intermediate_size))
            else:
                raise ValueError(f"Unsupported FusedMoe activation: {activation_name}.")

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=curr_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
        )

        if requires_separate_expert_assignment(gemm1_config, gemm2_config):
            (
                gemm2_sorted_token_ids,
                gemm2_expert_ids,
                gemm2_num_tokens_post_padded,
            ) = _moe_align_block_size(
                curr_topk_ids,
                gemm2_config["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map,
                ignore_invalid_experts=True,
            )
        else:
            gemm2_sorted_token_ids = sorted_token_ids
            gemm2_expert_ids = expert_ids
            gemm2_num_tokens_post_padded = num_tokens_post_padded

        _invoke_fused_moe_triton_kernel(
            qintermediate_cache2,
            w2,
            curr_cache3,
            a2q_scale,
            w2_scale,
            curr_topk_weights,
            gemm2_sorted_token_ids,
            gemm2_expert_ids,
            gemm2_num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            gemm2_config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
        )
        curr_output = out_hidden_states[begin_chunk_idx:end_chunk_idx]
        if not try_hygon_fixed_topk8_reduce(curr_cache3, curr_output):
            _moe_sum(curr_cache3, curr_output)

    return out_hidden_states

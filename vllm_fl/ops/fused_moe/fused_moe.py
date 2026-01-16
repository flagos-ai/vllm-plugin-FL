# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/fused_moe.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Literal, Optional, Union
import functools
import torch
import torch.nn.functional as F
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.config import (FUSED_MOE_UNQUANTIZED_CONFIG, 
                                                         FusedMoEQuantConfig, _get_config_dtype_str)
from vllm.model_executor.layers.fused_moe.fused_moe import (_get_config_quant_dtype, 
                                                            try_get_optimal_moe_config, invoke_fused_moe_kernel)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.triton_utils import tl, triton
from vllm_fl.ops.fused_moe.moe_align_block_size import moe_align_block_size
from vllm_fl.ops._fl_ops import FLOps as fl_ops

def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, ...]:
    return fl_ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize
    )

def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    topk_weights, topk_ids = vllm_topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
    )

    return topk_weights, topk_ids, token_expert_indices

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
        assert hidden_states.size(1) // 2 == w1.size(2), (
            "Hidden size mismatch")
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}")

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = _get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
                                         use_int8_w8a16=use_int8_w8a16,
                                         use_int4_w4a16=use_int4_w4a16,
                                         use_mxfp4_w4a4=False, ## dont support mxfp4
                                         dtype=hidden_states.dtype)

    # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
    # quantized prior to calling fused_experts.
    quant_dtype = _get_config_quant_dtype(use_fp8_w8a8=use_fp8_w8a8,
                                          use_int8_w8a8=use_int8_w8a8,
                                          use_mxfp4_w4a4=False)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, K),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.size(1)]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                 global_num_experts, expert_map))

        invoke_fused_moe_kernel(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                a1q_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                apply_router_weight_on_input,
                                top_k_num,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                B_bias=w1_bias)

        # Activation function with multiplication
        if activation == "silu":
            intermediate_cache2 = fl_ops.silu_and_mul(intermediate_cache1.view(-1, N))
            # torch.ops._C.silu_and_mul(intermediate_cache2,
            #                           intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            intermediate_cache2 = fl_ops.gelu_and_mul(intermediate_cache1.view(-1, N))
        # Activation function without multiplication
        elif activation == "silu_no_mul":
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == "gelu_no_mul":
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))

        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape)

        invoke_fused_moe_kernel(qintermediate_cache2,
                                w2,
                                intermediate_cache3,
                                a2q_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                not apply_router_weight_on_input,
                                1,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                B_bias=w2_bias)

        fl_ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.size()),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])

    return out_hidden_states

def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Optional[FusedMoEQuantConfig] = None,
    allow_deep_gemm: bool = False,
    allow_cutlass_block_scaled_grouped_gemm: bool = False,
) -> torch.Tensor:
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    ###NOTE: weight per_channel_quant same as act per_act_token_quant
    return fused_experts_impl(hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=quant_config.use_fp8_w8a8,
            use_int8_w8a8=quant_config.use_int8_w8a8,
            use_int8_w8a16=quant_config.use_int8_w8a16,
            use_int4_w4a16=quant_config.use_int4_w4a16,
            per_channel_quant=quant_config.per_act_token_quant,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            w1_zp=quant_config.w1_zp,
            w2_zp=quant_config.w2_zp,
            a1_scale=quant_config.a1_scale,
            a2_scale=quant_config.a2_scale,
            block_shape=quant_config.block_shape,
            w1_bias=quant_config.w1_bias,
            w2_bias=quant_config.w2_bias)
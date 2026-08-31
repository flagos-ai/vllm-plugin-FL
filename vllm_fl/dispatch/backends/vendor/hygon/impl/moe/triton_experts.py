# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon modular MoE experts with independent GEMM stage configs."""

from __future__ import annotations

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import (
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.triton_utils import tl

from vllm_fl.dispatch import CachedOp
from vllm_fl.ops.fused_moe.activation import apply_moe_activation
from vllm_fl.ops.fused_moe.fused_moe_utils import (
    TritonExpertsFL,
    _prepare_expert_assignment,
)

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

_invoke_fused_moe_triton_kernel = CachedOp("invoke_fused_moe_triton_kernel")
_moe_sum = CachedOp("moe_sum")


class HygonTritonExpertsFL(TritonExpertsFL):
    """Hygon Triton experts using separate GEMM1 and GEMM2 route plans."""

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del expert_tokens_meta

        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(
                2
            ), f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        num_local_experts, num_tokens, intermediate_size, hidden_size, top_k_num = (
            self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        )
        if global_num_experts == -1:
            global_num_experts = num_local_experts

        raw_config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )
        gemm1_config, gemm2_config = resolve_moe_stage_configs(raw_config)
        if "gemm1" in raw_config:
            logger.info_once(
                f"Using Hygon stage-specific MoE configs: "
                f"GEMM1={gemm1_config}, GEMM2={gemm2_config}"
            )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif hidden_states.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        intermediate_cache1 = _resize_cache(
            workspace2,
            (num_tokens, top_k_num, intermediate_size),
        )
        cache2_dim = self.adjust_N_for_activation(intermediate_size, activation)
        intermediate_cache2 = _resize_cache(
            workspace13,
            (num_tokens * top_k_num, cache2_dim),
        )
        intermediate_cache3 = _resize_cache(
            workspace2,
            (num_tokens, top_k_num, hidden_size),
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                gemm1_config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        )
        sorted_token_ids_lora = None
        expert_ids_lora = None
        num_tokens_post_padded_lora = None
        token_lora_mapping = None
        lora_context = self._lora_context
        activation_name = (
            activation.value if hasattr(activation, "value") else str(activation)
        )
        use_fused_gemm1_silu = supports_hygon_bf16_gemm1_silu(
            hidden_states,
            w1,
            intermediate_cache2,
            sorted_token_ids,
            expert_ids,
            gemm1_config,
            top_k=top_k_num,
            activation_name=activation_name,
            apply_router_weight_on_input=apply_router_weight_on_input,
            has_quantization=any(
                (
                    self.quant_config.use_fp8_w8a8,
                    self.quant_config.use_int8_w8a8,
                    self.quant_config.use_int8_w8a16,
                    self.quant_config.use_int4_w4a16,
                    self.per_act_token_quant,
                    self.quantization_emulation,
                    a1q_scale is not None,
                    self.w1_scale is not None,
                    self.block_shape is not None,
                )
            ),
            has_bias=self.w1_bias is not None,
            has_expert_map=expert_map is not None,
            has_lora=lora_context is not None,
        )
        if use_fused_gemm1_silu:
            invoke_hygon_bf16_gemm1_silu(
                hidden_states,
                w1,
                intermediate_cache2,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k=top_k_num,
            )
        else:
            _invoke_fused_moe_triton_kernel(
                hidden_states,
                w1,
                intermediate_cache1,
                a1q_scale,
                self.w1_scale,
                None,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                top_k_num,
                gemm1_config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w1_bias,
            )

            if lora_context is not None:
                (
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                    token_lora_mapping,
                ) = self.apply_w13_lora(
                    lora_context,
                    y=intermediate_cache1,
                    x=hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    expert_map=expert_map,
                    w1=w1,
                    w2=w2,
                    num_tokens=num_tokens,
                    top_k_num=top_k_num,
                )

            apply_moe_activation(
                activation,
                intermediate_cache2,
                intermediate_cache1.view(-1, intermediate_size),
            )
        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
            quantization_emulation=self.quantization_emulation,
        )

        if requires_separate_expert_assignment(gemm1_config, gemm2_config):
            (
                gemm2_sorted_token_ids,
                gemm2_expert_ids,
                gemm2_num_tokens_post_padded,
            ) = _prepare_expert_assignment(
                topk_ids,
                gemm2_config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        else:
            gemm2_sorted_token_ids = sorted_token_ids
            gemm2_expert_ids = expert_ids
            gemm2_num_tokens_post_padded = num_tokens_post_padded

        _invoke_fused_moe_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            self.w2_scale,
            topk_weights,
            gemm2_sorted_token_ids,
            gemm2_expert_ids,
            gemm2_num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            gemm2_config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w2_bias,
        )

        if lora_context is not None:
            self.apply_w2_lora(
                lora_context,
                y=intermediate_cache3,
                x=intermediate_cache2,
                topk_weights=topk_weights,
                sorted_token_ids_lora=sorted_token_ids_lora,
                expert_ids_lora=expert_ids_lora,
                num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                token_lora_mapping=token_lora_mapping,
                num_tokens=num_tokens,
                w1=w1,
                w2=w2,
                top_k_num=top_k_num,
            )

        if not try_hygon_fixed_topk8_reduce(intermediate_cache3, output):
            self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        _moe_sum(input, output)

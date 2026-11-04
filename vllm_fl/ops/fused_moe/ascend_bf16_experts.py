# SPDX-License-Identifier: Apache-2.0
"""Unquantized computation on rows already dispatched by Ascend MoE."""

import torch
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm_fl.dispatch import CachedOp
from .prepare_finalize import NpuExpertTokensMetadata

_grouped_experts = CachedOp("bf16_grouped_experts")


class AscendBF16DispatchedExperts(mk.FusedMoEExpertsModular):
    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        if (moe_config.is_lora_enabled or moe_config.has_bias
                or quant_config.quant_dtype is not None
                or quant_config.weight_quant_dtype is not None):
            raise NotImplementedError("Ascend dispatched experts require bias-free unquantized weights")

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device():
        from vllm.platforms import current_platform
        return (current_platform.device_type == "npu"
                and getattr(current_platform, "vendor_name", None) == "ascend")

    @staticmethod
    def _supports_no_act_and_mul():
        return True

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation):
        return activation.value in (
            "silu", "swigluoai_uninterleave", "gelu", "gelu_tanh",
            "silu_no_mul", "gelu_no_mul",
        )

    @staticmethod
    def _supports_parallel_config(config):
        return (config.use_ep and not config.is_sequence_parallel
                and config.pcp_size == 1 and not config.enable_eplb
                and not config.use_batched_activation_format)

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        # Received capacity is unrelated to the original local token count.
        return w1.shape[0], a1.shape[0], w1.shape[1], a1.shape[1], topk_ids.shape[1]

    def workspace_shapes(self, M, N, K, topk, global_num_experts,
                         local_num_experts, expert_tokens_meta, activation):
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceNoOP()

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        if (apply_router_weight_on_input or a1q_scale is not None
                or a2_scale is not None):
            raise NotImplementedError("BF16 experts require unquantized rows and output routing")
        if not isinstance(expert_tokens_meta, NpuExpertTokensMetadata):
            raise ValueError("Ascend dispatched expert metadata is required")
        if output.dtype != expert_tokens_meta.output_dtype:
            raise ValueError("Expert output dtype differs from original hidden states")
        output.copy_(_grouped_experts(
            hidden_states, w1, w2,
            group_list=expert_tokens_meta.group_list,
            group_list_type=expert_tokens_meta.group_list_type,
            activation=activation.value,
            alpha=self.quant_config.gemm1_alpha,
            beta=self.quant_config.gemm1_beta,
            limit=self.quant_config.gemm1_clamp_limit,
        ))

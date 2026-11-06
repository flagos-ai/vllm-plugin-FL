# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""M3 differences from the shared FL MoE pipeline; no copied backend selector."""

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.dispatch import CachedOp
from vllm_fl.minimax_m3_config import load
from vllm_fl.ops.fused_moe.activation import apply_moe_activation
from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL as BaseTritonExpertsFL

_moe_align_block_size = CachedOp("m3_moe_align_block_size")
_invoke_fused_moe_triton_kernel = CachedOp("m3_invoke_fused_moe_triton_kernel")
_moe_sum = CachedOp("m3_moe_sum")


class TritonExpertsFL(BaseTritonExpertsFL):
    def prepare_expert_assignment(self, *args, **kwargs):
        return super().prepare_expert_assignment(
            *args, **kwargs, align_block_size=_moe_align_block_size
        )

    def make_launch_context(
        self,
        topk_ids,
        config,
        num_tokens,
        top_k_num,
        global_num_experts,
        expert_map,
        assignment,
    ):
        from .moe import ExpertLaunchContext

        return ExpertLaunchContext(
            topk_ids,
            config,
            num_tokens,
            top_k_num,
            global_num_experts,
            expert_map,
            assignment,
            self.prepare_expert_assignment,
            dict(
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            ),
        )

    def invoke_gemm(self, context, stage, *args, **kwargs):
        context.invoke(stage, _invoke_fused_moe_triton_kernel, *args, **kwargs)

    def apply_activation(self, activation, output, input):
        if (
            activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
            and load().framework.flagos_swiglu
        ):
            from .layers import swiglu

            assert self.gemm1_clamp_limit is not None
            swiglu(
                input,
                self.gemm1_clamp_limit,
                self.gemm1_alpha,
                self.gemm1_beta,
                out=output,
            )
        else:
            # Preserve the accepted fallback exactly (including omitted kwargs).
            apply_moe_activation(activation, output, input)

    def moe_sum(self, input, output):
        _moe_sum(input, output)

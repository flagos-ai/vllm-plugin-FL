# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from compressed_tensors.quantization import QuantizationStrategy

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (
    CompressedTensorsW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)

from vllm_fl.dispatch.backends.vendor.metax.patches.scaled_mm import (
    MctlassInt8ScaledMMLinearKernel,
)
from vllm_fl.quantization.w8a8.moe_experts import TritonW8A8Experts


class MetaXTritonW8A8Experts(TritonW8A8Experts):
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
        from vllm_fl.ops.fused_moe.fused_moe import fused_experts_impl

        result = fused_experts_impl(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation.value,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=self.quant_config.per_act_token_quant,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=self.quant_config.w1_scale,
            w2_scale=self.quant_config.w2_scale,
            w1_zp=self.quant_config.w1_zp,
            w2_zp=self.quant_config.w2_zp,
            a1_scale=self.quant_config.a1_scale,
            a2_scale=self.quant_config.a2_scale,
            block_shape=self.quant_config.block_shape,
            w1_bias=self.quant_config.w1_bias,
            w2_bias=self.quant_config.w2_bias,
        )
        output.copy_(result)


class DeepseekV4W8A8Int8(CompressedTensorsW8A8Int8):
    def create_weights(self, *args, **kwargs) -> None:
        super().create_weights(*args, **kwargs)
        config = Int8ScaledMMLinearLayerConfig(
            is_channelwise=self.strategy == QuantizationStrategy.CHANNEL,
            is_static_input_scheme=self.is_static_input_scheme,
            input_symmetric=self.input_symmetric,
        )
        self.kernel = MctlassInt8ScaledMMLinearKernel(
            config,
            layer_param_names=[
                "weight",
                "weight_scale",
                "input_scale",
                "input_zero_point",
                "azp_adj",
            ],
        )


class DeepseekV4W8A8Config(CompressedTensorsConfig):
    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if (
            isinstance(method, CompressedTensorsLinearMethod)
            and type(layer.scheme) is CompressedTensorsW8A8Int8
        ):
            scheme = layer.scheme
            layer.scheme = DeepseekV4W8A8Int8(
                scheme.strategy,
                scheme.is_static_input_scheme,
                scheme.input_symmetric,
            )
        elif isinstance(method, CompressedTensorsW8A8Int8MoEMethod):
            method.experts_cls = MetaXTritonW8A8Experts
        return method

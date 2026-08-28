# Copyright 2026 FlagOS Contributors
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Ascend ModelSlim Dynamic W8A8 MoE method.

Routing and distributed output reduction stay in vLLM's ``MoERunner``. This
module owns only the ModelSlim expert parameter layout and the Ascend expert
compute path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from .w8a8 import _npu_dynamic_quant, _npu_format_nz, _npu_quant_matmul

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEQuantConfig,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEPrepareAndFinalizeModular,
    )


def _npu_moe_activation(
    activation: MoEActivation,
    gate_up: torch.Tensor,
) -> torch.Tensor:
    import torch_npu

    if activation == MoEActivation.SILU:
        return torch_npu.npu_swiglu(gate_up)
    if activation == MoEActivation.GELU:
        return torch_npu.npu_gelu_mul(gate_up)
    raise NotImplementedError(
        "Ascend ModelSlim Dynamic W8A8 MoE supports only silu and gelu "
        f"gated activations, got {activation.value!r}"
    )


def _require_zero_moe_weight_offsets(layer: torch.nn.Module) -> None:
    for name in ("w13_weight_offset", "w2_weight_offset"):
        offset = getattr(layer, name)
        if torch.count_nonzero(offset.detach()).item() != 0:
            raise ValueError(
                f"ModelSlim MoE layer {getattr(layer, 'prefix', '<unknown>')!r} "
                f"uses non-zero {name}, but the Ascend Dynamic W8A8 MoE "
                "kernel supports symmetric weights only"
            )


class AscendModelSlimW8A8DynamicMoEMethod(FusedMoEMethodBase):
    """Dynamic per-token activation and per-channel INT8 expert weights."""

    def __init__(self, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        if not moe.is_act_and_mul:
            raise NotImplementedError(
                "Ascend ModelSlim Dynamic W8A8 MoE requires gated activation"
            )
        if moe.has_bias:
            raise NotImplementedError(
                "Ascend ModelSlim Dynamic W8A8 MoE does not support expert bias"
            )
        if moe.is_lora_enabled:
            raise NotImplementedError(
                "Ascend ModelSlim Dynamic W8A8 MoE does not support LoRA"
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_specs = {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
        }
        for name, tensor in weight_specs.items():
            parameter = torch.nn.Parameter(tensor, requires_grad=False)
            set_weight_attrs(parameter, extra_weight_attrs)
            layer.register_parameter(name, parameter)

        channel_specs = {
            "w13_weight_scale": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=params_dtype,
            ),
            "w13_weight_offset": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=params_dtype,
            ),
            "w2_weight_scale": torch.empty(
                num_experts,
                hidden_size,
                1,
                dtype=params_dtype,
            ),
            "w2_weight_offset": torch.empty(
                num_experts,
                hidden_size,
                1,
                dtype=params_dtype,
            ),
        }
        channel_attrs = dict(extra_weight_attrs)
        channel_attrs["quant_method"] = (
            FusedMoeWeightScaleSupported.CHANNEL.value
        )
        for name, tensor in channel_specs.items():
            parameter = torch.nn.Parameter(tensor, requires_grad=False)
            set_weight_attrs(parameter, channel_attrs)
            layer.register_parameter(name, parameter)

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        _require_zero_moe_weight_offsets(layer)
        layer.w13_weight.data = _npu_format_nz(
            layer.w13_weight.data.transpose(1, 2).contiguous()
        )
        layer.w2_weight.data = _npu_format_nz(
            layer.w2_weight.data.transpose(1, 2).contiguous()
        )
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.flatten(1)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.flatten(1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.flatten(1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.flatten(1)

    def get_fused_moe_quant_config(
        self,
        layer: torch.nn.Module,
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        del routing_tables
        # This method uses vLLM's router and reduction, but owns expert compute.
        return None

    def _apply_expert(
        self,
        layer: FusedMoE,
        expert_idx: int,
        expert_input: torch.Tensor,
    ) -> torch.Tensor:
        quantized_input, input_scale = _npu_dynamic_quant(expert_input)
        gate_up = _npu_quant_matmul(
            quantized_input,
            layer.w13_weight[expert_idx],
            layer.w13_weight_scale[expert_idx],
            pertoken_scale=input_scale.reshape(-1),
            output_dtype=expert_input.dtype,
        )
        activated = _npu_moe_activation(layer.activation, gate_up)
        quantized_hidden, hidden_scale = _npu_dynamic_quant(activated)
        return _npu_quant_matmul(
            quantized_hidden,
            layer.w2_weight[expert_idx],
            layer.w2_weight_scale[expert_idx],
            pertoken_scale=hidden_scale.reshape(-1),
            output_dtype=expert_input.dtype,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts_input
        if x.ndim != 2:
            raise ValueError(
                "Ascend ModelSlim Dynamic W8A8 MoE expects 2D hidden states, "
                f"got shape {tuple(x.shape)}"
            )
        if topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "ModelSlim MoE topk_weights and topk_ids must have equal shapes"
            )

        if layer.expert_map is None:
            local_topk_ids = topk_ids.long()
        else:
            local_topk_ids = layer.expert_map[topk_ids.long()]

        output = x.new_zeros((x.shape[0], layer.w2_weight.shape[-1]))
        num_local_experts = layer.w13_weight.shape[0]
        for expert_idx in range(num_local_experts):
            token_indices, topk_indices = torch.where(
                local_topk_ids == expert_idx
            )
            if token_indices.numel() == 0:
                continue

            expert_input = x.index_select(0, token_indices)
            routing_weight = topk_weights[
                token_indices,
                topk_indices,
            ].unsqueeze(-1)
            if layer.apply_router_weight_on_input:
                expert_input = expert_input * routing_weight.to(x.dtype)

            expert_output = self._apply_expert(
                layer,
                expert_idx,
                expert_input,
            )
            if not layer.apply_router_weight_on_input:
                expert_output = expert_output * routing_weight.to(
                    expert_output.dtype
                )
            output.index_add_(0, token_indices, expert_output)

        return output


__all__ = ["AscendModelSlimW8A8DynamicMoEMethod"]

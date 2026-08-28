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
"""vLLM LinearMethod adapter for Ascend ModelSlim schemes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from .w8a8 import AscendW8A8LinearScheme


class AscendModelSlimLinearMethod(LinearMethodBase):
    """Delegate vLLM's loading lifecycle to one Ascend W8A8 scheme."""

    def __init__(self, scheme: AscendW8A8LinearScheme) -> None:
        self.scheme = scheme

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is None:
            raise ValueError("ModelSlim Linear requires a vLLM weight_loader")

        weight_specs = self.scheme.get_weight(
            input_size_per_partition,
            output_size_per_partition,
            params_dtype,
        )
        for name, tensor in weight_specs.items():
            parameter = torch.nn.Parameter(tensor, requires_grad=False)
            set_weight_attrs(parameter, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(parameter, extra_weight_attrs)
            layer.register_parameter(name, parameter)

        tensor_specs = self.scheme.get_pertensor_params(
            params_dtype,
            num_partitions=len(output_partition_sizes),
        )
        for name, tensor in tensor_specs.items():
            parameter = PerTensorScaleParameter(
                data=tensor,
                weight_loader=weight_loader,
            )
            parameter.ignore_warning = True
            layer.register_parameter(name, parameter)

        channel_specs = self.scheme.get_perchannel_params(
            output_size_per_partition,
            params_dtype,
        )
        for name, tensor in channel_specs.items():
            parameter = torch.nn.Parameter(tensor, requires_grad=False)
            set_weight_attrs(parameter, {"output_dim": 0})
            set_weight_attrs(parameter, extra_weight_attrs)
            layer.register_parameter(name, parameter)

        layer.ascend_quant_method = "ascend"

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tp_rank = (
            get_tensor_model_parallel_rank()
            if isinstance(layer, RowParallelLinear)
            else 0
        )
        return self.scheme.apply(layer, x, bias=bias, tp_rank=tp_rank)

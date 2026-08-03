# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run channelwise pack-quantized INT8 checkpoints as dynamic W8A8."""

from __future__ import annotations

from collections.abc import Callable

import torch

from vllm.model_executor.kernels.linear import init_int8_linear_kernel
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    PackedvLLMParameter,
)

from .int8_mode import (
    INT8_MODE_ENV,
    get_int8_inference_mode,
    is_packed_int8_weight_only,
    should_use_packed_w8a8,
)
from .reference import unpack_uint8b128_int32

_PATCH_MARKER = "_vllm_fl_packed_w8a8_v024"


class FLPackedW8A8Scheme(CompressedTensorsScheme):
    """Load ``weight_packed`` and execute dynamic-token/per-channel W8A8."""

    def __init__(self, layer_name: str | None = None) -> None:
        self.input_size_per_partition = 0
        self.layer_name = layer_name or self.__class__.__name__

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ) -> None:
        del params_dtype, kwargs
        if input_size_per_partition % 4:
            raise ValueError(
                "Packed INT8 requires input_size_per_partition divisible by 4"
            )

        self.kernel = init_int8_linear_kernel(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
            module_name=self.layer_name,
        )
        self.input_size_per_partition = input_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=4,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 4,
                dtype=torch.int32,
            ),
        )
        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                1,
                dtype=torch.float32,
            ),
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)
        layer.register_parameter("input_scale", None)
        layer.register_parameter("input_zero_point", None)
        layer.register_parameter("azp_adj", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        unpacked = unpack_uint8b128_int32(
            layer.weight_packed,
            in_features=self.input_size_per_partition,
        )
        delattr(layer, "weight_packed")
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(unpacked, requires_grad=False),
        )
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


def install_packed_w8a8_scheme() -> bool:
    """Patch vLLM 0.24 scheme selection for channelwise packed INT8 weights."""
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as ct_module,
    )

    config_cls = ct_module.CompressedTensorsConfig
    current = config_cls._get_scheme_from_parts
    if getattr(current, _PATCH_MARKER, False):
        return True

    def get_scheme_from_parts_fl(
        self,
        weight_quant,
        input_quant,
        output_quant=None,
        format=None,
        layer_name=None,
    ):
        effective_format = format if format is not None else self.quant_format
        if should_use_packed_w8a8(
            weight_quant,
            input_quant,
            effective_format,
        ):
            return FLPackedW8A8Scheme(layer_name=layer_name)
        return current(
            self,
            weight_quant,
            input_quant,
            output_quant=output_quant,
            format=format,
            layer_name=layer_name,
        )

    setattr(get_scheme_from_parts_fl, _PATCH_MARKER, True)
    config_cls._get_scheme_from_parts = get_scheme_from_parts_fl
    return True


__all__ = [
    "FLPackedW8A8Scheme",
    "INT8_MODE_ENV",
    "get_int8_inference_mode",
    "install_packed_w8a8_scheme",
    "is_packed_int8_weight_only",
    "should_use_packed_w8a8",
]

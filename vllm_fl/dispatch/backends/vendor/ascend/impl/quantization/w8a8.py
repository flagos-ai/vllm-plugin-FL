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
"""Ascend ModelSlim W8A8 static and dynamic Linear schemes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

_ACL_FORMAT_FRACTAL_NZ = 29


def _npu_format_nz(weight: torch.Tensor) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_format_cast(weight, _ACL_FORMAT_FRACTAL_NZ)


def _npu_static_quant(
    x: torch.Tensor,
    scale_reciprocal: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_quantize(
        x,
        scale_reciprocal,
        offset,
        torch.qint8,
        -1,
        False,
    )


def _npu_dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    return torch_npu.npu_dynamic_quant(x)


def _npu_quant_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_quant_matmul(x, weight, scale, **kwargs)


def _require_zero_weight_offset(layer: torch.nn.Module) -> None:
    offset = getattr(layer, "weight_offset", None)
    if offset is None:
        return
    if torch.count_nonzero(offset.detach()).item() != 0:
        raise ValueError(
            f"ModelSlim layer {getattr(layer, 'prefix', '<unknown>')!r} uses "
            "non-zero weight_offset, but the Ascend W8A8 Linear kernel "
            "supports symmetric weights only"
        )


class AscendW8A8LinearScheme(ABC):
    @abstractmethod
    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]: ...

    def get_pertensor_params(
        self,
        params_dtype: torch.dtype,
        *,
        num_partitions: int,
    ) -> dict[str, torch.Tensor]:
        del params_dtype, num_partitions
        return {}

    @abstractmethod
    def get_perchannel_params(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int = 0,
    ) -> torch.Tensor: ...


class AscendW8A8StaticLinearScheme(AscendW8A8LinearScheme):
    """Static per-tensor activation and per-channel INT8 weight Linear."""

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        del params_dtype
        return {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }

    def get_pertensor_params(
        self,
        params_dtype: torch.dtype,
        *,
        num_partitions: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "input_scale": torch.empty(num_partitions, dtype=params_dtype),
            "input_offset": torch.empty(num_partitions, dtype=torch.int8),
        }

    def get_perchannel_params(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if params_dtype == torch.bfloat16:
            deq_dtype = torch.float32
        elif params_dtype == torch.float16:
            deq_dtype = torch.int64
        else:
            raise ValueError(
                f"Static ModelSlim W8A8 does not support dtype {params_dtype}"
            )
        return {
            "quant_bias": torch.empty(output_size, dtype=torch.int32),
            "deq_scale": torch.empty(output_size, dtype=deq_dtype),
            "weight_scale": torch.empty(
                output_size,
                1,
                dtype=params_dtype,
            ),
            "weight_offset": torch.empty(
                output_size,
                1,
                dtype=params_dtype,
            ),
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        input_scale = layer.input_scale.detach().flatten()
        input_offset = layer.input_offset.detach().flatten()
        if input_scale.numel() > 1 and not torch.allclose(
            input_scale,
            input_scale[0].expand_as(input_scale),
        ):
            raise ValueError(
                f"Packed ModelSlim layer {getattr(layer, 'prefix', '<unknown>')!r} "
                "has different static input scales across shards"
            )
        if input_offset.numel() > 1 and not torch.equal(
            input_offset,
            input_offset[0].expand_as(input_offset),
        ):
            raise ValueError(
                f"Packed ModelSlim layer {getattr(layer, 'prefix', '<unknown>')!r} "
                "has different static input offsets across shards"
            )

        input_size = layer.weight.shape[1]
        scale = input_scale[:1].repeat(input_size)
        offset = input_offset[:1].repeat(input_size).to(scale.dtype)
        layer.aclnn_input_scale = torch.nn.Parameter(scale, requires_grad=False)
        layer.aclnn_input_scale_reciprocal = torch.nn.Parameter(
            scale.reciprocal(),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            offset,
            requires_grad=False,
        )

        _require_zero_weight_offset(layer)
        layer.weight.data = _npu_format_nz(
            layer.weight.data.transpose(0, 1).contiguous()
        )
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int = 0,
    ) -> torch.Tensor:
        del bias
        if x.dtype != torch.int8:
            x = _npu_static_quant(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
            )
        quant_bias = layer.quant_bias if tp_rank == 0 else None
        return _npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=layer.params_dtype,
        )


class AscendW8A8DynamicLinearScheme(AscendW8A8LinearScheme):
    """Dynamic per-token activation and per-channel INT8 weight Linear."""

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        del params_dtype
        return {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }

    def get_perchannel_params(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "weight_scale": torch.empty(
                output_size,
                1,
                dtype=params_dtype,
            ),
            "weight_offset": torch.empty(
                output_size,
                1,
                dtype=params_dtype,
            ),
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _require_zero_weight_offset(layer)
        layer.weight.data = _npu_format_nz(
            layer.weight.data.transpose(0, 1).contiguous()
        )
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.float()
        layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int = 0,
    ) -> torch.Tensor:
        del tp_rank
        output_shape = x.shape[:-1]
        flattened_x = x.reshape(-1, x.shape[-1])
        quantized_x, pertoken_scale = _npu_dynamic_quant(flattened_x)
        output = _npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=pertoken_scale.reshape(-1),
            bias=bias,
            output_dtype=x.dtype,
        )
        return output.reshape(*output_shape, output.shape[-1])


_LINEAR_SCHEMES: dict[str, type[AscendW8A8LinearScheme]] = {
    "W8A8": AscendW8A8StaticLinearScheme,
    "W8A8_DYNAMIC": AscendW8A8DynamicLinearScheme,
}


def get_w8a8_linear_scheme(quant_type: str) -> AscendW8A8LinearScheme:
    try:
        scheme_cls = _LINEAR_SCHEMES[quant_type]
    except KeyError as exc:
        raise NotImplementedError(
            f"No Ascend ModelSlim Linear scheme is registered for {quant_type!r}"
        ) from exc
    return scheme_cls()

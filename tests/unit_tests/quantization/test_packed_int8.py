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

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)

from vllm_fl.quantization.w8a8 import packed
from vllm_fl.quantization.w8a8.int8_mode import should_use_packed_w8a8


def _weight_args(strategy: QuantizationStrategy) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=strategy,
        symmetric=True,
        dynamic=False,
        group_size=128 if strategy == QuantizationStrategy.GROUP else None,
    )


def _activation_args(
    *,
    strategy: QuantizationStrategy = QuantizationStrategy.TOKEN,
    dynamic: bool = True,
) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=strategy,
        symmetric=True,
        dynamic=dynamic,
    )


def test_model_config_maps_packed_channelwise_w8a8():
    assert should_use_packed_w8a8(
        _weight_args(QuantizationStrategy.CHANNEL),
        _activation_args(),
        "pack-quantized",
    )


def test_missing_activation_config_keeps_weight_only_scheme():
    assert not should_use_packed_w8a8(
        _weight_args(QuantizationStrategy.CHANNEL),
        None,
        "pack-quantized",
    )


def test_packed_w8a8_rejects_groupwise_weights():
    with pytest.raises(ValueError, match="per-channel"):
        should_use_packed_w8a8(
            _weight_args(QuantizationStrategy.GROUP),
            _activation_args(),
            "pack-quantized",
        )


def test_static_tensor_activation_config_does_not_select_dynamic_w8a8():
    assert not should_use_packed_w8a8(
        _weight_args(QuantizationStrategy.CHANNEL),
        _activation_args(
            strategy=QuantizationStrategy.TENSOR,
            dynamic=False,
        ),
        "pack-quantized",
    )


def test_packed_scheme_reuses_native_w8a8_execution(monkeypatch):
    processed_layers = []

    class FakeKernel:
        def process_weights_after_loading(self, layer):
            processed_layers.append(layer)

        def apply_weights(self, layer, x, bias):
            raise AssertionError("not used")

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_int8.init_int8_linear_kernel",
        lambda *args, **kwargs: FakeKernel(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    scheme = packed.CompressedTensorsPackedW8A8Int8(layer_name="model.linear")
    assert isinstance(scheme, CompressedTensorsW8A8Int8)

    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=[4, 4],
        input_size_per_partition=8,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.logical_widths == [4, 4]
    assert not hasattr(layer, "weight")
    assert layer.weight_packed.shape == (8, 2)
    assert layer.weight_scale.shape == (8, 1)
    assert layer.weight_scale.dtype == torch.float32

    unpacked = torch.ones((8, 8), dtype=torch.int8)
    monkeypatch.setattr(
        packed,
        "unpack_uint8b128_int32",
        lambda *args, **kwargs: unpacked,
    )
    scheme.process_weights_after_loading(layer)

    assert not hasattr(layer, "weight_packed")
    assert torch.equal(layer.weight, unpacked)
    assert processed_layers == [layer]

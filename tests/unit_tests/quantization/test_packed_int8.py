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

from vllm_fl.quantization.w8a8 import packed
from vllm_fl.quantization.w8a8.int8_mode import (
    INT8_MODE_ENV,
    should_use_packed_w8a8,
)


def _weight_args(strategy: QuantizationStrategy) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=strategy,
        symmetric=True,
        dynamic=False,
        group_size=128 if strategy == QuantizationStrategy.GROUP else None,
    )


def test_auto_mode_maps_channelwise_packed_int8_to_w8a8(monkeypatch):
    monkeypatch.delenv(INT8_MODE_ENV, raising=False)
    assert should_use_packed_w8a8(
        _weight_args(QuantizationStrategy.CHANNEL),
        None,
        "pack-quantized",
    )
    assert not should_use_packed_w8a8(
        _weight_args(QuantizationStrategy.GROUP),
        None,
        "pack-quantized",
    )


def test_w8a16_mode_keeps_channelwise_checkpoint_weight_only(monkeypatch):
    monkeypatch.setenv(INT8_MODE_ENV, "w8a16")
    assert not should_use_packed_w8a8(
        _weight_args(QuantizationStrategy.CHANNEL),
        None,
        "pack-quantized",
    )


def test_w8a8_mode_rejects_groupwise_checkpoint(monkeypatch):
    monkeypatch.setenv(INT8_MODE_ENV, "w8a8")
    with pytest.raises(ValueError, match="--strategy channel"):
        should_use_packed_w8a8(
            _weight_args(QuantizationStrategy.GROUP),
            None,
            "pack-quantized",
        )


def test_packed_scheme_matches_vllm_024_layer_contract(monkeypatch):
    class FakeKernel:
        def process_weights_after_loading(self, layer):
            assert layer.weight.dtype == torch.int8

        def apply_weights(self, layer, x, bias):
            raise AssertionError("not used")

    monkeypatch.setattr(
        packed,
        "init_int8_linear_kernel",
        lambda **kwargs: FakeKernel(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    scheme = packed.FLPackedW8A8Scheme(layer_name="model.linear")
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=[4, 4],
        input_size_per_partition=8,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.logical_widths == [4, 4]
    assert layer.weight_packed.shape == (8, 2)
    assert layer.weight_scale.shape == (8, 1)
    assert layer.weight_scale.dtype == torch.float32

    values = torch.arange(-32, 32, dtype=torch.int8).reshape(8, 8)
    layer.weight_packed.data.copy_(
        (values.to(torch.int16) + 128).to(torch.uint8).contiguous().view(torch.int32)
    )
    scheme.process_weights_after_loading(layer)

    assert not hasattr(layer, "weight_packed")
    assert torch.equal(layer.weight, values)

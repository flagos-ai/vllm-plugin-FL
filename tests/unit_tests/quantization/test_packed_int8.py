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
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

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

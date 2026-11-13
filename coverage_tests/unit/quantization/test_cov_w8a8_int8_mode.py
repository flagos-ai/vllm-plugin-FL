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

"""CPU tests for the packed INT8 checkpoint selection policy.

The policy reads compressed-tensors metadata only.  The stand-in metadata
classes live in ``_w8a8_fakes`` so the test does not depend on the image's
compressed-tensors version.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

# The real package must load before the fake import window below;
# otherwise the first import of vllm_fl happens under the fakes.
import vllm_fl  # noqa: F401
from ._w8a8_fakes import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    build_ct_environment,
    import_under_fakes,
)

PACKED = "pack-quantized"


def _args(**changes) -> QuantizationArgs:
    base = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        symmetric=True,
        dynamic=False,
        strategy=QuantizationStrategy.CHANNEL,
    )
    return replace(base, **changes)


@pytest.fixture
def int8_mode():
    env = build_ct_environment()
    with import_under_fakes(
        "vllm_fl.quantization.w8a8.int8_mode",
        env.modules,
        purge=("vllm_fl.quantization.w8a8.int8_mode",),
    ) as module:
        yield module


def test_packed_int8_weight_accepts_channel_and_group(int8_mode):
    """Verifies packed int8 weights accept both channel and group strategies."""
    assert int8_mode.is_packed_int8_weight(_args(), PACKED) is True
    assert (
        int8_mode.is_packed_int8_weight(
            _args(strategy=QuantizationStrategy.GROUP, group_size=128), PACKED
        )
        is True
    )


def test_packed_int8_weight_rejects_none_and_wrong_format(int8_mode):
    """Verifies None args and a wrong quant format are rejected as not packed int8."""
    assert int8_mode.is_packed_int8_weight(None, PACKED) is False
    assert int8_mode.is_packed_int8_weight(_args(), "int-quantized") is False


@pytest.mark.parametrize(
    "changes",
    [
        {"num_bits": 4},
        {"type": QuantizationType.FLOAT},
        {"symmetric": False},
        {"dynamic": True},
        {"strategy": QuantizationStrategy.TOKEN},
    ],
)
def test_packed_int8_weight_rejects_non_packed_metadata(int8_mode, changes):
    """Verifies non-packed metadata attributes are rejected as packed int8 weights."""
    assert int8_mode.is_packed_int8_weight(_args(**changes), PACKED) is False


def test_dynamic_token_int8_accepts_canonical_metadata(int8_mode):
    """Verifies canonical token-wise dynamic int8 metadata is accepted."""
    args = _args(strategy=QuantizationStrategy.TOKEN, dynamic=True)
    assert int8_mode.is_dynamic_token_int8(args) is True


@pytest.mark.parametrize(
    "changes",
    [
        {"num_bits": 4},
        {"type": QuantizationType.FLOAT},
        {"strategy": QuantizationStrategy.CHANNEL},
        {"symmetric": False},
        {"dynamic": False},
    ],
)
def test_dynamic_token_int8_rejects_other_metadata(int8_mode, changes):
    """Verifies other metadata variants and None are rejected as dynamic token int8."""
    args = _args(strategy=QuantizationStrategy.TOKEN, dynamic=True)
    assert int8_mode.is_dynamic_token_int8(replace(args, **changes)) is False
    assert int8_mode.is_dynamic_token_int8(None) is False


def test_should_use_packed_w8a8_selects_channel_weights(int8_mode):
    """Verifies channel weights with dynamic token activations select packed W8A8."""
    weight = _args()
    activation = _args(strategy=QuantizationStrategy.TOKEN, dynamic=True)
    assert int8_mode.should_use_packed_w8a8(weight, activation, PACKED) is True


@pytest.mark.parametrize(
    ("weight", "activation", "quant_format"),
    [
        (None, _args(strategy=QuantizationStrategy.TOKEN, dynamic=True), PACKED),
        (
            _args(),
            None,
            PACKED,
        ),
        (
            _args(),
            _args(strategy=QuantizationStrategy.TOKEN, dynamic=False),
            PACKED,
        ),
        (
            _args(),
            _args(strategy=QuantizationStrategy.TOKEN, dynamic=True),
            "int-quantized",
        ),
    ],
)
def test_should_use_packed_w8a8_rejects_incomplete_config(
    int8_mode, weight, activation, quant_format
):
    """Verifies incomplete weight, activation or format configs are rejected."""
    assert int8_mode.should_use_packed_w8a8(weight, activation, quant_format) is False


def test_should_use_packed_w8a8_rejects_group_strategy_with_value_failure(int8_mode):
    """Verifies a group weight strategy raises ValueError requiring per-channel weights."""
    weight = _args(strategy=QuantizationStrategy.GROUP, group_size=128)
    activation = _args(strategy=QuantizationStrategy.TOKEN, dynamic=True)
    with pytest.raises(ValueError, match="requires per-channel weights"):
        int8_mode.should_use_packed_w8a8(weight, activation, PACKED)

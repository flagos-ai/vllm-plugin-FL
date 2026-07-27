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
from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearLayerConfig,
)
from vllm.platforms import PlatformEnum

from vllm_fl.quantization.w8a8 import linear


@pytest.mark.parametrize(
    (
        "channelwise",
        "static_input",
        "input_symmetric",
        "expected",
        "message",
    ),
    [
        (True, False, True, True, None),
        (False, False, True, False, "per-channel"),
        (True, True, True, False, "dynamic"),
        (True, False, False, False, "symmetric"),
    ],
)
def test_w8a8_linear_accepts_only_canonical_dynamic_token_scheme(
    channelwise,
    static_input,
    input_symmetric,
    expected,
    message,
):
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=channelwise,
        is_static_input_scheme=static_input,
        input_symmetric=input_symmetric,
    )
    supported, reason = linear.FLW8A8DynamicLinearKernel.can_implement(config)
    assert supported is expected
    if message is not None:
        assert message in reason


def test_w8a8_linear_registration_is_idempotent(monkeypatch):
    monkeypatch.setattr(linear, "_flaggems_available", lambda: True)
    registry = {PlatformEnum.OOT: []}
    assert linear.register_fl_w8a8_linear_kernel(registry) is True
    assert linear.register_fl_w8a8_linear_kernel(registry) is True
    assert registry[PlatformEnum.OOT] == [
        linear.FLW8A8DynamicLinearKernel
    ]


def test_w8a8_linear_is_not_registered_without_flaggems(monkeypatch):
    monkeypatch.setattr(linear, "_flaggems_available", lambda: False)
    registry = {PlatformEnum.OOT: []}
    assert linear.register_fl_w8a8_linear_kernel(registry) is False
    assert registry[PlatformEnum.OOT] == []

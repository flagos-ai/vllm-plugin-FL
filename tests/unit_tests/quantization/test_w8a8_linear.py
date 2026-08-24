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

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearLayerConfig,
)
from vllm.platforms import PlatformEnum

from vllm_fl.quantization.w8a8 import linear
from vllm_fl.quantization.w8a8.reference import (
    dynamic_per_token_quant_int8,
    w8a8_linear_reference,
)


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


def test_w8a8_linear_registration_is_non_nvidia_and_idempotent(monkeypatch):
    monkeypatch.setattr(linear, "is_nvidia_platform", lambda: False)
    monkeypatch.setattr(linear, "_flaggems_available", lambda: True)
    registry = {PlatformEnum.OOT: []}

    assert linear.register_fl_w8a8_linear_kernel(registry) is True
    assert linear.register_fl_w8a8_linear_kernel(registry) is True
    assert registry[PlatformEnum.OOT] == [linear.FLW8A8DynamicLinearKernel]


def test_w8a8_linear_is_not_registered_on_nvidia(monkeypatch):
    monkeypatch.setattr(linear, "is_nvidia_platform", lambda: True)
    monkeypatch.setattr(linear, "_flaggems_available", lambda: True)
    registry = {PlatformEnum.OOT: []}

    assert linear.register_fl_w8a8_linear_kernel(registry) is False
    assert registry[PlatformEnum.OOT] == []


def test_w8a8_linear_nvidia_guard_precedes_flaggems_policy(monkeypatch):
    monkeypatch.setattr(linear, "is_nvidia_platform", lambda: True)
    monkeypatch.setattr(
        linear,
        "use_flaggems_op",
        lambda op_name: pytest.fail(f"policy must not be queried for {op_name}"),
    )

    supported, reason = linear.FLW8A8DynamicLinearKernel.is_supported()

    assert supported is False
    assert "NVIDIA" in reason


def test_w8a8_linear_rejects_flaggems_without_returning_scaled_mm(
    monkeypatch,
):
    flag_gems = ModuleType("flag_gems")
    monkeypatch.delitem(sys.modules, "flag_gems.ops.scaled_mm", raising=False)
    monkeypatch.delitem(sys.modules, "flag_gems.ops", raising=False)
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setattr(linear, "find_spec", lambda name: object())

    assert linear._flaggems_available() is False


def test_w8a8_linear_quantizes_then_calls_flaggems_scaled_mm(monkeypatch):
    calls = []
    x = torch.ones((2, 4), dtype=torch.bfloat16)
    x_q = torch.ones((2, 4), dtype=torch.int8)
    x_scale = torch.full((2, 1), 0.25, dtype=torch.float32)
    weight = torch.ones((4, 3), dtype=torch.int8)
    weight_scale = torch.ones(3, dtype=torch.float32)
    expected = torch.full((2, 3), 2, dtype=torch.bfloat16)

    monkeypatch.setattr(
        linear,
        "_dynamic_per_token_quant_int8",
        lambda value: (x_q, x_scale),
    )
    flag_gems = ModuleType("flag_gems")

    def fake_scaled_mm(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    flag_gems.scaled_mm = fake_scaled_mm
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)

    kernel = object.__new__(linear.FLW8A8DynamicLinearKernel)
    kernel.layer_param_names = [
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    ]
    layer = SimpleNamespace(
        weight=weight,
        weight_scale=weight_scale,
        input_scale=None,
        input_zero_point=None,
        azp_adj=None,
    )

    output = kernel.apply_weights(layer, x)

    assert torch.equal(output, expected)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is x_q
    assert args[1] is weight
    assert args[2] is x_scale
    assert args[3] is weight_scale
    assert kwargs == {
        "bias": None,
        "out_dtype": torch.bfloat16,
    }


def test_w8a8_linear_weight_layout_and_numerics_match_flaggems_contract(
    monkeypatch,
):
    checkpoint_weight = torch.tensor(
        [
            [12, -7, 3, 9],
            [-5, 11, -13, 2],
            [4, 6, -8, 10],
        ],
        dtype=torch.int8,
    )
    checkpoint_scale = torch.tensor(
        [[0.01], [0.025], [0.04]],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [[[1.0, -2.0, 3.0, -4.0], [0.5, 0.25, -0.75, 1.25]]],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.5, -0.25, 0.125], dtype=torch.float32)
    expected = w8a8_linear_reference(
        x,
        checkpoint_weight,
        checkpoint_scale,
        bias,
    )

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(checkpoint_weight.clone(), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(checkpoint_scale.clone(), requires_grad=False),
    )
    layer.register_parameter("input_scale", None)
    layer.register_parameter("input_zero_point", None)
    layer.register_parameter("azp_adj", None)

    kernel = object.__new__(linear.FLW8A8DynamicLinearKernel)
    kernel.layer_param_names = [
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    ]
    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 3)
    assert layer.weight.stride() == (3, 1)
    assert torch.equal(layer.weight, checkpoint_weight.t().contiguous())
    assert layer.weight_scale.shape == (3,)
    assert layer.weight_scale.dtype == torch.float32
    assert torch.equal(layer.weight_scale, checkpoint_scale.reshape(-1))

    monkeypatch.setattr(
        linear,
        "_dynamic_per_token_quant_int8",
        dynamic_per_token_quant_int8,
    )
    flag_gems = ModuleType("flag_gems")

    def fake_scaled_mm(
        x_q,
        weight,
        x_scale,
        weight_scale,
        *,
        bias,
        out_dtype,
    ):
        assert x_q.shape == (2, 4)
        assert x_scale.shape == (2, 1)
        assert weight.shape == (4, 3)
        assert weight_scale.shape == (3,)
        accumulator = x_q.to(torch.int32) @ weight.to(torch.int32)
        output = accumulator.to(torch.float32) * x_scale * weight_scale.reshape(1, -1)
        if bias is not None:
            output += bias.reshape(1, -1)
        return output.to(out_dtype)

    flag_gems.scaled_mm = fake_scaled_mm
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)

    actual = kernel.apply_weights(layer, x, bias)

    assert actual.shape == (1, 2, 3)
    assert torch.equal(actual, expected)

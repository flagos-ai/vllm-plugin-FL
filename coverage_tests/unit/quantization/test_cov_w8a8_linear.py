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

"""CPU tests for the FlagGems dynamic-token W8A8 linear kernel.

The vLLM INT8 kernel base class, ``replace_parameter``, ``CachedOp`` and
``flag_gems.scaled_mm`` are stand-ins from ``_w8a8_fakes``.  Weight layout
conversion, the dynamic quant call and the scaled-mm argument marshalling
are all asserted, with the returned tensor re-derived from the recorded
integer operands.
"""

from __future__ import annotations

import pytest
import torch

from ._w8a8_fakes import build_linear_environment, import_under_fakes

MODULE = "vllm_fl.quantization.w8a8.linear"


@pytest.fixture
def linear():
    env = build_linear_environment()
    with import_under_fakes(MODULE, env.modules, purge=(MODULE,)) as module:
        yield module, env


def _layer(
    *,
    weight=None,
    weight_scale=None,
    input_scale=None,
    input_zp=None,
    azp_adj=None,
):
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.randint(-8, 8, (4, 8), dtype=torch.int8)
            if weight is None
            else weight,
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            torch.ones(4, dtype=torch.float16)
            if weight_scale is None
            else weight_scale,
            requires_grad=False,
        ),
    )
    layer.input_scale = input_scale
    layer.input_zp = input_zp
    layer.azp_adj = azp_adj
    return layer


def test_is_supported_follows_oot_and_policy_gates(linear):
    module, env = linear
    assert module.FLW8A8DynamicLinearKernel.is_supported(90) == (True, None)

    env.oot_enabled = False
    assert module.FLW8A8DynamicLinearKernel.is_supported() == (
        False,
        "FL W8A8 kernels are disabled",
    )

    env.oot_enabled = True
    env.flaggems_ops = {module.FLAGGEMS_W8A8_LINEAR_OP: False}
    assert module.FLW8A8DynamicLinearKernel.is_supported() == (
        False,
        "FlagGems W8A8 linear is disabled by policy",
    )


def test_can_implement_requires_dynamic_symmetric_channelwise(linear):
    module, env = linear
    config_cls = env.Int8ScaledMMLinearLayerConfig
    assert module.FLW8A8DynamicLinearKernel.can_implement(config_cls()) == (
        True,
        None,
    )
    assert module.FLW8A8DynamicLinearKernel.can_implement(
        config_cls(is_channelwise=False)
    ) == (False, "requires per-channel weights")
    assert module.FLW8A8DynamicLinearKernel.can_implement(
        config_cls(is_static_input_scheme=True)
    ) == (False, "requires dynamic input quantization")
    assert module.FLW8A8DynamicLinearKernel.can_implement(
        config_cls(input_symmetric=False)
    ) == (False, "requires symmetric INT8 activations")


def test_process_weights_transposes_and_normalizes(linear):
    module, env = linear
    kernel = module.FLW8A8DynamicLinearKernel()
    weight = torch.arange(32, dtype=torch.int8).reshape(4, 8)
    scale = torch.linspace(0.5, 2.0, 4, dtype=torch.float16)
    layer = _layer(
        weight=weight,
        weight_scale=scale,
        input_scale=torch.ones(8),
        input_zp=torch.zeros(8),
        azp_adj=torch.ones(8),
    )

    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (8, 4)
    assert layer.weight.dtype == torch.int8
    assert layer.weight.is_contiguous()
    assert torch.equal(layer.weight, weight.t())
    assert layer.weight_scale.dtype == torch.float32
    assert layer.weight_scale.shape == (4,)
    assert layer.weight_scale.is_contiguous()
    assert torch.equal(layer.weight_scale, scale.to(torch.float32))
    assert layer.input_scale is None
    assert layer.input_zp is None
    assert layer.azp_adj is None
    assert [(name, param.shape) for _, name, param in env.replace_calls] == [
        ("weight", (8, 4)),
        ("weight_scale", (4,)),
    ]


@pytest.mark.parametrize(
    ("weight", "weight_scale", "match"),
    [
        (torch.zeros(8, dtype=torch.int8), torch.ones(1), "2D int8"),
        (torch.zeros(4, 8, dtype=torch.float32), torch.ones(4), "2D int8"),
        (torch.zeros(4, 8, dtype=torch.int8), torch.ones(3), "one scale"),
    ],
)
def test_process_weights_rejects_bad_checkpoint(linear, weight, weight_scale, match):
    module, _ = linear
    kernel = module.FLW8A8DynamicLinearKernel()
    layer = _layer(weight=weight, weight_scale=weight_scale)
    with pytest.raises(ValueError, match=match):
        kernel.process_weights_after_loading(layer)


def test_apply_weights_quantizes_and_calls_scaled_mm(linear):
    module, env = linear
    kernel = module.FLW8A8DynamicLinearKernel()
    weight = torch.randint(-8, 8, (4, 8), dtype=torch.int8)
    layer = _layer(weight=weight)
    kernel.process_weights_after_loading(layer)

    x = torch.randn(2, 3, 8, dtype=torch.float16)
    x_q = torch.randint(-8, 8, (6, 8), dtype=torch.int8)
    x_scale = torch.rand(6, 1)
    env.cached_op.result = (x_q, x_scale)
    expected = (
        (x_q.to(torch.float32) @ layer.weight.to(torch.float32))
        * x_scale
        * layer.weight_scale.reshape(1, -1)
    )
    env.scaled_mm_result = expected.to(torch.float16)
    bias = torch.randn(4, dtype=torch.float16)

    out = kernel.apply_weights(layer, x, bias)

    assert env.cached_op.op_name == "dynamic_per_token_quant_int8"
    quant_args = env.cached_op.calls[0][0]
    assert quant_args[0].shape == (6, 8)
    assert quant_args[0].is_contiguous()

    args, kwargs = env.scaled_mm_calls[0]
    assert args[0] is x_q
    assert args[1] is layer.weight
    assert args[2] is x_scale
    assert args[3] is layer.weight_scale
    assert kwargs == {"bias": bias, "out_dtype": torch.float16}
    assert out.shape == (2, 3, 4)
    assert torch.equal(out, env.scaled_mm_result.reshape(2, 3, 4))


def test_register_prepends_kernel_and_keeps_duplicates_out(linear):
    module, env = linear
    registry = {}
    assert module.register_fl_w8a8_linear_kernel(registry) is True
    candidates = registry[env.PlatformEnum.OOT]
    assert candidates == [module.FLW8A8DynamicLinearKernel]

    existing = object()
    registry = {env.PlatformEnum.OOT: [existing]}
    module.register_fl_w8a8_linear_kernel(registry)
    assert registry[env.PlatformEnum.OOT] == [
        module.FLW8A8DynamicLinearKernel,
        existing,
    ]

    module.register_fl_w8a8_linear_kernel(registry)
    assert registry[env.PlatformEnum.OOT].count(module.FLW8A8DynamicLinearKernel) == 1

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


def _dynamic_channel_config():
    return Int8ScaledMMLinearLayerConfig(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
    )


def _set_flaggems_support(monkeypatch, supported: bool, reason: str | None = None):
    monkeypatch.setattr(
        linear.FLW8A8DynamicLinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (supported, reason)),
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


def test_w8a8_linear_registration_is_idempotent(monkeypatch):
    monkeypatch.setattr(linear, "_flaggems_available", lambda: True)
    registry = {PlatformEnum.OOT: []}
    assert linear.register_fl_w8a8_linear_kernel(registry) is True
    assert linear.register_fl_w8a8_linear_kernel(registry) is True
    assert registry[PlatformEnum.OOT] == [linear.FLW8A8DynamicLinearKernel]


def test_w8a8_linear_is_not_registered_without_flaggems(monkeypatch):
    monkeypatch.setattr(linear, "_flaggems_available", lambda: False)
    registry = {PlatformEnum.OOT: []}
    assert linear.register_fl_w8a8_linear_kernel(registry) is False
    assert registry[PlatformEnum.OOT] == []


@pytest.mark.parametrize("backend", ["auto", "flaggems", "vllm"])
def test_w8a8_linear_backend_env(monkeypatch, backend):
    monkeypatch.setenv(linear.W8A8_LINEAR_BACKEND_ENV, backend.upper())
    assert linear.get_w8a8_linear_backend() == backend


def test_w8a8_linear_backend_env_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(linear.W8A8_LINEAR_BACKEND_ENV, "unknown")
    with pytest.raises(ValueError, match=linear.W8A8_LINEAR_BACKEND_ENV):
        linear.get_w8a8_linear_backend()


def test_auto_backend_preserves_flaggems_priority(monkeypatch):
    monkeypatch.delenv(linear.W8A8_LINEAR_BACKEND_ENV, raising=False)
    _set_flaggems_support(monkeypatch, True)

    def unexpected_vllm(*args, **kwargs):
        raise AssertionError("native vLLM selector must not run")

    monkeypatch.setattr(
        linear,
        "_create_vllm_w8a8_linear_kernel",
        unexpected_vllm,
    )
    kernel = linear.create_w8a8_linear_kernel(
        _dynamic_channel_config(),
        ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"],
        "test",
    )
    assert isinstance(kernel, linear.FLW8A8DynamicLinearKernel)


@pytest.mark.parametrize("backend", ["auto", "vllm"])
def test_native_vllm_backend_can_be_selected(monkeypatch, backend):
    monkeypatch.setenv(linear.W8A8_LINEAR_BACKEND_ENV, backend)
    _set_flaggems_support(monkeypatch, False, "disabled")
    sentinel = object()
    monkeypatch.setattr(
        linear,
        "_create_vllm_w8a8_linear_kernel",
        lambda *args, **kwargs: sentinel,
    )

    kernel = linear.create_w8a8_linear_kernel(
        _dynamic_channel_config(),
        ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"],
        "test",
    )
    assert kernel is sentinel


def test_explicit_flaggems_backend_does_not_fall_through(monkeypatch):
    monkeypatch.setenv(linear.W8A8_LINEAR_BACKEND_ENV, "flaggems")
    _set_flaggems_support(monkeypatch, False, "disabled")

    def unexpected_vllm(*args, **kwargs):
        raise AssertionError("native vLLM selector must not run")

    monkeypatch.setattr(
        linear,
        "_create_vllm_w8a8_linear_kernel",
        unexpected_vllm,
    )
    with pytest.raises(RuntimeError, match="disabled"):
        linear.create_w8a8_linear_kernel(
            _dynamic_channel_config(),
            [
                "weight",
                "weight_scale",
                "input_scale",
                "input_zero_point",
                "azp_adj",
            ],
            "test",
        )


def test_native_selector_excludes_fl_adapter(monkeypatch):
    import vllm.model_executor.kernels.linear as linear_module
    from vllm.platforms import current_platform

    class NativeKernel:
        def __init__(self, config, layer_param_names):
            self.config = config
            self.layer_param_names = layer_param_names

    monkeypatch.setitem(
        linear_module._POSSIBLE_INT8_KERNELS,
        current_platform._enum,
        [linear.FLW8A8DynamicLinearKernel, NativeKernel],
    )

    def choose_kernel(config, candidates):
        assert candidates[current_platform._enum] == [NativeKernel]
        return NativeKernel

    monkeypatch.setattr(
        linear_module,
        "choose_scaled_mm_linear_kernel",
        choose_kernel,
    )
    names = [
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    ]
    kernel = linear._create_vllm_w8a8_linear_kernel(
        _dynamic_channel_config(),
        names,
        "test",
    )
    assert isinstance(kernel, NativeKernel)
    assert kernel.layer_param_names == names

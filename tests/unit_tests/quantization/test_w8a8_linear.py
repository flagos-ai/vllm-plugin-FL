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

import pytest

from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearLayerConfig,
)

from vllm_fl.quantization.w8a8 import linear


def _dynamic_channel_config():
    return Int8ScaledMMLinearLayerConfig(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
    )


def _layer_param_names():
    return [
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    ]


def test_w8a8_linear_uses_native_vllm_selector(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        linear,
        "_create_vllm_w8a8_linear_kernel",
        lambda *args, **kwargs: sentinel,
    )

    kernel = linear.create_w8a8_linear_kernel(
        _dynamic_channel_config(),
        _layer_param_names(),
        "test",
    )

    assert kernel is sentinel


def test_native_selector_uses_current_platform_candidates(monkeypatch):
    import vllm.model_executor.kernels.linear as linear_module
    from vllm.platforms import current_platform

    class NativeKernel:
        def __init__(self, config, layer_param_names):
            self.config = config
            self.layer_param_names = layer_param_names

    monkeypatch.setitem(
        linear_module._POSSIBLE_INT8_KERNELS,
        current_platform._enum,
        [NativeKernel],
    )

    def choose_kernel(config, candidates):
        assert candidates[current_platform._enum] == [NativeKernel]
        return NativeKernel

    monkeypatch.setattr(
        linear_module,
        "choose_scaled_mm_linear_kernel",
        choose_kernel,
    )

    kernel = linear._create_vllm_w8a8_linear_kernel(
        _dynamic_channel_config(),
        _layer_param_names(),
        "test",
    )

    assert isinstance(kernel, NativeKernel)
    assert kernel.layer_param_names == _layer_param_names()


def test_native_selector_rejects_missing_candidates(monkeypatch):
    import vllm.model_executor.kernels.linear as linear_module
    from vllm.platforms import current_platform

    monkeypatch.setitem(
        linear_module._POSSIBLE_INT8_KERNELS,
        current_platform._enum,
        [],
    )

    with pytest.raises(RuntimeError, match="no native INT8"):
        linear.create_w8a8_linear_kernel(
            _dynamic_channel_config(),
            _layer_param_names(),
            "test",
        )


def test_native_selector_reports_unsupported_platform(monkeypatch):
    import vllm.model_executor.kernels.linear as linear_module
    from vllm.platforms import current_platform

    class NativeKernel:
        pass

    monkeypatch.setitem(
        linear_module._POSSIBLE_INT8_KERNELS,
        current_platform._enum,
        [NativeKernel],
    )
    monkeypatch.setattr(
        linear_module,
        "choose_scaled_mm_linear_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unsupported")),
    )

    with pytest.raises(RuntimeError, match="no native vLLM INT8"):
        linear.create_w8a8_linear_kernel(
            _dynamic_channel_config(),
            _layer_param_names(),
            "test",
        )

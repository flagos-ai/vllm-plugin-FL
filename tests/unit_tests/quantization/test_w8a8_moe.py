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

import sys
from importlib import import_module
from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.quantization.w8a8 import moe as moe_adapter


def _mock_platform(monkeypatch, *, cuda, oot=False):
    # Patch the exact module attribute read by the selector, independently of
    # vendor class overrides and prior platform import-isolation tests.
    platforms = import_module("vllm.platforms")
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: cuda, is_out_of_tree=lambda: oot),
    )


def _install_with_fake_modules(monkeypatch, upstream_selector, upstream_builder):
    oracle = SimpleNamespace(
        Int8MoeBackend=SimpleNamespace(TRITON="triton"),
        select_int8_moe_backend=upstream_selector,
    )
    scheme = SimpleNamespace(
        select_int8_moe_backend=upstream_selector,
        make_int8_moe_quant_config=upstream_builder,
    )
    modules = {
        moe_adapter._ORACLE_MODULE: oracle,
        moe_adapter._SCHEME_MODULE: scheme,
    }
    monkeypatch.setattr(moe_adapter, "import_module", lambda name: modules[name])
    return oracle, scheme


def test_w8a8_moe_builder_preserves_dynamic_per_token_config(monkeypatch):
    def upstream_selector(*args, **kwargs):
        return "upstream"

    def upstream_builder(*args, **kwargs):
        return "upstream-config"

    oracle, scheme = _install_with_fake_modules(
        monkeypatch,
        upstream_selector,
        upstream_builder,
    )
    assert moe_adapter.install_fl_w8a8_moe_selector()
    assert oracle.select_int8_moe_backend is scheme.select_int8_moe_backend

    config_module = SimpleNamespace(
        int8_w8a8_moe_quant_config=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.config",
        config_module,
    )
    dynamic_config = scheme.make_int8_moe_quant_config(
        w1_scale="w1",
        w2_scale="w2",
        a1_scale=None,
        a2_scale=None,
        w1_bias="b1",
        w2_bias="b2",
        per_act_token_quant=True,
    )
    assert dynamic_config == {
        "w1_scale": "w1",
        "w2_scale": "w2",
        "a1_scale": None,
        "a2_scale": None,
        "w1_bias": "b1",
        "w2_bias": "b2",
        "per_act_token_quant": True,
    }
    assert (
        scheme.make_int8_moe_quant_config(
            w1_scale="w1",
            w2_scale="w2",
            per_act_token_quant=False,
        )
        == "upstream-config"
    )


def test_w8a8_moe_selector_uses_fl_experts_on_non_nvidia_oot(monkeypatch):
    upstream_calls = []

    def upstream_selector(*args, **kwargs):
        upstream_calls.append((args, kwargs))
        return "upstream"

    oracle, _ = _install_with_fake_modules(
        monkeypatch,
        upstream_selector,
        lambda **kwargs: kwargs,
    )
    _mock_platform(monkeypatch, cuda=False, oot=True)

    fl_utils = import_module("vllm_fl.utils")

    monkeypatch.setattr(fl_utils, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(fl_utils, "use_flaggems_op", lambda op_name: True)
    moe_adapter.install_fl_w8a8_moe_selector()
    config = SimpleNamespace(
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=False,
        ),
    )

    backend, experts_cls = oracle.select_int8_moe_backend(
        config,
        weight_key=None,
        activation_key=None,
    )

    from vllm_fl.quantization.w8a8.moe_experts import FlagGemsW8A8Experts

    assert backend == "triton"
    assert experts_cls is FlagGemsW8A8Experts
    assert upstream_calls == []


def test_w8a8_moe_selector_uses_flaggems_on_nvidia(monkeypatch):
    upstream_calls = []

    def upstream_selector(*args, **kwargs):
        upstream_calls.append((args, kwargs))
        return "nvidia-native"

    oracle, _ = _install_with_fake_modules(
        monkeypatch,
        upstream_selector,
        lambda **kwargs: kwargs,
    )

    fl_utils = import_module("vllm_fl.utils")

    monkeypatch.setattr(fl_utils, "is_oot_enabled", lambda: True)
    _mock_platform(monkeypatch, cuda=True)
    monkeypatch.setattr(
        fl_utils,
        "use_flaggems_op",
        lambda op_name: True,
    )
    moe_adapter.install_fl_w8a8_moe_selector()
    config = SimpleNamespace(
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=False,
        ),
    )

    backend, experts_cls = oracle.select_int8_moe_backend(
        config,
        weight_key=None,
        activation_key=None,
    )

    from vllm_fl.quantization.w8a8.moe_experts import FlagGemsW8A8Experts

    assert backend == "triton"
    assert experts_cls is FlagGemsW8A8Experts
    assert upstream_calls == []


def test_w8a8_moe_selector_nvidia_policy_disable_uses_native_fallback(monkeypatch):
    upstream_calls = []

    def upstream_selector(*args, **kwargs):
        upstream_calls.append((args, kwargs))
        return "nvidia-native"

    oracle, _ = _install_with_fake_modules(
        monkeypatch,
        upstream_selector,
        lambda **kwargs: kwargs,
    )

    fl_utils = import_module("vllm_fl.utils")

    _mock_platform(monkeypatch, cuda=True)
    monkeypatch.setattr(fl_utils, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(fl_utils, "use_flaggems_op", lambda op_name: False)
    moe_adapter.install_fl_w8a8_moe_selector()
    config = SimpleNamespace(
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=False,
        ),
    )

    backend, experts_cls = oracle.select_int8_moe_backend(
        config,
        weight_key=None,
        activation_key=None,
    )

    from vllm_fl.quantization.w8a8.moe_experts import (
        VllmFunctionalW8A8Experts,
    )

    assert backend == "triton"
    assert experts_cls is VllmFunctionalW8A8Experts
    assert upstream_calls == []


@pytest.mark.parametrize(
    "activation,params",
    [
        (
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            dict(swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0),
        ),
        (MoEActivation.SILU, dict(swiglu_limit=7.0)),
        (MoEActivation.SILU, dict(swiglu_alpha=1.702)),
        (MoEActivation.SILU, dict(swiglu_beta=1.0)),
    ],
)
def test_native_fallback_preserves_configurable_activation(
    monkeypatch, activation, params
):
    calls = []
    from vllm_fl.quantization.w8a8.moe_experts import (
        TritonExperts,
        VllmFunctionalW8A8Experts,
    )

    def upstream_selector(config, **kwargs):
        calls.append((config, kwargs))
        return "triton", TritonExperts

    oracle, _ = _install_with_fake_modules(
        monkeypatch, upstream_selector, lambda **kwargs: kwargs
    )
    import vllm_fl.utils as fl_utils

    _mock_platform(monkeypatch, cuda=True)
    monkeypatch.setattr(fl_utils, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(fl_utils, "use_flaggems_op", lambda op_name: False)
    config = SimpleNamespace(activation=activation, **params)
    moe_adapter.install_fl_w8a8_moe_selector()
    assert oracle.select_int8_moe_backend(config) == (
        "triton",
        VllmFunctionalW8A8Experts,
    )
    assert calls == [(config, {"weight_key": None, "activation_key": None})]
    assert calls[0][0].activation is activation
    for key, value in params.items():
        assert getattr(calls[0][0], key) == value


def test_w8a8_moe_selector_nvidia_noncanonical_falls_back_without_flaggems(
    monkeypatch,
):
    upstream_calls = []

    def upstream_selector(*args, **kwargs):
        upstream_calls.append((args, kwargs))
        return "nvidia-native"

    oracle, _ = _install_with_fake_modules(
        monkeypatch,
        upstream_selector,
        lambda **kwargs: kwargs,
    )

    fl_utils = import_module("vllm_fl.utils")

    _mock_platform(monkeypatch, cuda=True)
    monkeypatch.setattr(
        fl_utils,
        "use_flaggems_op",
        lambda op_name: (_ for _ in ()).throw(
            AssertionError("noncanonical W8A8 must not consult the FlagGems gate")
        ),
    )
    moe_adapter.install_fl_w8a8_moe_selector()
    config = SimpleNamespace()

    result = oracle.select_int8_moe_backend(
        config,
        weight_key="noncanonical",
        activation_key="noncanonical",
    )

    assert result == "nvidia-native"
    assert len(upstream_calls) == 1


@pytest.mark.parametrize("api", ["builder", "selector", "factory"])
def test_moe_install_validates_all_signatures_before_writing(monkeypatch, api):
    def compatible(*args, **kwargs):
        return None

    oracle, scheme = _install_with_fake_modules(monkeypatch, compatible, compatible)
    platforms = import_module("vllm.platforms")
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(vendor_name="ascend", device_type="npu"),
    )
    module = import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe"
    )

    class Factory:
        get_moe_method = staticmethod(compatible)

    monkeypatch.setattr(module, "CompressedTensorsMoEMethod", Factory)
    if api == "builder":
        scheme.make_int8_moe_quant_config = lambda: None
    elif api == "selector":
        oracle.select_int8_moe_backend = lambda: None
    else:
        Factory.get_moe_method = staticmethod(lambda: None)
    snapshots = [dict(vars(owner)) for owner in (oracle, scheme, Factory)]
    with pytest.raises(TypeError):
        moe_adapter.install_fl_w8a8_moe_selector()
    assert [dict(vars(owner)) for owner in (oracle, scheme, Factory)] == snapshots


@pytest.mark.parametrize("failure_target", ["factory", "oracle", "scheme"])
@pytest.mark.parametrize("preinstalled", [False, True])
def test_moe_install_rolls_back_descriptors_and_wrappers(
    monkeypatch, failure_target, preinstalled
):
    def compatible(*args, **kwargs):
        return None

    def builder(*args, **kwargs):
        return "existing-wrapper"

    if preinstalled:
        setattr(builder, moe_adapter._CONFIG_BUILDER_MARKER, True)

    class FailOnceNamespace(SimpleNamespace):
        def __setattr__(self, name, value):
            if name == "select_int8_moe_backend" and self.fail:
                super().__setattr__("fail", False)
                raise RuntimeError("injected MoE installation failure")
            super().__setattr__(name, value)

    oracle = FailOnceNamespace(
        fail=failure_target == "oracle", select_int8_moe_backend=compatible
    )
    scheme = FailOnceNamespace(
        fail=failure_target == "scheme",
        select_int8_moe_backend=compatible,
        make_int8_moe_quant_config=builder,
    )
    modules = {moe_adapter._ORACLE_MODULE: oracle, moe_adapter._SCHEME_MODULE: scheme}
    monkeypatch.setattr(moe_adapter, "import_module", lambda name: modules[name])
    monkeypatch.setattr(
        import_module("vllm.platforms"),
        "current_platform",
        SimpleNamespace(vendor_name="ascend", device_type="npu"),
    )

    class FailOnceMeta(type):
        def __setattr__(cls, name, value):
            if name == "get_moe_method" and cls.fail:
                type.__setattr__(cls, "fail", False)
                raise RuntimeError("injected MoE installation failure")
            super().__setattr__(name, value)

    class BaseFactory:
        get_moe_method = staticmethod(compatible)

    class Factory(BaseFactory, metaclass=FailOnceMeta):
        fail = failure_target == "factory"

    module = import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe"
    )
    monkeypatch.setattr(module, "CompressedTensorsMoEMethod", Factory)

    def snapshot():
        return [
            {name: value for name, value in vars(owner).items() if name != "fail"}
            for owner in (oracle, scheme, Factory)
        ]

    originals = snapshot()
    descriptor = vars(BaseFactory)["get_moe_method"]
    with pytest.raises(RuntimeError, match="injected MoE installation failure"):
        moe_adapter.install_fl_w8a8_moe_selector()
    assert snapshot() == originals
    assert vars(BaseFactory)["get_moe_method"] is descriptor
    assert "get_moe_method" not in vars(Factory)
    assert scheme.make_int8_moe_quant_config is builder
    assert moe_adapter.install_fl_w8a8_moe_selector()
    installed = snapshot()
    assert isinstance(vars(Factory)["get_moe_method"], staticmethod)
    assert moe_adapter.install_fl_w8a8_moe_selector()
    assert snapshot() == installed

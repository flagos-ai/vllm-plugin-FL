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

"""CPU tests for the W8A8 MoE backend selector.

The vLLM 0.24 oracle/scheme modules are stand-ins from ``_w8a8_fakes``.
The tests drive ``install_fl_w8a8_moe_selector`` and then exercise the
installed wrappers, asserting the exact platform/whitelist gate and the
backend/experts class each branch returns.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ._w8a8_fakes import (
    Int8MoeBackend,
    MoEActivation,
    build_moe_selector_environment,
    import_under_fakes,
)

MODULE = "vllm_fl.quantization.w8a8.moe"

_EXC = {"value": ValueError, "not_impl": NotImplementedError}


@pytest.fixture
def moe():
    env = build_moe_selector_environment()
    with import_under_fakes(MODULE, env.modules, purge=(MODULE,)) as module:
        yield module, env


def _config(**attrs):
    defaults = {
        "is_lora_enabled": False,
        "activation": MoEActivation.SILU,
        "swiglu_limit": None,
        "swiglu_alpha": None,
        "swiglu_beta": None,
        "moe_parallel_config": SimpleNamespace(
            use_batched_activation_format=False
        ),
    }
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def test_install_patches_both_modules_and_is_idempotent(moe):
    module, env = moe
    assert module.install_fl_w8a8_moe_selector() is True
    wrapper = env.oracle_module.select_int8_moe_backend
    assert env.scheme_module.select_int8_moe_backend is wrapper

    assert module.install_fl_w8a8_moe_selector() is True
    assert env.oracle_module.select_int8_moe_backend is wrapper
    assert env.scheme_module.select_int8_moe_backend is wrapper


def test_install_wraps_quant_config_builder_for_static_inputs(moe):
    module, env = moe
    module.install_fl_w8a8_moe_selector()
    builder = env.scheme_module.make_int8_moe_quant_config
    result = builder(
        w1_scale="w1",
        w2_scale="w2",
        a1_scale="a1",
        a2_scale="a2",
        w1_bias="b1",
        w2_bias="b2",
        per_act_token_quant=False,
    )
    assert result == "original-quant-config"
    assert env.make_quant_calls == [
        {
            "w1_scale": "w1",
            "w2_scale": "w2",
            "a1_scale": "a1",
            "a2_scale": "a2",
            "w1_bias": "b1",
            "w2_bias": "b2",
            "per_act_token_quant": False,
        }
    ]
    assert env.int8_config_calls == []


def test_install_wraps_quant_config_builder_for_dynamic_tokens(moe):
    module, env = moe
    module.install_fl_w8a8_moe_selector()
    builder = env.scheme_module.make_int8_moe_quant_config
    result = builder(w1_scale="w1", w2_scale="w2", per_act_token_quant=True)
    assert result == "w8a8-quant-config"
    assert env.int8_config_calls == [
        {
            "w1_scale": "w1",
            "w2_scale": "w2",
            "a1_scale": None,
            "a2_scale": None,
            "w1_bias": None,
            "w2_bias": None,
            "per_act_token_quant": True,
        }
    ]
    assert env.make_quant_calls == []


def test_selector_prefers_flaggems_on_canonical_w8a8(moe):
    module, env = moe
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    backend, experts = selector(
        _config(),
        weight_key=env.kInt8StaticChannelSym,
        activation_key=env.kInt8DynamicTokenSym,
    )
    assert backend is Int8MoeBackend.TRITON
    assert experts is env.FlagGemsW8A8Experts
    assert env.oracle_selector_calls == []
    messages = [call[0][0] for call in env.logger_calls if call[0]]
    assert "Using FlagGems W8A8 MoE experts." in messages


def test_selector_rejects_lora_for_flaggems_path(moe):
    module, env = moe
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    with pytest.raises(NotImplementedError, match="does not support LoRA"):
        selector(_config(is_lora_enabled=True))


def test_selector_rejects_batched_activation_for_flaggems_path(moe):
    module, env = moe
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    config = _config(
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=True
        )
    )
    with pytest.raises(ValueError, match="standard activation format"):
        selector(config)


def test_selector_honours_flaggems_policy_gate(moe):
    module, env = moe
    env.flaggems_ops = {"fused_moe": False}
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    backend, experts = selector(_config())
    assert backend is Int8MoeBackend.TRITON
    assert experts is env.VllmFunctionalW8A8Experts


def test_selector_requires_oot_enabled_for_flaggems(moe):
    module, env = moe
    env.oot_enabled = False
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    _, experts = selector(_config())
    assert experts is env.VllmFunctionalW8A8Experts


def test_selector_delegates_non_cuda_or_non_canonical(moe):
    module, env = moe
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    config = _config()

    assert selector(config, weight_key=object()) == env.oracle_selector_result
    assert env.oracle_selector_calls[-1]["weight_key"] is not None

    env.is_cuda = False
    assert selector(config) == env.oracle_selector_result
    assert env.oracle_selector_calls[-1]["config"] is config


def test_native_fallback_returns_functional_bridge(moe):
    module, env = moe
    env.flaggems_default = False
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    backend, experts = selector(_config())
    assert backend is Int8MoeBackend.TRITON
    assert experts is env.VllmFunctionalW8A8Experts
    assert env.oracle_selector_calls == []


def test_native_fallback_replaces_triton_experts_for_modular_activation(moe):
    module, env = moe
    env.flaggems_default = False
    env.oracle_selector_result = (Int8MoeBackend.CUTLASS, env.TritonExperts)
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    config = _config(activation=MoEActivation.GELU)
    backend, experts = selector(config)
    assert backend is Int8MoeBackend.CUTLASS
    assert experts is env.VllmFunctionalW8A8Experts
    assert env.oracle_selector_calls[-1]["config"] is config


def test_native_fallback_treats_swiglu_params_as_modular(moe):
    module, env = moe
    env.flaggems_default = False
    env.oracle_selector_result = (Int8MoeBackend.TRITON, env.TritonExperts)
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    backend, experts = selector(_config(swiglu_limit=7.0))
    assert backend is Int8MoeBackend.TRITON
    assert experts is env.VllmFunctionalW8A8Experts


def test_native_fallback_keeps_non_triton_experts(moe):
    module, env = moe
    env.flaggems_default = False
    other = type("OtherExperts", (), {})
    env.oracle_selector_result = (Int8MoeBackend.CUTLASS, other)
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    backend, experts = selector(_config(activation=MoEActivation.GELU))
    assert backend is Int8MoeBackend.CUTLASS
    assert experts is other


@pytest.mark.parametrize(
    ("config", "exc_key", "match"),
    [
        (
            _config(is_lora_enabled=True),
            "not_impl",
            "functional W8A8 MoE bridge does not support LoRA",
        ),
        (
            _config(
                moe_parallel_config=SimpleNamespace(
                    use_batched_activation_format=True
                )
            ),
            "value",
            "functional W8A8 MoE bridge requires the standard",
        ),
    ],
)
def test_native_fallback_rejects_unsupported_config(
    moe, config, exc_key, match
):
    module, env = moe
    env.flaggems_default = False
    module.install_fl_w8a8_moe_selector()
    selector = env.oracle_module.select_int8_moe_backend
    with pytest.raises(_EXC[exc_key], match=match):
        selector(config)

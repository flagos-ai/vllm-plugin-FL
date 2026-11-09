# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, False), ("0", False), ("false", False), ("1", True), ("TRUE", True)],
)
def test_fused_package_requires_explicit_enable(monkeypatch, value, enabled):
    import vllm_fl.envs as fl_envs

    monkeypatch.delenv("USE_FLAGGEMS", raising=False)
    if value is None:
        monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_VLLM", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", value)
    assert fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM is enabled
    assert fl_envs.USE_FLAGGEMS is True


def test_default_quantization_does_not_require_fused_package(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.impl import quantization

    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_VLLM", raising=False)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    x = object()
    expected = (object(), object())
    calls = []

    def native(value):
        calls.append(value)
        return expected

    monkeypatch.setattr(
        quantization, "dynamic_per_token_quant_int8_flaggems_triton", native
    )
    assert quantization.dynamic_per_token_quant_int8_flaggems_vllm(x) is expected
    assert calls == [x]


def test_enabled_quantization_requires_fused_package(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.impl import quantization

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    for name in list(sys.modules):
        if name.startswith("flaggems_vllm."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    with pytest.raises(ModuleNotFoundError):
        quantization.dynamic_per_token_quant_int8_flaggems_vllm(object())


def test_default_preserves_compilation_operator_selection(monkeypatch):
    from vllm_fl.ops.activation import enable_fused_activations_for_compile

    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_VLLM", raising=False)
    config = SimpleNamespace(custom_ops=["-SiluAndMul", "+existing"])
    enable_fused_activations_for_compile(config)
    assert config.custom_ops == ["-SiluAndMul", "+existing"]

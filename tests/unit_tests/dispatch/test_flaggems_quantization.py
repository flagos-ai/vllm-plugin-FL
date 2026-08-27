# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import ModuleType

import pytest
import torch

from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
    dynamic_per_token_quant_int8_flaggems,
)


def _install_flaggems_vllm_quant_module(monkeypatch, quant_fn):
    package = ModuleType("flaggems_vllm")
    package.__path__ = []
    ops_package = ModuleType("flaggems_vllm.ops")
    ops_package.__path__ = []
    quant_module = ModuleType("flaggems_vllm.ops.scaled_int8_quant")
    quant_module.dynamic_per_token_quant_int8 = quant_fn

    monkeypatch.setitem(sys.modules, "flaggems_vllm", package)
    monkeypatch.setitem(sys.modules, "flaggems_vllm.ops", ops_package)
    monkeypatch.setitem(
        sys.modules,
        "flaggems_vllm.ops.scaled_int8_quant",
        quant_module,
    )


def test_dynamic_per_token_quant_prefers_flaggems_vllm(monkeypatch):
    x = torch.ones((2, 4), dtype=torch.bfloat16)
    expected_q = torch.ones_like(x, dtype=torch.int8)
    expected_scale = torch.full((2, 1), 0.25, dtype=torch.float32)
    calls = []

    def quant_fn(value):
        calls.append(value)
        return expected_q, expected_scale

    _install_flaggems_vllm_quant_module(monkeypatch, quant_fn)

    actual_q, actual_scale = dynamic_per_token_quant_int8_flaggems(x)

    assert calls == [x]
    assert actual_q is expected_q
    assert actual_scale is expected_scale


def test_dynamic_per_token_quant_propagates_failure_for_dispatch_fallback(
    monkeypatch,
):
    def unavailable(_value):
        raise RuntimeError("FlagGems-vLLM kernel is unavailable")

    _install_flaggems_vllm_quant_module(monkeypatch, unavailable)

    with pytest.raises(RuntimeError, match="kernel is unavailable"):
        dynamic_per_token_quant_int8_flaggems(torch.ones((1, 4)))

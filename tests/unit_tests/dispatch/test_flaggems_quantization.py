# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import ModuleType

import pytest
import torch

from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
    dynamic_per_token_quant_int8_flaggems_triton,
    dynamic_per_token_quant_int8_flaggems_vllm,
)


def _install_flaggems_vllm_quant_module(monkeypatch, quant_fn):
    package = ModuleType("flaggems_vllm")
    package.__path__ = []
    ops_package = ModuleType("flaggems_vllm.ops")
    ops_package.__path__ = []
    quant_module = ModuleType("flaggems_vllm.ops.scaled_int8_quant")
    quant_module.scaled_int8_quant = quant_fn

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

    def quant_fn(value, *, scale, azp, symmetric):
        calls.append((value, scale, azp, symmetric))
        return expected_q, expected_scale, None

    _install_flaggems_vllm_quant_module(monkeypatch, quant_fn)

    actual_q, actual_scale = dynamic_per_token_quant_int8_flaggems_vllm(x)

    assert calls == [(x, None, None, True)]
    assert actual_q is expected_q
    assert actual_scale is expected_scale


def test_dynamic_per_token_quant_propagates_failure_for_dispatch_fallback(
    monkeypatch,
):
    def unavailable(_value, *, scale, azp, symmetric):
        raise RuntimeError("FlagGems-vLLM kernel is unavailable")

    _install_flaggems_vllm_quant_module(monkeypatch, unavailable)

    with pytest.raises(RuntimeError, match="kernel is unavailable"):
        dynamic_per_token_quant_int8_flaggems_vllm(torch.ones((1, 4)))


def test_local_triton_quant_validates_input_contract():
    with pytest.raises(ValueError, match="2D"):
        dynamic_per_token_quant_int8_flaggems_triton(torch.ones((1, 2, 4)))
    with pytest.raises(TypeError, match="floating point"):
        dynamic_per_token_quant_int8_flaggems_triton(
            torch.ones((1, 4), dtype=torch.int8)
        )
    with pytest.raises(ValueError, match="hidden_size"):
        dynamic_per_token_quant_int8_flaggems_triton(torch.ones((1, 0)))


def test_flaggems_quantization_registers_ordered_fallbacks(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems import register_ops
    from vllm_fl.dispatch.types import BackendPriority

    registered = []

    class Registry:
        def register_many(self, impls):
            registered.extend(impls)

    monkeypatch.setattr(
        register_ops,
        "use_flaggems_op",
        lambda op_name: op_name == "dynamic_per_token_quant_int8",
    )

    register_ops.register_builtins(Registry())

    assert [impl.impl_id for impl in registered] == [
        "default.flagos",
        "default.flagos_triton",
    ]
    assert [impl.priority for impl in registered] == [
        BackendPriority.DEFAULT + 10,
        BackendPriority.DEFAULT,
    ]


@pytest.mark.gpu
@pytest.mark.flaggems
def test_local_triton_quant_matches_reference_on_cuda(device):
    if device.type != "cuda":
        pytest.skip("local Triton quantization contract is validated on CUDA")

    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8 as reference_quant,
    )

    x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.5 / 127.0, 1.5 / 127.0],
            [7.75, -8.0, 0.03125, -0.5],
        ],
        device=device,
        dtype=torch.float32,
    )

    actual_q, actual_scale = dynamic_per_token_quant_int8_flaggems_triton(x)
    expected_q, expected_scale = reference_quant(x)

    assert torch.equal(actual_q, expected_q)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=1e-7)

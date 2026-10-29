# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

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
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
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
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")

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


def _int8_dispatch(monkeypatch, enabled):
    import os

    import flag_gems

    import vllm.model_executor.kernels.linear.scaled_mm.triton as native

    from vllm_fl.dispatch.backends.flaggems import register_ops as fg_registry
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.dispatch.backends.reference.register_ops import (
        register_builtins as register_reference,
    )
    from vllm_fl.dispatch.manager import OpManager
    from vllm_fl.dispatch.registry import OpRegistry

    if enabled is None:
        monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_VLLM", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", enabled)
    selected = enabled in ("1", "true")
    out = torch.zeros(2, 4, dtype=torch.bfloat16)
    fv, ref = (Mock(return_value=out), Mock(return_value=out))
    fp8 = Mock(side_effect=AssertionError("FP8 API must never handle INT8"))
    monkeypatch.setattr(flag_gems, "scaled_mm", fp8)
    monkeypatch.setattr(native, "triton_scaled_mm", ref)
    monkeypatch.setitem(
        sys.modules,
        "flaggems_vllm",
        SimpleNamespace(triton_scaled_mm=fv) if selected else None,
    )
    monkeypatch.setattr(FlagGemsBackend, "is_available", lambda self: True)
    monkeypatch.setattr(fg_registry, "use_flaggems_op", lambda op: True)
    registry = OpRegistry()
    fg_registry.register_builtins(registry)
    register_reference(registry)
    manager = OpManager(registry=registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    args = (
        torch.ones(2, 8, dtype=torch.int8),
        torch.ones(8, 4, dtype=torch.int8),
        torch.ones(2, 1),
        torch.ones(1, 4),
        torch.bfloat16,
        None,
    )
    return (manager, args, fv, ref, fp8, out)


@pytest.mark.parametrize("enabled", [None, "0", "1"])
@pytest.mark.parametrize("strict", [False, True])
def test_int8_registry_selects_fv_or_native_without_fp8(monkeypatch, enabled, strict):
    from vllm_fl.dispatch.policy import (
        SelectionPolicy,
        reset_global_policy,
        set_global_policy,
    )

    manager, args, fv, ref, fp8, out = _int8_dispatch(monkeypatch, enabled)
    selected = enabled in ("1", "true")
    set_global_policy(SelectionPolicy(strict=strict))
    try:
        assert manager.get_selected_impl_id("scaled_int8_mm") == (
            "default.flagos" if selected else "reference.torch"
        )
        assert manager.call("scaled_int8_mm", *args) is out
        (fv if selected else ref).assert_called_once_with(*args)
        (ref if selected else fv).assert_not_called()
        fp8.assert_not_called()
    finally:
        reset_global_policy()


@pytest.mark.parametrize("strict", [False, True])
def test_int8_selected_failure_preserves_strict_or_native_fallback(monkeypatch, strict):
    from vllm_fl.dispatch.policy import (
        SelectionPolicy,
        reset_global_policy,
        set_global_policy,
    )

    manager, args, fv, ref, fp8, out = _int8_dispatch(monkeypatch, "1")
    fv.side_effect = RuntimeError("INT8 provider failed")
    set_global_policy(SelectionPolicy(strict=strict))
    try:
        if strict:
            with pytest.raises(RuntimeError, match="INT8 provider failed"):
                manager.call("scaled_int8_mm", *args)
            ref.assert_not_called()
        else:
            assert manager.call("scaled_int8_mm", *args) is out
            ref.assert_called_once_with(*args)
        fv.assert_called_once_with(*args)
        fp8.assert_not_called()
    finally:
        reset_global_policy()


def test_dynamic_quant_disabled_switch_uses_local_path(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.impl import quantization

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    original = Mock(return_value=("original", "scale"))
    monkeypatch.setattr(
        quantization, "dynamic_per_token_quant_int8_flaggems_triton", original
    )
    monkeypatch.setitem(sys.modules, "flaggems_vllm.ops.scaled_int8_quant", None)
    assert quantization.dynamic_per_token_quant_int8_flaggems_vllm(None) == (
        "original",
        "scale",
    )
    original.assert_called_once_with(None)

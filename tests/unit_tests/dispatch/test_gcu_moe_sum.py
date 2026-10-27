# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU routing/validation contracts for MoE sum, not GCU kernel numerics."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_SOURCE = (
    Path(__file__).parents[3] / "vllm_fl/dispatch/backends/flaggems/impl/fused_moe.py"
)


@pytest.fixture
def moe_adapter(monkeypatch):
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    utils = ModuleType("vllm.utils")
    utils.__path__ = []
    triton_utils = ModuleType("vllm.triton_utils")
    triton_utils.triton = SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b)
    math_utils = ModuleType("vllm.utils.math_utils")
    math_utils.round_up = lambda n, b: ((n + b - 1) // b) * b
    for name, module in (
        ("vllm", vllm),
        ("vllm.utils", utils),
        ("vllm.triton_utils", triton_utils),
        ("vllm.utils.math_utils", math_utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_gcu_moe_sum_contract", _SOURCE)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


class _Tensor:
    def __init__(self, device, contiguous):
        self.device = SimpleNamespace(type=device)
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous


def _install_providers(monkeypatch, official, fused, validator):
    gems = ModuleType("flag_gems")
    gems.vendor_name = "enflame"
    if official is not None:
        gems.moe_sum = official
    fused_package = ModuleType("flaggems_vllm")
    fused_package.__path__ = []
    if fused is not None:
        fused_package.moe_sum = fused
    ops = ModuleType("flaggems_vllm.ops")
    ops.__path__ = []
    module = ModuleType("flaggems_vllm.ops.moe_sum")
    if validator is not None:
        module._check_moe_sum_inputs = validator
    for name, value in (
        ("flag_gems", gems),
        ("flaggems_vllm", fused_package),
        ("flaggems_vllm.ops", ops),
        ("flaggems_vllm.ops.moe_sum", module),
    ):
        monkeypatch.setitem(sys.modules, name, value)


@pytest.mark.parametrize("device", ["gcu", "cuda", "npu", "xpu", "cpu"])
@pytest.mark.parametrize(
    "input_contiguous,output_contiguous",
    [(True, True), (False, True), (True, False), (False, False)],
    ids=["contiguous", "strided-input", "strided-output", "strided-both"],
)
def test_sum_provider_follows_device_and_both_layouts(
    monkeypatch, moe_adapter, device, input_contiguous, output_contiguous
):
    inp = _Tensor(device, input_contiguous)
    out = _Tensor(device, output_contiguous)
    events = []

    def validator(actual_input, actual_output):
        assert actual_input is inp and actual_output is out
        events.append("validated")

    def official(actual_input, actual_output):
        assert actual_input is inp and actual_output is out
        assert events == ["validated"]
        events.append("official")

    def fused(actual_input, actual_output):
        assert actual_input is inp and actual_output is out
        # The public fused provider retains responsibility for its own checks.
        events.append("fused")

    _install_providers(monkeypatch, official, fused, validator)
    assert moe_adapter.moe_sum_flaggems(inp, out) is None
    expected = (
        ["validated", "official"]
        if device == "gcu" and input_contiguous and output_contiguous
        else ["fused"]
    )
    assert events == expected


@pytest.mark.parametrize(
    "message",
    [
        "input must be 3D",
        "output must be 2D",
        "topk must be >= 1",
        "output shape mismatch",
        "dtype mismatch",
        "unsupported dtype",
        "device mismatch",
    ],
)
def test_official_route_preserves_dependency_validator_failure(
    monkeypatch, moe_adapter, message
):
    inp, out = _Tensor("gcu", True), _Tensor("gcu", True)
    checked = []

    def validator(actual_input, actual_output):
        assert actual_input is inp and actual_output is out
        checked.append((actual_input, actual_output))
        raise AssertionError(message)

    def forbidden(*args):
        raise RuntimeError("An invalid input must not launch either provider")

    _install_providers(monkeypatch, forbidden, forbidden, validator)
    with pytest.raises(AssertionError, match=message):
        moe_adapter.moe_sum_flaggems(inp, out)
    assert len(checked) == 1


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("error_type", [RuntimeError, TypeError])
def test_provider_exception_propagates_without_retry(
    monkeypatch, moe_adapter, contiguous, error_type
):
    inp, out = _Tensor("gcu", contiguous), _Tensor("gcu", True)
    calls = []

    def official(*args):
        calls.append("official")
        raise error_type("native failure")

    def fused(*args):
        calls.append("fused")
        raise error_type("native failure")

    _install_providers(monkeypatch, official, fused, lambda *args: None)
    with pytest.raises(error_type, match="native failure"):
        moe_adapter.moe_sum_flaggems(inp, out)
    assert calls == ["official" if contiguous else "fused"]


def test_missing_official_provider_is_not_silently_replaced(monkeypatch, moe_adapter):
    def forbidden(*args):
        raise AssertionError("The GCU route must not fall back after import failure")

    _install_providers(monkeypatch, None, forbidden, lambda *args: None)
    with pytest.raises(ImportError):
        moe_adapter.moe_sum_flaggems(_Tensor("gcu", True), _Tensor("gcu", True))


def test_missing_dependency_validator_fails_before_provider(monkeypatch, moe_adapter):
    def forbidden(*args):
        raise AssertionError("A missing validator must not launch either provider")

    _install_providers(monkeypatch, forbidden, forbidden, None)
    with pytest.raises(ImportError):
        moe_adapter.moe_sum_flaggems(_Tensor("gcu", True), _Tensor("gcu", True))

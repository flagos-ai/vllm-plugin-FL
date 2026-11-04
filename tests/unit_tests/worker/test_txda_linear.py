# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU behavior contracts for the narrowly scoped TXDA BF16 GEMM provider."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/worker/txda_linear.py"


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location("txda_linear_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [(6,), (1, 6), (3, 6), (2, 3, 6)])
def test_bf16_linear_flattens_batches_and_uses_transposed_weight(adapter, shape):
    gen = torch.Generator().manual_seed(77)
    x = torch.randn(shape, generator=gen, dtype=torch.bfloat16)
    weight = torch.randn(11, 6, generator=gen, dtype=torch.bfloat16)
    original = Mock(side_effect=lambda layer, a, w, b: F.linear(a, w, b))
    mm = Mock(side_effect=torch.mm)
    report = Mock()
    fn = adapter._make_txda_gemm(original, mm, device_type="cpu", on_first_use=report)
    actual = fn(None, x, weight)
    torch.testing.assert_close(actual, F.linear(x, weight))
    assert actual.shape == (*shape[:-1], 11)
    assert mm.call_args.args[0].shape == (x.numel() // 6, 6)
    assert mm.call_args.args[1].shape == (6, 11)
    assert mm.call_args.args[1].stride() == (1, 6)
    original.assert_not_called()
    fn(None, x, weight)
    report.assert_called_once_with(x, weight)


@pytest.mark.parametrize(
    "kind", ["float32", "float16", "bias", "empty", "grad", "device"]
)
def test_unsupported_cases_preserve_the_original_implementation(adapter, kind):
    x = torch.ones(2, 6, dtype=torch.bfloat16)
    weight = torch.ones(11, 6, dtype=torch.bfloat16)
    bias = None
    device_type = "cpu"
    if kind in ("float32", "float16"):
        dtype = getattr(torch, kind)
        x, weight = x.to(dtype), weight.to(dtype)
    elif kind == "bias":
        bias = torch.ones(11, dtype=torch.bfloat16)
    elif kind == "empty":
        x = x[:0]
    elif kind == "grad":
        x.requires_grad_(True)
    elif kind == "device":
        device_type = "txda"
    expected = object()
    original = Mock(return_value=expected)
    mm = Mock()
    fn = adapter._make_txda_gemm(original, mm, device_type=device_type)
    assert fn("layer", x, weight, bias) is expected
    original.assert_called_once_with("layer", x, weight, bias)
    mm.assert_not_called()


def test_vendor_errors_do_not_silently_fall_back(adapter):
    original = Mock()
    mm = Mock(side_effect=RuntimeError("vendor failed"))
    fn = adapter._make_txda_gemm(original, mm, device_type="cpu")
    with pytest.raises(RuntimeError, match="vendor failed"):
        fn(
            None,
            torch.ones(1, 6, dtype=torch.bfloat16),
            torch.ones(11, 6, dtype=torch.bfloat16),
        )
    original.assert_not_called()


def test_vendor_output_dtype_is_checked(adapter):
    original = Mock()
    fn = adapter._make_txda_gemm(
        original, lambda a, b: torch.mm(a, b).float(), device_type="cpu"
    )
    with pytest.raises(RuntimeError, match="incorrect device or dtype"):
        fn(
            None,
            torch.ones(1, 6, dtype=torch.bfloat16),
            torch.ones(11, 6, dtype=torch.bfloat16),
        )
    original.assert_not_called()


def test_install_is_txda_only_idempotent_and_has_explicit_opt_out(adapter, monkeypatch):
    def module(name, package=False):
        value = ModuleType(name)
        if package:
            value.__path__ = []
        monkeypatch.setitem(sys.modules, name, value)
        return value

    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "flag_gems",
        "flag_gems.runtime",
        "flag_gems.runtime.backend",
        "flag_gems.runtime.backend._tsingmicro",
        "flag_gems.runtime.backend._tsingmicro.ops",
    ):
        module(name, True)
    platform = module("vllm.platforms")
    platform.current_platform = SimpleNamespace(device_type="cuda")
    utils = module("vllm.model_executor.layers.utils")
    original = lambda layer, x, w, b=None: F.linear(x, w, b)
    utils.default_unquantized_gemm = original
    logger = module("vllm.logger")
    logger.init_logger = Mock(return_value=Mock())
    vendor = module("flag_gems.runtime.backend._tsingmicro.ops.mm")

    def mm(a, b):
        return torch.mm(a, b)

    vendor.mm = mm
    assert not adapter.patch_txda_unquantized_gemm(rank=0)
    assert utils.default_unquantized_gemm is original
    platform.current_platform.device_type = "txda"
    monkeypatch.setenv("FL_TXDA_USE_VENDOR_GEMM", "0")
    assert not adapter.patch_txda_unquantized_gemm(rank=0)
    assert utils.default_unquantized_gemm is original
    monkeypatch.setenv("FL_TXDA_USE_VENDOR_GEMM", "1")
    assert adapter.patch_txda_unquantized_gemm(rank=0)
    installed = utils.default_unquantized_gemm
    assert installed.__wrapped__ is original
    assert installed._fl_txda_bf16_gemm
    assert adapter.patch_txda_unquantized_gemm(rank=0)
    assert utils.default_unquantized_gemm is installed
    logger.init_logger.assert_called_once_with("vllm.fl.txda.linear")

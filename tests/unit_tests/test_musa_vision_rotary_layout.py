# Copyright (c) 2026 BAAI. All rights reserved.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


class _CpuTensorReportedAsMusa(torch.Tensor):
    # Only the Python device guard sees MUSA; tensor storage and all math are CPU.
    @property
    def device(self):
        return SimpleNamespace(type="musa")


@pytest.fixture
def rotary_patch(monkeypatch):
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/musa/patch.py"
    )
    spec = importlib.util.spec_from_file_location("musa_rotary_patch", source)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    original = Mock()

    def forward_native(self, x, cos, sin):
        return original(self, x, cos, sin)

    rotary = type("ApplyRotaryEmb", (), {"forward_native": forward_native})
    common = ModuleType("vllm.model_executor.layers.rotary_embedding.common")
    common.ApplyRotaryEmb = rotary

    def install():
        with monkeypatch.context() as imports:
            imports.setitem(sys.modules, common.__name__, common)
            patch.patch_musa_vision_rotary_layout()

    install()
    return rotary, original, install


def _forward(rotary, x, cos, sin, neox, fp32=False):
    instance = rotary()
    instance.is_neox_style = neox
    instance.enable_fp32_compute = fp32
    return instance.forward_native(x, cos, sin).as_subclass(torch.Tensor)


@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("offset", [0, 2])
def test_rotation_preserves_values_and_copies_only_offset_half_views(
    monkeypatch, rotary_patch, neox, offset
):
    rotary, _, _ = rotary_patch
    base = torch.arange(2 * 3 * (8 + offset), device="cpu", dtype=torch.float32)
    native = base.reshape(2, 3, 8 + offset)[..., offset:]
    x = native.as_subclass(_CpuTensorReportedAsMusa)
    before = native.clone()
    cos = torch.zeros(2, 4, device="cpu")
    sin = torch.ones_like(cos)
    # A quarter turn has an independent index/sign oracle for either layout.
    indices = [4, 5, 6, 7, 0, 1, 2, 3] if neox else [1, 0, 3, 2, 5, 4, 7, 6]
    signs = [-1] * 4 + [1] * 4 if neox else [-1, 1] * 4
    expected = native[..., indices] * torch.tensor(signs, device="cpu")
    copies = []
    contiguous = torch.Tensor.contiguous

    def record_copy(tensor, *args, **kwargs):
        copies.append((tensor.is_contiguous(), tensor.storage_offset()))
        return contiguous(tensor.as_subclass(torch.Tensor), *args, **kwargs)

    with monkeypatch.context() as methods:
        methods.setattr(_CpuTensorReportedAsMusa, "contiguous", record_copy)
        actual = _forward(rotary, x, cos, sin, neox)

    assert torch.equal(actual, expected)
    assert torch.equal(native, before)
    assert actual.device.type == "cpu"
    expected_offsets = [offset, offset + (4 if neox else 1)]
    assert copies == [(False, value) for value in expected_offsets if value]


def test_contiguous_half_with_nonzero_offset_is_not_copied(monkeypatch, rotary_patch):
    rotary, _, _ = rotary_patch
    x = torch.arange(8, device="cpu", dtype=torch.float32).reshape(1, 1, 8)
    x = x.as_subclass(_CpuTensorReportedAsMusa)
    assert x.chunk(2, -1)[1].is_contiguous()
    assert x.chunk(2, -1)[1].storage_offset() == 4
    with monkeypatch.context() as methods:
        copied = Mock(side_effect=AssertionError("Unnecessary layout copy"))
        methods.setattr(torch.Tensor, "contiguous", copied)
        actual = _forward(
            rotary,
            x,
            torch.zeros(1, 4, device="cpu"),
            torch.ones(1, 4, device="cpu"),
            True,
        )
    assert torch.equal(
        actual, torch.tensor([[[-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]]])
    )
    copied.assert_not_called()


@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("fp32", [True, False])
def test_fp32_preserves_coefficient_precision_and_restores_input_dtype(
    rotary_patch, neox, fp32
):
    rotary, _, _ = rotary_patch
    x = torch.ones(2, 3, 8, dtype=torch.bfloat16, device="cpu")
    x = x.as_subclass(_CpuTensorReportedAsMusa)
    cos = torch.full((2, 4), 1 + 1 / 256, dtype=torch.float32, device="cpu")
    sin = torch.ones_like(cos)
    actual = _forward(rotary, x, cos, sin, neox, fp32)
    # BF16 rounds 1 + 2**-8 to 1; FP32 must retain the difference until output.
    low = 1 / 256 if fp32 else 0
    expected = [low] * 4 + [2] * 4 if neox else [low, 2] * 4
    expected = torch.tensor(expected, dtype=torch.bfloat16, device="cpu")
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected.expand(2, 3, 8))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_musa_delegates_original_result_and_exception(rotary_patch, device):
    rotary, original, _ = rotary_patch
    instance = rotary()
    x = SimpleNamespace(device=SimpleNamespace(type=device))
    cos, sin, sentinel = object(), object(), object()
    original.return_value = sentinel
    assert instance.forward_native(x, cos, sin) is sentinel
    original.assert_called_once_with(instance, x, cos, sin)
    error = RuntimeError("Original rotary failure")
    original.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        instance.forward_native(x, cos, sin)
    assert caught.value is error


def test_rotary_patch_is_idempotent(rotary_patch):
    rotary, original, install = rotary_patch
    installed = rotary.forward_native
    install()
    assert rotary.forward_native is installed
    assert rotary.forward_native is not original

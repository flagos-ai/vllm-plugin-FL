# Copyright (c) 2026 BAAI. All rights reserved.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def musa_patch():
    # Load only the patch source: importing vllm_fl also initializes the
    # dispatch backends, which is unnecessary for this compatibility contract.
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/musa/patch.py"
    )
    spec = importlib.util.spec_from_file_location("musa_accelerator_patch", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vendor_modules():
    def unsupported(*args, **kwargs):
        raise RuntimeError("DeviceAllocator INTERNAL ASSERT FAILED")

    torch = ModuleType("torch")
    torch.accelerator = SimpleNamespace(
        empty_cache=unsupported,
        max_memory_allocated=unsupported,
        memory_stats=unsupported,
        memory_reserved=unsupported,
        reset_peak_memory_stats=unsupported,
    )
    musa = ModuleType("torch_musa")
    musa.empty_cache = Mock(return_value=None)
    musa.max_memory_allocated = Mock(return_value=128)
    musa.memory_stats = Mock(return_value={"allocated_bytes.all.peak": 128})
    musa.memory_reserved = Mock(return_value=256)
    musa.reset_peak_memory_stats = Mock(return_value=None)
    musa.mem_get_info = Mock(return_value=(512, 1024))
    musa.device = Mock(return_value=object())
    return torch, musa


def _apply_patch(monkeypatch, musa_patch, vendor_modules):
    torch, musa = vendor_modules
    # Replace imports only while calling the patch, restoring any real Torch
    # modules before the test returns control to the rest of the suite.
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "torch", torch)
        imports.setitem(sys.modules, "torch_musa", musa)
        musa_patch.patch_accelerator_missing_attrs()


@pytest.mark.parametrize(
    ("accelerator_name", "vendor_name"),
    [
        ("max_memory_allocated", "max_memory_allocated"),
        ("memory_stats", "memory_stats"),
        ("memory_reserved", "memory_reserved"),
        ("reset_peak_memory_stats", "reset_peak_memory_stats"),
        ("get_memory_info", "mem_get_info"),
    ],
)
@pytest.mark.parametrize("device", [None, 1, "musa:1", object()])
def test_memory_api_forwards_explicit_device_and_result(
    monkeypatch, musa_patch, vendor_modules, accelerator_name, vendor_name, device
):
    torch, musa = vendor_modules
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    accelerator_api = getattr(torch.accelerator, accelerator_name)
    vendor_api = getattr(musa, vendor_name)
    assert accelerator_api is vendor_api
    assert accelerator_api(device=device) is vendor_api.return_value
    vendor_api.assert_called_once_with(device=device)


def test_memory_profiling_uses_current_device_when_argument_is_omitted(
    monkeypatch, musa_patch, vendor_modules
):
    torch, musa = vendor_modules
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert torch.accelerator.memory_stats() == {"allocated_bytes.all.peak": 128}
    assert torch.accelerator.memory_reserved() == 256
    assert torch.accelerator.get_memory_info() == (512, 1024)
    assert torch.accelerator.reset_peak_memory_stats() is None
    for name in (
        "memory_stats",
        "memory_reserved",
        "mem_get_info",
        "reset_peak_memory_stats",
    ):
        getattr(musa, name).assert_called_once_with()


@pytest.mark.parametrize("existing_device_index", [False, True])
def test_existing_empty_cache_and_device_context_behavior_is_preserved(
    monkeypatch, musa_patch, vendor_modules, existing_device_index
):
    torch, musa = vendor_modules
    if existing_device_index:
        torch.accelerator.device_index = Mock(return_value=object())
        expected_context = torch.accelerator.device_index
    else:
        expected_context = musa.device
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert torch.accelerator.empty_cache() is None
    musa.empty_cache.assert_called_once_with()
    assert torch.accelerator.device_index is expected_context
    assert torch.accelerator.device_index(1) is expected_context.return_value
    expected_context.assert_called_once_with(1)


def test_repeated_patch_is_idempotent(monkeypatch, musa_patch, vendor_modules):
    torch, musa = vendor_modules
    _apply_patch(monkeypatch, musa_patch, vendor_modules)
    patched = vars(torch.accelerator).copy()

    # A second application must leave the first bindings and device context
    # intact even if another component changes torch_musa's attributes.
    musa.mem_get_info = Mock()
    musa.empty_cache = Mock()
    musa.device = Mock()
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert vars(torch.accelerator) == patched
    assert torch.accelerator._musa_attrs_patched is True


def test_vendor_memory_errors_are_not_hidden(monkeypatch, musa_patch, vendor_modules):
    torch, musa = vendor_modules
    musa.mem_get_info.side_effect = RuntimeError("MUSA device is unavailable")
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    with pytest.raises(RuntimeError, match="MUSA device is unavailable"):
        torch.accelerator.get_memory_info(device=2)
    musa.mem_get_info.assert_called_once_with(device=2)


@pytest.mark.parametrize(
    "vendor_name",
    [
        "empty_cache",
        "max_memory_allocated",
        "memory_stats",
        "memory_reserved",
        "reset_peak_memory_stats",
        "mem_get_info",
        "device",
    ],
)
@pytest.mark.parametrize("invalid_api", ["missing", "not_callable"])
def test_invalid_vendor_api_leaves_original_bindings_and_allows_retry(
    monkeypatch, musa_patch, vendor_modules, caplog, vendor_name, invalid_api
):
    torch, musa = vendor_modules
    original = vars(torch.accelerator).copy()
    vendor_api = getattr(musa, vendor_name)
    if invalid_api == "missing":
        delattr(musa, vendor_name)
    else:
        setattr(musa, vendor_name, None)

    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert vars(torch.accelerator) == original
    assert not getattr(torch.accelerator, "_musa_attrs_patched", False)
    assert "Failed to patch torch.accelerator attrs for MUSA" in caplog.text

    setattr(musa, vendor_name, vendor_api)
    _apply_patch(monkeypatch, musa_patch, vendor_modules)
    assert torch.accelerator._musa_attrs_patched is True
    assert torch.accelerator.get_memory_info is musa.mem_get_info
    assert torch.accelerator.device_index is musa.device


def test_patch_entry_point_retries_after_incomplete_vendor_api(
    monkeypatch, musa_patch, vendor_modules
):
    torch, musa = vendor_modules
    original = vars(torch.accelerator).copy()
    memory_info = musa.mem_get_info
    del musa.mem_get_info
    other_patches = [
        "patch_topk_topp_sampler",
        "patch_triton_reshape_and_cache_flash",
        "patch_cuda_get_device_properties",
        "patch_cuda_stream_for_musa",
        "patch_inductor_triton_for_musa",
        "patch_moe_topk_softmax_for_musa",
        "patch_triton_mtgpu_alias_for_musa",
        "patch_device_config_for_musa",
    ]
    for name in other_patches:
        monkeypatch.setattr(musa_patch, name, Mock())

    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "torch", torch)
        imports.setitem(sys.modules, "torch_musa", musa)
        musa_patch.apply_musa_patches()
        assert musa_patch._patches_applied is False
        assert vars(torch.accelerator) == original

        musa.mem_get_info = memory_info
        musa_patch.apply_musa_patches()
        assert musa_patch._patches_applied is True
        assert torch.accelerator._musa_attrs_patched is True
        assert torch.accelerator.get_memory_info is memory_info

        calls = {name: getattr(musa_patch, name).call_count for name in other_patches}
        musa_patch.apply_musa_patches()
        assert calls == {
            name: getattr(musa_patch, name).call_count for name in other_patches
        }

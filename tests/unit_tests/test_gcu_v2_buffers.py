# Copyright (c) 2026 BAAI. All rights reserved.
"""Regression tests for transactional GCU V2 hook installation."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def hooks(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "gcu_buffers_under_test",
        root / "vllm_fl/dispatch/backends/vendor/gcu/v2_buffers.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    platform = types.SimpleNamespace(device_type="gcu")
    original = lambda: True
    buffers = types.ModuleType("vllm.v1.worker.gpu.buffer_utils")
    buffers.is_uva_available = original
    gpu = types.ModuleType("vllm.v1.worker.gpu")
    gpu.buffer_utils = buffers
    block = types.ModuleType("vllm.v1.worker.gpu.block_table")
    for name, value in {
        "vllm.platforms": types.SimpleNamespace(current_platform=platform),
        "vllm.v1.worker.gpu": gpu,
        "vllm.v1.worker.gpu.buffer_utils": buffers,
        "vllm.v1.worker.gpu.block_table": block,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    return module, platform, buffers, block, original


def test_missing_dependency_then_retry(hooks):
    module, platform, buffers, block, original = hooks
    with pytest.raises(ImportError):
        module.install_gcu_v2_buffers()
    assert buffers.is_uva_available is original

    class BlockTables:
        def _make_ptr_tensor(self, tensors):
            return tensors

    block.BlockTables = BlockTables
    ptr = BlockTables._make_ptr_tensor
    module.install_gcu_v2_buffers()
    installed = buffers.is_uva_available
    assert installed() is False
    platform.device_type = "cuda"
    assert installed() is True
    assert BlockTables._make_ptr_tensor is not ptr
    assert BlockTables._make_ptr_tensor(
        types.SimpleNamespace(device=types.SimpleNamespace(type="cuda")), [1, 2]
    ) == [1, 2]
    module.install_gcu_v2_buffers()
    assert buffers.is_uva_available is installed


def test_gcu_pointer_conversion_and_idempotence(hooks, monkeypatch):
    module, _, buffers, block, _ = hooks
    raw = object()
    converted = object()
    calls = []

    class BlockTables:
        def _make_ptr_tensor(self, tensors):
            calls.append(tensors)
            return raw

    def convert(value):
        assert value is raw
        return converted

    zeroer = types.ModuleType("vllm_fl.dispatch.backends.vendor.gcu.impl.kv_zeroer")
    zeroer._device_segment_addresses = convert
    monkeypatch.setitem(sys.modules, zeroer.__name__, zeroer)
    block.BlockTables = BlockTables
    module.install_gcu_v2_buffers()
    installed_ptr = BlockTables._make_ptr_tensor
    installed_uva = buffers.is_uva_available
    module.install_gcu_v2_buffers()
    assert BlockTables._make_ptr_tensor is installed_ptr
    assert buffers.is_uva_available is installed_uva
    instance = types.SimpleNamespace(device=types.SimpleNamespace(type="gcu"))
    assert installed_ptr(instance, [1, 2]) is converted
    assert calls == [[1, 2]]


def test_missing_abi_leaves_original(hooks):
    module, _, buffers, block, original = hooks
    block.BlockTables = type("BlockTables", (), {})
    with pytest.raises(AttributeError):
        module.install_gcu_v2_buffers()
    assert buffers.is_uva_available is original


def test_assignment_failure_rolls_back_and_retries(hooks):
    module, _, buffers, block, original = hooks

    class RejectOnce(type):
        reject = True

        def __setattr__(cls, name, value):
            if name == "_make_ptr_tensor" and cls.reject:
                cls.reject = False
                raise RuntimeError("injected assignment failure")
            super().__setattr__(name, value)

    class BlockTables(metaclass=RejectOnce):
        def _make_ptr_tensor(self, tensors):
            return tensors

    block.BlockTables = BlockTables
    ptr = BlockTables._make_ptr_tensor
    with pytest.raises(RuntimeError, match="injected"):
        module.install_gcu_v2_buffers()
    assert buffers.is_uva_available is original
    assert BlockTables._make_ptr_tensor is ptr
    module.install_gcu_v2_buffers()
    assert buffers.is_uva_available._fl_gcu_uva_guard
    assert BlockTables._make_ptr_tensor._fl_gcu_v2_buffers_guard


@pytest.mark.parametrize("broken_hook", ["uva", "ptr"])
def test_noncallable_abi_leaves_original_and_allows_retry(hooks, broken_hook):
    module, _, buffers, block, original = hooks

    class BlockTables:
        def _make_ptr_tensor(self, tensors):
            return tensors

    block.BlockTables = BlockTables
    ptr = BlockTables._make_ptr_tensor
    if broken_hook == "uva":
        buffers.is_uva_available = None
    else:
        BlockTables._make_ptr_tensor = None
    previous_uva = buffers.is_uva_available
    previous_ptr = BlockTables._make_ptr_tensor
    with pytest.raises(TypeError, match="callable upstream"):
        module.install_gcu_v2_buffers()
    assert buffers.is_uva_available is previous_uva
    assert BlockTables._make_ptr_tensor is previous_ptr
    buffers.is_uva_available = original
    BlockTables._make_ptr_tensor = ptr
    module.install_gcu_v2_buffers()
    assert buffers.is_uva_available._fl_gcu_uva_guard
    assert BlockTables._make_ptr_tensor._fl_gcu_v2_buffers_guard

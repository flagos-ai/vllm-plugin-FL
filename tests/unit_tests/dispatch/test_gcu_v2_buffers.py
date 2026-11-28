# Copyright (c) 2026 BAAI. All rights reserved.
"""Dependency/ABI failures must leave both V2 hooks retryable."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_install_failure_then_complete_retry(monkeypatch):
    path = Path(__file__).resolve().parents[3] / (
        "vllm_fl/dispatch/backends/vendor/gcu/v2_buffers.py"
    )
    spec = importlib.util.spec_from_file_location("gcu_buffer_hooks_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_uva = lambda: True
    original_ptr = lambda self, tensors: tensors
    buffers = ModuleType("vllm.v1.worker.gpu.buffer_utils")
    buffers.is_uva_available = original_uva
    gpu = ModuleType("vllm.v1.worker.gpu")
    gpu.buffer_utils = buffers
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type="gcu")
    block = ModuleType("vllm.v1.worker.gpu.block_table")
    for name, obj in {
        "vllm.platforms": platforms,
        "vllm.v1.worker.gpu": gpu,
        "vllm.v1.worker.gpu.buffer_utils": buffers,
        "vllm.v1.worker.gpu.block_table": block,
    }.items():
        monkeypatch.setitem(sys.modules, name, obj)
    with pytest.raises(ImportError):
        module.install_gcu_v2_buffers()
    assert buffers.is_uva_available is original_uva

    class HookMeta(type):
        reject_install = False

        def __setattr__(cls, name, value):
            if (
                name == "_make_ptr_tensor"
                and cls.reject_install
                and value is not original_ptr
            ):
                raise RuntimeError("injected hook assignment failure")
            super().__setattr__(name, value)

    class BlockTables(metaclass=HookMeta):
        _make_ptr_tensor = None

    block.BlockTables = BlockTables
    with pytest.raises(TypeError):
        module.install_gcu_v2_buffers()
    assert buffers.is_uva_available is original_uva
    BlockTables._make_ptr_tensor = original_ptr
    BlockTables.reject_install = True
    with pytest.raises(RuntimeError, match="injected hook assignment failure"):
        module.install_gcu_v2_buffers()
    assert buffers.is_uva_available is original_uva
    assert BlockTables._make_ptr_tensor is original_ptr
    assert not getattr(original_uva, "_fl_gcu_uva_guard", False)
    BlockTables.reject_install = False
    module.install_gcu_v2_buffers()
    assert buffers.is_uva_available() is False
    assert buffers.is_uva_available._fl_gcu_uva_guard
    assert BlockTables._make_ptr_tensor._fl_gcu_v2_buffers_guard
    ptr = BlockTables._make_ptr_tensor
    module.install_gcu_v2_buffers()
    assert BlockTables._make_ptr_tensor is ptr
    cpu = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    assert ptr(cpu, [1, 2]) == [1, 2]

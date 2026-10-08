# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_fl.dispatch.backends.vendor.ascend.patches import accelerator_compat
from vllm_fl.dispatch.backends.vendor.ascend.patches.accelerator_compat import (
    patch_accelerator_memory,
)


def _fake_memory_stack(monkeypatch):
    def broken_allocator(*args, **kwargs):
        raise AssertionError("DeviceAllocator is not initialized")

    accelerator = SimpleNamespace(
        current_accelerator=lambda: SimpleNamespace(type="npu"),
        **{name: broken_allocator for name in accelerator_compat._MEMORY_APIS},
    )
    npu = SimpleNamespace(
        **{
            npu_name: (lambda *args, **kwargs: 7)
            for npu_name in accelerator_compat._MEMORY_APIS.values()
        }
    )
    npu.mem_get_info = lambda *args, **kwargs: (100, 200)
    fake_torch = SimpleNamespace(
        __version__="2.10.0+cpu", accelerator=accelerator, npu=npu
    )
    monkeypatch.setattr(accelerator_compat, "torch", fake_torch)
    return accelerator, npu


def test_allocator_bridge_waits_for_all_npu_functions(monkeypatch):
    accelerator, npu = _fake_memory_stack(monkeypatch)
    originals = {
        name: getattr(accelerator, name) for name in accelerator_compat._MEMORY_APIS
    }
    del npu.mem_get_info  # The last entry must be resolved before any publish.

    assert accelerator_compat.patch_accelerator_memory() is False
    assert all(
        getattr(accelerator, name) is original for name, original in originals.items()
    )


def test_allocator_bridge_rolls_back_and_can_retry(monkeypatch):
    accelerator, _npu = _fake_memory_stack(monkeypatch)

    class RejectOneAssignment:
        def __init__(self, original):
            self.__dict__.update(vars(original))
            self.fail_once = True

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "memory_reserved" and getattr(self, "fail_once", False):
                object.__setattr__(self, "fail_once", False)
                raise RuntimeError("injected installation failure")

    accelerator = RejectOneAssignment(accelerator)
    accelerator_compat.torch.accelerator = accelerator
    originals = {
        name: getattr(accelerator, name) for name in accelerator_compat._MEMORY_APIS
    }

    assert accelerator_compat.patch_accelerator_memory() is False
    assert all(
        getattr(accelerator, name) is original for name, original in originals.items()
    )

    assert accelerator_compat.patch_accelerator_memory() is True
    installed = {
        name: getattr(accelerator, name) for name in accelerator_compat._MEMORY_APIS
    }
    assert installed["memory_allocated"]() == 7
    assert installed["get_memory_info"]() == (100, 200)
    assert installed["empty_cache"].__wrapped__ is originals["empty_cache"]

    assert accelerator_compat.patch_accelerator_memory() is True
    assert all(
        getattr(accelerator, name) is replacement
        for name, replacement in installed.items()
    )


def test_allocator_bridge_preserves_existing_wrapper(monkeypatch):
    accelerator, _npu = _fake_memory_stack(monkeypatch)
    calls = []

    def owner_wrapper():
        calls.append("owner")
        raise AssertionError("DeviceAllocator is not initialized")

    accelerator.empty_cache = owner_wrapper
    assert accelerator_compat.patch_accelerator_memory() is True
    assert accelerator.empty_cache.__wrapped__ is owner_wrapper
    assert accelerator.empty_cache() == 7
    assert calls == ["owner"]


def test_allocator_bridge_skips_working_native_api(monkeypatch):
    accelerator, _npu = _fake_memory_stack(monkeypatch)
    accelerator.get_memory_info = lambda: (100, 200)
    accelerator.memory_allocated = lambda: 0
    original_empty_cache = accelerator.empty_cache

    assert accelerator_compat.patch_accelerator_memory() is False
    assert accelerator.empty_cache is original_empty_cache


def test_allocator_bridge_checks_allocator_capability(monkeypatch):
    accelerator, _npu = _fake_memory_stack(monkeypatch)
    accelerator.get_memory_info = lambda: (100, 200)
    accelerator.memory_allocated = lambda: 0
    accelerator_compat.torch._C = SimpleNamespace(
        _accelerator_isAllocatorInitialized=lambda: False
    )

    assert accelerator_compat.patch_accelerator_memory() is True
    assert accelerator_compat._is_our_bridge(accelerator.get_memory_info)
    assert accelerator.get_memory_info() == (100, 200)


@pytest.mark.gpu
def test_vllm_memory_snapshot_on_ascend():
    from vllm.utils.mem_utils import MemorySnapshot

    patch_accelerator_memory()
    device = torch.device("npu", torch.npu.current_device())
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats(device)
    before = MemorySnapshot(device=device)
    allocation = torch.ones(1024 * 1024, device=device, dtype=torch.float32)
    after = MemorySnapshot(device=device)

    assert after.torch_allocated >= before.torch_allocated + allocation.nbytes
    assert after.torch_peak >= after.torch_allocated
    assert after.torch_memory >= after.torch_allocated
    assert 0 < after.free_memory <= after.total_memory
    del allocation
    torch.accelerator.empty_cache()

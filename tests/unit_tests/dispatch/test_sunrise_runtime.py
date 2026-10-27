# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU checks for Sunrise's vLLM 0.28 initialization and memory APIs."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.accelerator

from vllm_fl.dispatch.backends.vendor.sunrise import patch


@pytest.fixture
def ptpu_runtime(monkeypatch):
    platform_module = ModuleType("vllm.platforms")
    platform_module.current_platform = SimpleNamespace(device_type="ptpu")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform_module)
    runtime = SimpleNamespace(
        empty_cache=Mock(),
        memory_allocated=Mock(return_value=32),
        max_memory_allocated=Mock(return_value=48),
        memory_reserved=Mock(return_value=64),
        reset_peak_memory_stats=Mock(),
        mem_get_info=Mock(return_value=(192, 256)),
    )
    monkeypatch.setattr(torch, "ptpu", runtime, raising=False)
    monkeypatch.setattr(patch, "_patches_applied", False)
    # The patch assigns functions directly; register their original values with
    # monkeypatch so the real CPU/CUDA accelerator APIs are restored afterward.
    for name in (
        "empty_cache",
        "memory_stats",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "reset_peak_memory_stats",
        "get_memory_info",
    ):
        monkeypatch.setattr(
            torch.accelerator,
            name,
            getattr(torch.accelerator, name, None),
            raising=False,
        )
    monkeypatch.setattr(
        torch.accelerator, "_sunrise_memory_api_patched", False, raising=False
    )
    return runtime


def test_v028_memory_queries_reach_ptpu(ptpu_runtime):
    patch.patch_accelerator_memory_stats()
    patch.patch_accelerator_memory_api()

    device = 2
    stats = torch.accelerator.memory_stats(device)
    assert stats["allocated_bytes.all.peak"] == 48
    assert stats["allocated_bytes.all.current"] == 32
    # v0.28 adds only transient peak headroom to total device consumption.
    assert (
        stats["allocated_bytes.all.peak"] - stats["allocated_bytes.all.current"] == 16
    )
    assert stats["reserved_bytes.all.current"] == 64
    assert torch.accelerator.get_memory_info(device) == (192, 256)
    assert torch.accelerator.memory_reserved(device) == 64
    assert torch.accelerator.memory_allocated(device) == 32
    assert torch.accelerator.max_memory_allocated(device) == 48
    torch.accelerator.reset_peak_memory_stats(device)
    torch.accelerator.empty_cache()
    ptpu_runtime.mem_get_info.assert_called_once_with(device)
    ptpu_runtime.reset_peak_memory_stats.assert_called_once_with(device)
    ptpu_runtime.empty_cache.assert_called_once_with()


def test_native_ptpu_stats_are_preserved(ptpu_runtime):
    native_stats = Mock(return_value={"allocated_bytes.all.peak": 123})
    ptpu_runtime.memory_stats = native_stats

    patch.patch_accelerator_memory_stats()
    patch.patch_accelerator_memory_api()

    assert torch.accelerator.memory_stats(0) == {"allocated_bytes.all.peak": 123}
    assert ptpu_runtime.memory_stats is native_stats
    native_stats.assert_called_once_with(0)


def test_memory_counters_cannot_produce_negative_peak_headroom(ptpu_runtime):
    # Old runtimes sample whole-device use separately for the two counters.
    ptpu_runtime.memory_allocated.return_value = 60
    ptpu_runtime.max_memory_allocated.return_value = 48
    patch.patch_accelerator_memory_stats()

    stats = ptpu_runtime.memory_stats()

    assert stats["allocated_bytes.all.peak"] == 60
    assert stats["allocated_bytes.all.current"] == 60


def test_memory_adapter_is_idempotent(ptpu_runtime):
    patch.patch_accelerator_memory_stats()
    patch.patch_accelerator_memory_api()
    stats_fn = torch.accelerator.memory_stats
    memory_info_fn = torch.accelerator.get_memory_info

    patch.patch_accelerator_memory_stats()
    patch.patch_accelerator_memory_api()

    assert torch.accelerator.memory_stats is stats_fn
    assert torch.accelerator.get_memory_info is memory_info_fn


@pytest.mark.parametrize("device_type", ["cuda", "npu", "cpu", "musa"])
def test_sunrise_patches_do_not_change_other_platforms(monkeypatch, device_type):
    platform_module = ModuleType("vllm.platforms")
    platform_module.current_platform = SimpleNamespace(device_type=device_type)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform_module)
    monkeypatch.setattr(patch, "_patches_applied", False)
    memory_stats_fn = torch.accelerator.memory_stats
    memory_info_fn = torch.accelerator.get_memory_info

    patch.apply_sunrise_patches()
    patch.patch_accelerator_memory_stats()
    patch.patch_accelerator_memory_api()

    assert torch.accelerator.memory_stats is memory_stats_fn
    assert torch.accelerator.get_memory_info is memory_info_fn
    assert patch._patches_applied is False


def test_runtime_memory_is_ready_before_model_patches(monkeypatch, ptpu_runtime):
    def check_memory_api():
        assert torch.accelerator.get_memory_info() == (192, 256)
        assert torch.accelerator.memory_stats()["allocated_bytes.all.current"] == 32

    graph_patch = Mock(side_effect=check_memory_api)
    stream_patch = Mock(side_effect=check_memory_api)
    op_patch = Mock(side_effect=check_memory_api)
    monkeypatch.setattr(patch, "patch_graph_capture_runtime", graph_patch)
    monkeypatch.setattr(patch, "patch_flagcx_stream_adapter", stream_patch)
    monkeypatch.setattr(patch, "patch_distributed_runtime", Mock())
    monkeypatch.setattr(patch, "patch_op_cls", op_patch)

    patch.apply_sunrise_patches()
    patch.apply_sunrise_patches()

    graph_patch.assert_called_once_with()
    stream_patch.assert_called_once_with()
    op_patch.assert_called_once_with()


def test_sunrise_skips_cuda_only_quant_kernel_import(ptpu_runtime, monkeypatch):
    import vllm_fl

    class UnavailableCudaKernels(ModuleType):
        def __getattr__(self, name):
            raise AssertionError("Sunrise must not import CUDA quant kernels")

    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.quantization.quant_linear",
        UnavailableCudaKernels("vllm_fl.quantization.quant_linear"),
    )
    vllm_fl.register_quant_linear()

# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise test fixtures must select real PTPU devices instead of the CPU."""

import builtins
import sys
from types import SimpleNamespace

import pytest

from tests.utils import device_utils


@pytest.fixture
def sunrise_runtime(monkeypatch):
    monkeypatch.setattr(device_utils, "_BACKEND", "")
    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    runtime = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    monkeypatch.setattr(device_utils.torch, "ptpu", runtime, raising=False)
    return runtime


def test_sunrise_is_detected_without_cuda(sunrise_runtime):
    assert device_utils.get_backend() == "sunrise"
    assert device_utils.is_accelerator_available()
    assert device_utils.get_device_count() == 2


def test_unregistered_sunrise_runtime_is_loaded_lazily(monkeypatch, sunrise_runtime):
    monkeypatch.delattr(device_utils.torch, "ptpu")
    original_import = builtins.__import__

    def import_runtime(name, *args, **kwargs):
        if name == "torch_ptpu":
            device_utils.torch.ptpu = sunrise_runtime
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_runtime)

    assert device_utils.get_backend() == "sunrise"
    assert device_utils.get_device_count() == 2


def test_sunrise_device_uses_requested_ptpu_index(monkeypatch, sunrise_runtime):
    # CPU-only Torch does not recognize ptpu; record the device constructor.
    requested = []
    monkeypatch.setattr(
        device_utils.torch, "device", lambda name: requested.append(name) or name
    )
    assert device_utils.get_device(1) == "ptpu:1"
    assert requested == ["ptpu:1"]


def test_explicit_sunrise_without_devices_is_unavailable(monkeypatch, sunrise_runtime):
    monkeypatch.setattr(device_utils, "_BACKEND", "sunrise")
    sunrise_runtime.is_available = lambda: False
    assert device_utils.get_backend() == "sunrise"
    assert not device_utils.is_accelerator_available()
    assert device_utils.get_device_count() == 0


def test_missing_sunrise_runtime_falls_back_to_cpu(monkeypatch, sunrise_runtime):
    monkeypatch.delattr(device_utils.torch, "ptpu")
    monkeypatch.setitem(sys.modules, "torch_ptpu", None)
    assert device_utils.get_backend() == "cpu"
    assert device_utils.get_device_count() == 0


def test_explicit_sunrise_without_runtime_fails_device_selection(
    monkeypatch, sunrise_runtime
):
    monkeypatch.setattr(device_utils, "_BACKEND", "sunrise")
    monkeypatch.delattr(device_utils.torch, "ptpu")
    monkeypatch.setitem(sys.modules, "torch_ptpu", None)
    assert not device_utils.is_accelerator_available()
    assert device_utils.get_device_count() == 0
    with pytest.raises(RuntimeError, match="torch_ptpu/PTPU"):
        device_utils.get_device()


def test_cuda_detection_keeps_priority(monkeypatch, sunrise_runtime):
    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    assert device_utils.get_backend() == "nvidia"

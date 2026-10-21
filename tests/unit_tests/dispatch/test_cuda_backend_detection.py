# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for NVIDIA vendor backend detection on vLLM CUDA platforms."""

from unittest.mock import Mock

import torch
from vllm import platforms

from vllm_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend
from vllm_fl.dispatch.builtin_ops import _get_current_vendor_backend_dirs


def test_in_tree_cuda_platform_selects_cuda_vendor_backend(monkeypatch):
    platform = Mock()
    platform.vendor_name = None
    platform.device_name = "cuda"
    platform.is_cuda.return_value = True
    monkeypatch.setattr(platforms, "current_platform", platform)

    assert _get_current_vendor_backend_dirs({"cuda", "ascend"}) == "cuda"


def test_cuda_backend_available_for_in_tree_cuda_platform(monkeypatch):
    platform = Mock()
    platform.device_name = "cuda"
    platform.is_cuda.return_value = True
    monkeypatch.setattr(platforms, "current_platform", platform)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(CudaBackend, "_available", None)

    assert CudaBackend().is_available()


def test_cuda_alike_platform_does_not_select_nvidia_backend(monkeypatch):
    platform = Mock()
    platform.vendor_name = None
    platform.device_name = "cuda"
    platform.is_cuda.return_value = False
    monkeypatch.setattr(platforms, "current_platform", platform)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(CudaBackend, "_available", None)

    assert _get_current_vendor_backend_dirs({"cuda", "ascend"}) is None
    assert not CudaBackend().is_available()

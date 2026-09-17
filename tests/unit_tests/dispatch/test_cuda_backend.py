# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

from vllm_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend


def test_cuda_backend_uses_vendor_identity_not_device_label(monkeypatch):
    monkeypatch.setattr(CudaBackend, "_available", None)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(
                vendor_name="nvidia",
                device_name="NVIDIA A800-SXM4-80GB",
            ),
        ),
    ):
        assert CudaBackend().is_available()


def test_cuda_backend_rejects_cuda_alike_non_nvidia_vendor(monkeypatch):
    monkeypatch.setattr(CudaBackend, "_available", None)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(vendor_name="metax", device_name="cuda"),
        ),
    ):
        assert not CudaBackend().is_available()

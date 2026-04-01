# Copyright (c) 2025 BAAI. All rights reserved.

"""
Backend-agnostic device utilities for tests.

Provides a unified interface for device detection, selection, and test skipping
across multiple chip backends (NVIDIA CUDA, Huawei Ascend, Tianshu, etc.).

The backend can be explicitly set via the FL_BACKEND environment variable,
or auto-detected from available hardware.
"""

import os

import pytest
import torch

_BACKEND = os.environ.get("FL_BACKEND", "").lower()


def get_backend() -> str:
    """Detect current backend from env or hardware.

    Priority:
        1. FL_BACKEND environment variable (set by CI workflow)
        2. Auto-detect from available hardware
    """
    if _BACKEND:
        return _BACKEND
    if torch.cuda.is_available():
        return "nvidia"
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "ascend"
    except ImportError:
        pass  # torch_npu not installed; ascend backend not available
    return "cpu"


def is_accelerator_available() -> bool:
    """Check if any accelerator (GPU/NPU) is available."""
    return get_backend() not in ("cpu", "")


def get_device(index: int = 0) -> torch.device:
    """Get torch device for current backend."""
    backend = get_backend()
    if backend == "nvidia":
        return torch.device(f"cuda:{index}")
    elif backend == "ascend":
        return torch.device(f"npu:{index}")
    elif backend == "tianshu":
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def get_device_count() -> int:
    """Get the number of available accelerators."""
    backend = get_backend()
    if backend == "nvidia":
        return torch.cuda.device_count()
    elif backend == "ascend":
        try:
            import torch_npu  # noqa: F401

            return torch.npu.device_count()
        except ImportError:
            return 0
    elif backend == "tianshu":
        return torch.cuda.device_count()
    return 0


def get_model_base_path() -> str:
    """Get base path for test models from env or default."""
    return os.environ.get("FL_MODEL_BASE_PATH", "/data/models")


skip_if_no_accelerator = pytest.mark.skipif(
    not is_accelerator_available(), reason="Accelerator not available"
)

skip_if_not_multi_accelerator = pytest.mark.skipif(
    not is_accelerator_available() or get_device_count() < 2,
    reason="Multiple accelerators not available",
)

skip_if_not_nvidia = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA-specific test"
)

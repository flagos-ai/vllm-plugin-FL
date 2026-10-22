# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unit test fixtures and configuration.

Note: Common fixtures (device, cpu_device, mock_tensor, has_accelerator,
markers) are inherited from the root tests/conftest.py.
Only unit-test-specific fixtures belong here.
"""

import os
from unittest.mock import MagicMock, NonCallableMagicMock

import pytest
import torch

# =============================================================================
# Environment Detection Helpers
# =============================================================================


def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def has_flagcx():
    """Check if flagcx is available."""
    flagcx_path = os.getenv("FLAGCX_PATH")
    if not flagcx_path:
        return False
    lib_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
    return os.path.exists(lib_path)


def has_vllm_profiler():
    """Check if vllm profiler is available."""
    try:
        from vllm.profiler.wrapper import TorchProfilerWrapper  # noqa: F401

        return True
    except ImportError:
        return False


# =============================================================================
# Mock Factory Fixtures
# =============================================================================


@pytest.fixture
def mock_module():
    """
    Create a mock that behaves like a Python module.

    Use this when you need to mock a module object that:
    - Is not callable
    - May have specific attributes
    """
    return NonCallableMagicMock(spec=["__name__", "__file__"])


@pytest.fixture
def mock_module_with_register():
    """
    Create a mock module with a register function.

    Useful for testing plugin discovery.
    """
    module = NonCallableMagicMock(spec=["register"])
    module.register = MagicMock()
    return module


@pytest.fixture
def mock_process_group():
    """
    Create a mock torch distributed ProcessGroup.

    Useful for testing distributed communication.
    """
    group = MagicMock()
    group.rank.return_value = 0
    group.size.return_value = 1
    return group


# =============================================================================
# Tensor Fixtures
# =============================================================================


@pytest.fixture
def batch_tensors():
    """Create a batch of tensors for testing."""
    return {
        "small": torch.randn(2, 8),
        "medium": torch.randn(4, 16, 32),
        "large": torch.randn(8, 32, 64, 128),
    }


@pytest.fixture
def dtype_tensors():
    """Create tensors with different dtypes."""
    return {
        "float32": torch.randn(2, 4, dtype=torch.float32),
        "float16": torch.randn(2, 4, dtype=torch.float16),
        "bfloat16": torch.randn(2, 4, dtype=torch.bfloat16),
    }

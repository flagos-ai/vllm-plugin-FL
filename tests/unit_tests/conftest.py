# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unit test fixtures and configuration.

This module provides shared fixtures for all unit tests.
"""

import os
import pytest
import torch
from unittest.mock import MagicMock, NonCallableMagicMock


# =============================================================================
# Environment Detection Helpers
# =============================================================================

def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def has_flagcx():
    """Check if flagcx is available."""
    flagcx_path = os.getenv('FLAGCX_PATH')
    if not flagcx_path:
        return False
    lib_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
    return os.path.exists(lib_path)


def has_vllm_profiler():
    """Check if vllm profiler is available."""
    try:
        from vllm.profiler.wrapper import TorchProfilerWrapper
        return True
    except ImportError:
        return False


# =============================================================================
# Basic Fixtures
# =============================================================================

@pytest.fixture
def mock_tensor():
    """Create a simple tensor for testing."""
    return torch.randn(2, 4, 8)


@pytest.fixture
def device():
    """Get the available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cpu_device():
    """Always return CPU device."""
    return torch.device("cpu")


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
    return NonCallableMagicMock(spec=['__name__', '__file__'])


@pytest.fixture
def mock_module_with_register():
    """
    Create a mock module with a register function.

    Useful for testing plugin discovery.
    """
    module = NonCallableMagicMock(spec=['register'])
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
        'small': torch.randn(2, 8),
        'medium': torch.randn(4, 16, 32),
        'large': torch.randn(8, 32, 64, 128),
    }


@pytest.fixture
def dtype_tensors():
    """Create tensors with different dtypes."""
    return {
        'float32': torch.randn(2, 4, dtype=torch.float32),
        'float16': torch.randn(2, 4, dtype=torch.float16),
        'bfloat16': torch.randn(2, 4, dtype=torch.bfloat16),
    }


# =============================================================================
# Pytest Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "flagcx: marks tests as requiring flagcx"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )

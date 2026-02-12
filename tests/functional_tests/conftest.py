# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional test fixtures and configuration.
"""

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "multi_gpu: marks tests as requiring multiple GPUs")
    config.addinivalue_line("markers", "flaggems: marks tests as requiring flag_gems library")


@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(has_gpu):
    """Get the test device."""
    if has_gpu:
        return torch.device("cuda:0")
    return torch.device("cpu")

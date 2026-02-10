# Copyright (c) 2025 BAAI. All rights reserved.

"""
Pytest fixtures for inference tests.
"""

import pytest
from .vllm_runner import VllmRunner


@pytest.fixture(scope="session")
def vllm_runner():
    """Provide VllmRunner class as a fixture."""
    return VllmRunner

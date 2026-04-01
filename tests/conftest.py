# Copyright (c) 2025 BAAI. All rights reserved.

"""
Global pytest fixtures and configuration for all tests.

This root-level conftest is loaded by pytest before any sub-package conftest.
It provides:
  - Backend-agnostic device fixtures (via tests/utils/device_utils)
  - Platform-aware fixtures: tolerance, platform_config (via run.py)
  - Gold-value comparison support (--save-gold flag)
  - Custom markers auto-registration
  - Common skip helpers
  - Session-scoped shared resources (model paths, temp dirs, etc.)
"""

import os
import tempfile

import pytest
import torch

from tests.utils.device_utils import (
    get_backend,
    get_device,
    get_device_count,
    get_model_base_path,
    is_accelerator_available,
)

# ---------------------------------------------------------------------------
# CLI options (injected by run.py)
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Register custom CLI options passed by run.py."""
    parser.addoption(
        "--platform",
        default=os.environ.get("FL_TEST_PLATFORM", ""),
        help="Platform name (e.g., cuda, ascend)",
    )
    parser.addoption(
        "--device",
        default=os.environ.get("FL_TEST_DEVICE", ""),
        help="Device type (e.g., a100, 910b)",
    )
    parser.addoption(
        "--save-gold",
        action="store_true",
        default=False,
        help="Save test outputs as gold values instead of comparing",
    )


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register all custom markers in a single place."""
    for line in [
        "gpu: marks tests as requiring a single GPU / accelerator",
        "multi_gpu: marks tests as requiring multiple GPUs / accelerators",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
        "integration: marks tests as integration tests",
        "e2e: marks tests as end-to-end tests",
        "flaggems: marks tests as requiring the flag_gems library",
        "flagcx: marks tests as requiring the FlagCX library",
        "functional: marks tests as functional tests",
    ]:
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that carry hardware markers when hardware is absent."""
    skip_gpu = pytest.mark.skip(reason="No accelerator available")
    skip_multi = pytest.mark.skip(reason="Multiple accelerators not available")

    accel = is_accelerator_available()
    multi = accel and get_device_count() >= 2

    for item in items:
        if "gpu" in item.keywords and not accel:
            item.add_marker(skip_gpu)
        if "multi_gpu" in item.keywords and not multi:
            item.add_marker(skip_multi)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def backend():
    """Return the detected backend name (e.g. 'nvidia', 'ascend', 'cpu')."""
    return get_backend()


@pytest.fixture(scope="session")
def has_accelerator():
    """Whether any accelerator is available."""
    return is_accelerator_available()


@pytest.fixture(scope="session")
def device():
    """Return a ``torch.device`` for the current backend."""
    return get_device()


@pytest.fixture(scope="session")
def device_count():
    """Number of available accelerators."""
    return get_device_count()


@pytest.fixture(scope="session")
def model_base_path():
    """Base directory that holds test model checkpoints."""
    return get_model_base_path()


# ---------------------------------------------------------------------------
# Function-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_device():
    """Always return CPU device."""
    return torch.device("cpu")


@pytest.fixture
def mock_tensor(device):
    """Create a simple tensor on the test device."""
    return torch.randn(2, 4, 8, device=device)


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a per-test temporary directory (shorthand for ``tmp_path``)."""
    return tmp_path


@pytest.fixture(scope="session")
def shared_tmp_dir():
    """Session-scoped temporary directory, cleaned up at the end."""
    d = tempfile.mkdtemp(prefix="vllm_fl_test_")
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Environment helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env_override(monkeypatch):
    """Return a helper to temporarily set environment variables.

    Usage::

        def test_something(env_override):
            env_override("MY_VAR", "1")
            ...
    """

    def _set(key: str, value: str):
        monkeypatch.setenv(key, value)

    return _set


@pytest.fixture
def modelscope_env(monkeypatch):
    """Enable ModelScope for the duration of the test."""
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "True")


# ---------------------------------------------------------------------------
# Platform-aware fixtures (injected by run.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def platform_config(request):
    """Load PlatformConfig if --platform was provided, else None."""
    platform = request.config.getoption("--platform", default="")
    if not platform:
        return None
    from tests.utils.platform_config import PlatformConfig

    device = request.config.getoption("--device", default=None)
    return PlatformConfig.load(platform, device or None)


@pytest.fixture(scope="session")
def platform_name(request):
    """Return the platform name (e.g., 'cuda', 'ascend'), or ''."""
    return request.config.getoption("--platform", default="")


@pytest.fixture(scope="session")
def device_type(request):
    """Return the device type (e.g., 'a100', '910b'), or ''."""
    return request.config.getoption("--device", default="")


@pytest.fixture(scope="session")
def tolerance(platform_config):
    """Return a helper to look up tolerance for a category/dtype.

    Usage in tests::

        def test_inference(tolerance):
            tol = tolerance("inference", "bfloat16")
            assert numpy.allclose(actual, expected, rtol=tol.rtol, atol=tol.atol)
    """
    if platform_config is None:
        from tests.utils.platform_config import Tolerance

        return lambda category="inference", dtype="default": Tolerance()

    def _get(category: str = "inference", dtype: str = "default"):
        return platform_config.get_tolerance(category, dtype)

    return _get


@pytest.fixture(scope="session")
def save_gold_mode(request):
    """Return True if --save-gold was passed."""
    return request.config.getoption("--save-gold", default=False)


@pytest.fixture(scope="session")
def model_config():
    """Return a helper to load shared model configs.

    Usage::

        def test_something(model_config):
            cfg = model_config("qwen3_next")
            llm = LLM(**cfg.engine_kwargs())
    """
    from tests.utils.model_config import ModelConfig

    return ModelConfig.load

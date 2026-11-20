# Copyright (c) 2025 BAAI. All rights reserved.

"""Pytest configuration for the supplementary coverage suite.

Registers the ``supplementary`` marker locally so this directory can be
collected even without changes to the root conftest or pyproject marker
list, and provides small shared fixtures specific to supplementary tests.

The root ``tests/conftest.py`` fixtures (``device``, ``env_override``, ...)
remain available because pytest loads parent conftests as well.
"""

import pytest


def pytest_configure(config):
    """Register the supplementary marker for this test tree only."""
    config.addinivalue_line(
        "markers",
        "supplementary: supplementary coverage test (idle-time/weekend runs only)",
    )
    config.addinivalue_line(
        "markers",
        "area(name): area tag used by the per-area coverage report",
    )


@pytest.fixture
def clean_fl_env(monkeypatch):
    """Return a helper that clears every ``VLLM_FL_*`` env var for a test.

    Restores the previous values on teardown so supplementary tests never
    leak environment state into each other.
    """
    import os

    fl_vars = [k for k in os.environ if k.startswith("VLLM_FL_")]
    for key in fl_vars:
        monkeypatch.delenv(key, raising=False)

    def _set(key: str, value: str):
        monkeypatch.setenv(key, value)

    return _set

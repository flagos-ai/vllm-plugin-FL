# Copyright 2025 BAAI. All rights reserved.

"""Pytest configuration for the multi-platform supplementary coverage suite.

Registers the ``supplementary`` marker locally so this directory can be
collected even without changes to the root conftest or pyproject marker
list, and provides small shared fixtures specific to supplementary tests.

The root ``tests/conftest.py`` fixtures (``device``, ``env_override``, ...)
remain available because pytest loads parent conftests as well.
"""

from __future__ import annotations

import os

import pytest

#: Platforms this suite carries vendor-specific supplementary tests for.
SUPPORTED_PLATFORMS = ("kunlunxin", "hygon", "iluvatar", "thead")


def _active_platform(config) -> str:
    """Resolve the platform this pytest session was started for.

    Priority matches the root conftest's ``--platform`` option default:
    the CLI option (injected by the runners) first, then the
    ``FL_TEST_PLATFORM`` environment variable the runners also export.
    """
    value = ""
    try:
        value = config.getoption("--platform") or ""
    except (ValueError, KeyError):
        # option not registered (bare pytest without the runner)
        pass
    return value or os.environ.get("FL_TEST_PLATFORM", "")


def pytest_configure(config):
    """Register the supplementary markers for this test tree only."""
    config.addinivalue_line(
        "markers",
        "supplementary: supplementary coverage test (idle-time/weekend runs only)",
    )
    config.addinivalue_line(
        "markers",
        "area(name): area tag used by the per-area coverage report",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests gated to platforms other than the one under test.

    A test marked with ``pytest.mark.platforms("hygon")`` only runs when the
    session's active platform is one of the listed names. Tests with no
    ``platforms`` marker are platform-agnostic and always collected; they
    must not depend on vendor hardware by construction (stub-driven).
    """
    active = _active_platform(config)
    for item in items:
        marker = item.get_closest_marker("platforms")
        if marker is None:
            continue
        allowed = marker.args
        if active not in allowed:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"supplementary platform gate: requires one of "
                    f"{list(allowed)}, session platform is {active or 'unset'!r}"
                )
            )


@pytest.fixture
def clean_fl_env(monkeypatch):
    """Return a helper that clears every ``VLLM_FL_*`` env var for a test.

    Restores the previous values on teardown so supplementary tests never
    leak environment state into each other.
    """
    fl_vars = [k for k in os.environ if k.startswith("VLLM_FL_")]
    for key in fl_vars:
        monkeypatch.delenv(key, raising=False)

# Copyright (c) 2026 BAAI. All rights reserved.

import os

import pytest

from tests.functional_tests import conftest as functional_config


@pytest.mark.parametrize("backend", ["enflame", "nvidia", "cpu"])
@pytest.mark.parametrize("exitstatus", [0, 1])
def test_non_ascend_functional_session_can_finalize_reports(
    monkeypatch, backend, exitstatus
):
    monkeypatch.setattr(functional_config, "get_backend", lambda: backend)

    def unexpected_exit(status):
        pytest.fail(f"Functional session terminated before reports: {status}")

    monkeypatch.setattr(os, "_exit", unexpected_exit)
    functional_config.pytest_sessionfinish(None, exitstatus)

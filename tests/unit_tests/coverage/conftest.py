# Copyright 2025 BAAI. All rights reserved.

"""Pytest configuration for the multi-platform coverage suite.

This directory holds supplementary coverage tests that live inside the
standard ``tests/unit_tests`` discovery tree (collected automatically by
``tests/run.py --scope unit``). Everything that used to live in the former
``tests/supplementary_tests/conftest.py`` has moved to the appropriate
parent during the standard-tree migration:

* ``supplementary`` / ``platforms`` / ``area`` marker registration —
  now in the root ``tests/conftest.py`` ``pytest_configure``;
* the ``platforms`` collection gate — now the ``_platforms_gate`` step in
  the root ``tests/conftest.py`` ``pytest_collection_modifyitems`` (same
  shape as the ``gpu`` hardware skip, and it also covers
  ``tests/functional_tests/coverage/``).

What remains here is only the fixture specific to this sub-tree. The root
``tests/conftest.py`` fixtures (``device``, ``env_override``, ...) and the
``tests/unit_tests/conftest.py`` mock factories stay available because
pytest loads every parent conftest as well.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def clean_fl_env(monkeypatch):
    """Return a helper that clears every ``VLLM_FL_*`` env var for a test.

    Restores the previous values on teardown so coverage tests never
    leak environment state into each other.
    """
    fl_vars = [k for k in os.environ if k.startswith("VLLM_FL_")]
    for key in fl_vars:
        monkeypatch.delenv(key, raising=False)

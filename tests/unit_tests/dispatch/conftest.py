# Copyright (c) 2026 BAAI. All rights reserved.

"""
Shared fixtures for dispatch unit tests.

* Attaches ``caplog`` to dispatch loggers that have ``propagate=False``
  so that ``caplog``-based assertions work in IO inspector/dumper tests.
  The dispatch logger manager configures ``vllm_fl.dispatch`` (and
  per-module children such as ``vllm_fl.dispatch.io_dump``) with
  ``propagate=False`` plus a private stderr handler as soon as
  ``logger_manager.get_logger`` is called (which happens whenever
  ``builtin_ops`` / ``discovery`` are imported); records emitted by
  ``vllm_fl.dispatch.io_common`` therefore never reach pytest's
  root-attached caplog handler unless attached directly.
* ``clean_fl_env`` clears every ``VLLM_FL_*`` env var for tests that
  assert default-config behaviour and need a deterministic baseline.
"""

import logging
import os

import pytest

_LOGGER_NAMES = [
    "vllm_fl.dispatch",
    "vllm_fl.dispatch.io_common",
    "vllm_fl.dispatch.io_dump",
    "vllm_fl.dispatch.io_print",
    "vllm_fl.dispatch.logger_manager",
]


@pytest.fixture(autouse=True)
def _attach_caplog_to_dispatch_loggers(caplog):
    """Attach the caplog handler directly to dispatch loggers.

    Attaching to the loggers themselves (rather than relying on root
    propagation) keeps ``caplog`` assertions working regardless of whether
    ``logger_manager`` has already disabled ``propagate`` on the logger or
    one of its ancestors.
    """
    attached = []
    for name in _LOGGER_NAMES:
        lgr = logging.getLogger(name)
        if caplog.handler not in lgr.handlers:
            lgr.addHandler(caplog.handler)
            attached.append(lgr)
    yield
    for lgr in attached:
        lgr.removeHandler(caplog.handler)


@pytest.fixture
def clean_fl_env(monkeypatch):
    """Return a helper that clears every ``VLLM_FL_*`` env var for a test.

    Restores the previous values on teardown so tests never leak
    environment state into each other.
    """
    fl_vars = [k for k in os.environ if k.startswith("VLLM_FL_")]
    for key in fl_vars:
        monkeypatch.delenv(key, raising=False)

    def _set(key: str, value: str):
        monkeypatch.setenv(key, value)

    return _set

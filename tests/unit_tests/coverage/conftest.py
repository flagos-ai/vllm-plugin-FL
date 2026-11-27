# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared fixtures for the coverage-focused unit tests.

Merges the isolation machinery the migrated tests rely on (previously
spread across worker/ops/quantization/patches conftests):

* ``fl_sys_modules`` — snapshot/restore of sys.modules entries under
  vllm/vllm_fl/flag_gems (+ torch_npu where relevant) so stub installs
  never leak across files. Autouse: some moved files install module-level
  stubs at collection time.
* ``clean_fl_env`` — clears every VLLM_FL_* env var per test.
* caplog attachment for the five dispatch loggers that set
  propagate=False.
"""

import logging
import os
import sys

import pytest

_STUB_ROOTS = ("vllm", "vllm_fl", "flag_gems", "torch_npu")

_MISSING = object()


@pytest.fixture(autouse=True)
def fl_sys_modules():
    """Snapshot ``sys.modules`` and restore it exactly after the test."""
    saved = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_ROOTS:
            saved[name] = sys.modules.get(name, _MISSING)
    try:
        yield saved
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name.split(".")[0] in _STUB_ROOTS and name not in saved
        ]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def clean_fl_env(monkeypatch):
    """Return a helper that clears every ``VLLM_FL_*`` env var for a test."""
    fl_vars = [k for k in os.environ if k.startswith("VLLM_FL_")]
    for key in fl_vars:
        monkeypatch.delenv(key, raising=False)

    def _set(key: str, value: str):
        monkeypatch.setenv(key, value)

    return _set


_LOGGER_NAMES = [
    "vllm_fl.dispatch",
    "vllm_fl.dispatch.io_common",
    "vllm_fl.dispatch.io_dump",
    "vllm_fl.dispatch.io_print",
    "vllm_fl.dispatch.logger_manager",
]


@pytest.fixture(autouse=True)
def _attach_caplog_to_dispatch_loggers(caplog):
    """Attach caplog directly to dispatch loggers with propagate=False."""
    attached = []
    for name in _LOGGER_NAMES:
        lgr = logging.getLogger(name)
        if caplog.handler not in lgr.handlers:
            lgr.addHandler(caplog.handler)
            attached.append(lgr)
    yield
    for lgr in attached:
        lgr.removeHandler(caplog.handler)

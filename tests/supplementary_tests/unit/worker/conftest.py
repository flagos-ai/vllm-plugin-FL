# Copyright (c) 2025 BAAI. All rights reserved.

"""Shared fixtures for supplementary worker-side unit tests.

The only machinery defined here is strict ``sys.modules`` isolation: the
modules under test (``vllm_fl.worker.runtime``, ``vllm_fl.worker.scheduler_fl``,
``vllm_fl.patches.*``) import the external ``vllm`` / ``flag_gems`` packages,
which are absent on CPU-only test machines.  Each test installs lightweight
stubs for those *external boundaries* and this fixture guarantees that no
stub, nor any ``vllm_fl`` submodule imported as a side effect, leaks into a
later test.
"""

import sys

import pytest

_STUB_ROOTS = ("vllm", "vllm_fl", "flag_gems")

_MISSING = object()


@pytest.fixture
def fl_sys_modules():
    """Snapshot ``sys.modules`` and restore it exactly after the test.

    Records every ``vllm*`` / ``vllm_fl*`` / ``flag_gems*`` entry present
    before the test, removes entries the test added, and puts back the
    original module objects for entries the test replaced.  This keeps tests
    order-independent even though they import plugin submodules lazily and
    install fake ``vllm`` packages.
    """
    saved = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_ROOTS or name == "vllm":
            saved[name] = sys.modules.get(name, _MISSING)
    try:
        yield saved
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if (name.split(".")[0] in _STUB_ROOTS or name == "vllm")
            and name not in saved
        ]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

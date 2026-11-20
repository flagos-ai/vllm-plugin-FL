# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared fixtures for supplementary ops unit tests.

The modules under test (``vllm_fl.ops.custom_ops``, ``vllm_fl.ops.fla.*``)
import the external ``vllm`` package at module level, which is absent on
CPU-only test machines.  The tests install stub ``vllm`` graphs — some at
module (collection) time — so this conftest guarantees that no stub, nor
any ``vllm_fl`` submodule imported as a side effect, leaks into a later
test outside this directory.
"""

import sys

import pytest

_STUB_ROOTS = ("vllm", "vllm_fl", "flag_gems")

_MISSING = object()


@pytest.fixture(autouse=True)
def fl_sys_modules():
    """Snapshot ``sys.modules`` and restore it exactly after the test.

    Records every ``vllm*`` / ``vllm_fl*`` / ``flag_gems*`` entry present
    before the test, removes entries the test added, and puts back the
    original module objects for entries the test replaced.  Module-level
    stub installs done at collection time are captured in the first
    snapshot and re-instated after every test, so later files see the
    same environment this directory's tests were written against.
    """
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

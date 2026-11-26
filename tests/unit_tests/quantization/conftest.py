# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared fixtures for quantization unit tests.

``vllm_fl.quantization.quant_linear`` imports ``vllm.platforms`` at module
level, which is absent on CPU-only test machines.  The tests install a stub
``vllm`` graph and reload the module under test; this conftest guarantees
that neither the stubs nor the reloaded ``vllm_fl`` submodules leak into a
later test outside this directory.
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

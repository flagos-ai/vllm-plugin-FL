# Copyright (c) 2026 BAAI. All rights reserved.

"""Platform guard for the HY4 unit tests.

The HY4 tests (main #453, H100 FlagOS support) target the NVIDIA
platform and carry no platform guard of their own: on any other vendor
their platform mocks are incomplete and the tests raise instead of
skipping. Under this branch's CI every platform runs the full unit
tree, so keep them out of collection outside NVIDIA.

Guarding at collection time (collect_ignore_glob) rather than by
marking collected items: a file that fails during import produces
collection errors no marker can reach, and on older vendor triton
stacks the HY4 import chain does exactly that.

``FL_TEST_PLATFORM`` is always set by ``tests/run.py``; local runs
without it default to collecting the tests (NVIDIA developers).
"""

import os

import pytest

_PLATFORM = os.environ.get("FL_TEST_PLATFORM", "nvidia").lower()

if _PLATFORM != "nvidia":
    collect_ignore_glob = ["test_hy_v4*"]


def pytest_collection_modifyitems(config, items):
    # Kept for reporting: anything that still slips through collection
    # (e.g. a hy_v4 file imported by another module) is skipped with an
    # explicit reason instead of running.
    if _PLATFORM == "nvidia":
        return
    skip = pytest.mark.skip(
        reason=f"HY4 tests target the NVIDIA platform (FL_TEST_PLATFORM={_PLATFORM})"
    )
    for item in items:
        basename = os.path.basename(str(item.fspath))
        if basename.startswith("test_hy_v4"):
            item.add_marker(skip)

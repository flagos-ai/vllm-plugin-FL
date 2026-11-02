# Copyright (c) 2026 BAAI. All rights reserved.

"""Platform guard for the HY4 functional tests.

Same policy as tests/unit_tests/patches/conftest.py: the HY4 tests
from main #453 target the NVIDIA platform. test_hy4_runtime.py skips
itself via ``current_platform.is_cuda()``, but that check is True on
every CUDA-shaped vendor, and test_portable_sparse_indexer.py asserts
``is_cuda()`` instead of skipping — worse, its import chain pulls
triton kernels that error during collection on older vendor stacks
(enflame's torch_gcu 2.11 triton lacks tl.math.pow), which no item
marker can reach. Keep both files out of collection outside NVIDIA.

``FL_TEST_PLATFORM`` is always set by ``tests/run.py``; local runs
without it default to collecting the tests (NVIDIA developers).
"""

import os

import pytest

_PLATFORM = os.environ.get("FL_TEST_PLATFORM", "nvidia").lower()

_HY4_FUNCTIONAL_TESTS = (
    "test_hy4_runtime.py",
    "test_portable_sparse_indexer.py",
    "test_portable_indexer_fp8_cache.py",
)

if _PLATFORM != "nvidia":
    collect_ignore = list(_HY4_FUNCTIONAL_TESTS)


def pytest_collection_modifyitems(config, items):
    # Kept for reporting: a HY4 test reached through another module's
    # import is skipped with an explicit reason instead of running.
    if _PLATFORM == "nvidia":
        return
    skip = pytest.mark.skip(
        reason=f"HY4 tests target the NVIDIA platform (FL_TEST_PLATFORM={_PLATFORM})"
    )
    for item in items:
        basename = os.path.basename(str(item.fspath))
        if basename in _HY4_FUNCTIONAL_TESTS:
            item.add_marker(skip)

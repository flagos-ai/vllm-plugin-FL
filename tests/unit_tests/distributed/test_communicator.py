# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for distributed communicator module.

Note: Tests require the flagcx package (or a source tree named by FLAGCX_PATH).
Tests are skipped if flagcx is not available.
"""

import os

import pytest


def has_flagcx():
    """Check if flagcx is available: the installed package, or a source tree
    named by FLAGCX_PATH."""
    try:
        import flagcx.api  # noqa: F401

        return True
    except ImportError:
        pass

    flagcx_path = os.getenv("FLAGCX_PATH")
    if not flagcx_path:
        return False
    for rel in ("lib/libflagcx.so", "build/lib/libflagcx.so"):
        if os.path.exists(os.path.join(flagcx_path, rel)):
            return True
    return False


# Skip all tests if flagcx is not available (communicator depends on it)
pytestmark = pytest.mark.skipif(
    not has_flagcx(), reason="flagcx package not installed and no FlagCX library found"
)


# Note: CommunicatorFL requires multi-process GPU environment for meaningful tests.
# Integration tests should be in functional_tests/.
# Unit tests here are minimal as the class requires distributed infrastructure.

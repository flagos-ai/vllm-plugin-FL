# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for flagcx communicator module.

Note: Tests require the flagcx package (or a source tree named by FLAGCX_PATH).
Tests are skipped if flagcx is not available.

Integration tests for actual distributed operations should be in functional_tests/.
"""

import os

import pytest


def has_flagcx():
    """Check if flagcx is available (both library and Python bindings): the
    installed package, or a source tree named by FLAGCX_PATH."""
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


# Mark all tests in this module as requiring flagcx
pytestmark = pytest.mark.skipif(
    not has_flagcx(),
    reason="flagcx package not installed and no FlagCX library found",
)


# Note: PyFlagcxCommunicator requires multi-GPU distributed environment for meaningful tests.
# Unit tests for dtype/op conversions are moved here but require the plugin module.
# Integration tests should be in functional_tests/.

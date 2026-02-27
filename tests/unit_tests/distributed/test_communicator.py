# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for distributed communicator module.

Note: Tests require FLAGCX_PATH environment variable to be set.
Tests are skipped if flagcx is not available.
"""

import os
import pytest


def has_flagcx():
    """Check if flagcx is available."""
    flagcx_path = os.getenv('FLAGCX_PATH')
    if not flagcx_path:
        return False
    lib_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
    return os.path.exists(lib_path)


# Skip all tests if flagcx is not available (communicator depends on it)
pytestmark = pytest.mark.skipif(
    not has_flagcx(),
    reason="FLAGCX_PATH not set or flagcx library not found"
)


# Note: CommunicatorFL requires multi-process GPU environment for meaningful tests.
# Integration tests should be in functional_tests/.
# Unit tests here are minimal as the class requires distributed infrastructure.

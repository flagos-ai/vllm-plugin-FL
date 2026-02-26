# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for flagcx communicator module.

Note: Tests require FLAGCX_PATH environment variable and the flagcx Python bindings.
Tests are skipped if flagcx is not available.

Integration tests for actual distributed operations should be in functional_tests/.
"""

import os
import pytest


def has_flagcx():
    """Check if flagcx is available (both library and Python bindings)."""
    flagcx_path = os.getenv('FLAGCX_PATH')
    if not flagcx_path:
        return False
    lib_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
    if not os.path.exists(lib_path):
        return False
    # Also check Python bindings
    try:
        from plugin.interservice.flagcx_wrapper import flagcxDataTypeEnum
        return True
    except ImportError:
        return False


# Mark all tests in this module as requiring flagcx
pytestmark = pytest.mark.skipif(
    not has_flagcx(),
    reason="FLAGCX_PATH not set, flagcx library not found, or Python bindings unavailable"
)


# Note: PyFlagcxCommunicator requires multi-GPU distributed environment for meaningful tests.
# Unit tests for dtype/op conversions are moved here but require the plugin module.
# Integration tests should be in functional_tests/.

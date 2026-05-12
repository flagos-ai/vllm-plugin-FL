# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unit tests for FlagGems ops discovery functionality.
"""

import pytest

from vllm_fl.utils import get_flaggems_all_ops


def test_get_flaggems_all_ops_contains_silu():
    ops = get_flaggems_all_ops()
    if not ops:
        pytest.skip("flag_gems._FULL_CONFIG not available (requires FlagGems >= 5.0)")
    assert "silu" in ops

# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unit tests for FlagGems ops discovery functionality.
"""

from vllm_fl.utils import get_flaggems_all_ops


def test_get_flaggems_all_ops_contains_silu():
    ops = get_flaggems_all_ops()
    assert "silu" in ops

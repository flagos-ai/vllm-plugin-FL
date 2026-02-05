# Copyright (c) 2026 BAAI. All rights reserved.


from vllm_fl.utils import get_flaggems_all_ops


def test_get_flaggems_all_ops_contains_silu():
    ops = get_flaggems_all_ops()
    assert "silu" in ops

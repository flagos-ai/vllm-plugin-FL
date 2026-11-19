"""BV=16 only while its fwd_h grid fits in one wave (measured on H100, 132 SMs)."""

import pytest

from vllm_fl.cp.fla_tune import pick_small_bv


@pytest.mark.parametrize(
    "num_seqs,expected", [(1, True), (2, True), (3, False), (4, False), (8, False)]
)
def test_small_bv_matches_h100_measurements(num_seqs, expected):
    # 8 v-heads, V=128 per TP4 rank: BV16 grid = 8 * num_seqs * 8 blocks.
    assert pick_small_bv(num_seqs, heads=8, v_dim=128, num_sms=132) is expected

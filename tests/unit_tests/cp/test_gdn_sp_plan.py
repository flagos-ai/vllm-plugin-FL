"""Sequence-parallel GDN plan: local segments match the zigzag local rows."""

from itertools import accumulate

import numpy as np
import pytest

from vllm_fl.cp.gdn_sp import SPPlan
from vllm_fl.cp.layout import ZigZagPlan


@pytest.mark.parametrize(
    "lengths", [[65535], [32767], [16383] * 4, [16383, 16383, 16383, 16380, 64]]
)
def test_segments_match_zigzag_rows_and_gather_order(lengths, world=4):
    starts = [0, *accumulate(lengths)]
    zig = ZigZagPlan.build_all(starts, lengths, world)
    plans = [SPPlan(lengths, world, r) for r in range(world)]
    nb = 2 * world
    for r, (sp, z) in enumerate(zip(plans, zig)):
        assert sp.rows == len(z.rows)
        rows = np.asarray(z.rows)
        for req, block, offset, length in sp.local:
            q = lengths[req]
            a = starts[req] + q * block // nb
            np.testing.assert_array_equal(
                rows[offset : offset + length], np.arange(a, a + length)
            )
            assert block in (r, nb - 1 - r) and length >= 4
    # Gathered index of (req, block) = owner rank's position in its local list.
    for (req, block), index in plans[0].gathered.items():
        owner = block if block < world else nb - 1 - block
        local = [(q, b) for q, b, _, _ in plans[owner].local]
        assert index == owner * len(local) + local.index((req, block))
    assert sorted(plans[0].gathered) == [
        (q, b) for q in range(len(lengths)) for b in range(nb)
    ]

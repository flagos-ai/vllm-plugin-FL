# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_fl.models.hy_v4_flash.nvidia.indexpool_workspace import (
    profile_peak_bytes,
    profile_workspace_bytes,
    scoring_chunk_rows,
)


def test_scoring_chunk_respects_logits_budget_and_selector_limit():
    assert scoring_chunk_rows(512, 512) == 1
    assert scoring_chunk_rows(1023, 512) == 1
    assert scoring_chunk_rows(1024, 512) == 2
    assert scoring_chunk_rows(2**24, 64) == 8192
    with pytest.raises(ValueError, match="single scoring row"):
        scoring_chunk_rows(511, 512)


def test_profile_bounds_narrow_eager_and_full_graph_widths():
    kwargs = dict(
        tokens=8192,
        batch=64,
        ratio=4,
        tail_ring=4,
        heads=16,
        budget=511,
        logits_elems=2**27,
        num_sms=132,
        page_size=64,
    )
    largest = 51200
    peak = profile_peak_bytes(max_width=largest, **kwargs)
    assert peak >= profile_workspace_bytes(width=largest, **kwargs)
    assert peak >= profile_workspace_bytes(width=16384, **kwargs)
    assert peak >= profile_workspace_bytes(width=64, **kwargs)
    # The logits budget alone does not cover live Q, metadata and selection.
    assert peak > kwargs["logits_elems"] * 4

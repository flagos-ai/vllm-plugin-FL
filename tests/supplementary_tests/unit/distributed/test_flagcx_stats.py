# Copyright 2025 BAAI. All rights reserved.
"""PR-#809-only increments for vllm_fl.distributed.kv_transfer.flagcx_stats.

Coverage boundary (post-refactor): the common behaviour of
``FlagCXKVConnectorStats`` (post-init reset, record paths, clone_and_reset
snapshots, is_empty transitions, empty-other aggregate, the reduce summary)
and all of ``FlagCXPromMetrics`` is owned by PR #799's
``tests/supplementary_tests/unit/distributed/test_flagcx_stats.py``
(which additionally pins concurrent column alignment and P90 millisecond
rounding). This file keeps only the #809-specific increments that #799
does not carry:

* ``__getstate__`` lock-stripping isolation (without the pickle roundtrip
  #799 couples it with);
* ``deepcopy`` snapshot independence (the per-attribute copy semantics);
* ``aggregate``'s column extension across every data key (including the
  num_successful_transfers counter);
* ``to_dict`` returning the underlying data mapping itself.

CPU-only, pure-logic tests.
"""

from __future__ import annotations

import copy

import pytest

pytestmark = [pytest.mark.supplementary]

from vllm_fl.distributed.kv_transfer.flagcx_stats import (  # noqa: E402
    FlagCXKVConnectorStats,
)

ALL_KEYS = (
    "transfer_duration",
    "bytes_transferred",
    "num_descriptors",
    "num_failed_transfers",
    "num_failed_recvs",
    "num_kv_expired_reqs",
)


class TestFlagCXKVConnectorStatsIncrements:
    """#809-only serialisation and aggregation details."""

    def test_getstate_strips_lock(self):
        stats = FlagCXKVConnectorStats()
        state = stats.__getstate__()
        assert "_lock" not in state
        assert set(state["data"]) == set(ALL_KEYS)

    def test_deepcopy_snapshot_is_independent(self):
        stats = FlagCXKVConnectorStats()
        stats.record_transfer(0.1, 8, 1)
        clone = copy.deepcopy(stats)
        clone.record_failed_recv()
        assert stats.data["num_failed_recvs"] == []
        assert clone.data["num_failed_recvs"] == [1]

    def test_aggregate_extends_columns(self):
        acc = FlagCXKVConnectorStats()
        acc.record_transfer(0.1, 8, 1)
        other = FlagCXKVConnectorStats()
        other.record_transfer(0.2, 16, 2)
        other.record_failed_recv()

        acc.aggregate(other)

        assert acc.data["transfer_duration"] == [0.1, 0.2]
        assert acc.data["bytes_transferred"] == [8, 16]
        assert acc.data["num_descriptors"] == [1, 2]
        assert acc.data["num_failed_recvs"] == [1]
        assert acc.num_successful_transfers == 2

    def test_to_dict_returns_data(self):
        stats = FlagCXKVConnectorStats()
        stats.record_failed_transfer()
        assert stats.to_dict() is stats.data

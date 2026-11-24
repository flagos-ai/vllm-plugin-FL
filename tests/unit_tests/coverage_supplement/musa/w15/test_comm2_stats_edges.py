# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Residual branch coverage for ``vllm_fl.distributed.kv_transfer.flagcx_stats``."""

import pytest

import vllm_fl.distributed.kv_transfer.flagcx_stats as stats_mod


Stats = stats_mod.FlagCXKVConnectorStats


def test_aggregate_asserts_self_columns_are_lists():
    target = Stats(data={"transfer_duration": (1.0,)})
    other = Stats()
    other.record_transfer(0.5, 1024, 1)

    with pytest.raises(AssertionError):
        target.aggregate(other)


def test_reduce_asserts_descriptor_column_alignment():
    stats = Stats(
        data={
            "transfer_duration": [0.1],
            "bytes_transferred": [1024],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_recvs": [],
            "num_kv_expired_reqs": [],
        }
    )
    with pytest.raises(AssertionError):
        stats.reduce()


def test_getstate_and_setstate_rebuild_lock_without_pickle():
    stats = Stats()
    stats.record_transfer(0.5, 2**20, 2)
    state = stats.__getstate__()
    assert "_lock" not in state
    assert state["data"]["transfer_duration"] == [0.5]

    restored = Stats.__new__(Stats)
    restored.__setstate__(state)
    restored.record_failed_recv()
    assert restored.data["num_failed_recvs"] == [1]


def test_reduce_counts_expired_only_without_success():
    stats = Stats()
    stats.record_kv_expired_req()
    result = stats.reduce()
    assert result["Num successful transfers"] == 0
    assert result["Num KV expired reqs"] == 1
    assert result["Num failed transfers"] == 0

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coverage tests for vllm_fl/distributed/kv_transfer/flagcx_stats.py.

FlagCXKVConnectorStats is a lock-guarded column store for transfer
metrics. Everything here is pure CPU state manipulation, so each test
derives the expected reduced/observed values in closed form and
compares.
"""


from __future__ import annotations

import pickle

from . import _flagcx_env  # noqa: F401  (installs inert flag_gems if triton is absent)

_flagcx_env.ensure_importable_env()

import pytest


def _stats():
    from vllm_fl.distributed.kv_transfer.flagcx_stats import FlagCXKVConnectorStats

    return FlagCXKVConnectorStats()


def test_post_init_reserves_all_columns():
    """A fresh stats container resets to six aligned empty columns."""
    s = _stats()
    assert s.data == {
        "transfer_duration": [],
        "bytes_transferred": [],
        "num_descriptors": [],
        "num_failed_transfers": [],
        "num_failed_recvs": [],
        "num_kv_expired_reqs": [],
    }
    assert s.is_empty()


def test_post_init_keeps_passed_in_data():
    """__post_init__ only resets when constructed without data."""
    from vllm_fl.distributed.kv_transfer.flagcx_stats import FlagCXKVConnectorStats

    s = FlagCXKVConnectorStats(data={"transfer_duration": [0.5]})
    assert s.data == {"transfer_duration": [0.5]}
    assert s.num_successful_transfers == 1


def test_record_transfer_appends_aligned_row():
    """Each successful transfer appends one value to each success column."""
    s = _stats()
    s.record_transfer(duration_s=0.25, total_bytes=1024, num_descs=4)
    s.record_transfer(duration_s=0.5, total_bytes=2048, num_descs=8)
    assert s.data["transfer_duration"] == [0.25, 0.5]
    assert s.data["bytes_transferred"] == [1024, 2048]
    assert s.data["num_descriptors"] == [4, 8]
    assert s.num_successful_transfers == 2
    assert not s.is_empty()


def test_failure_counters_append_ones():
    """Failure counters append literal 1s so Prom can iterate .inc(item)."""
    s = _stats()
    s.record_failed_transfer()
    s.record_failed_transfer()
    s.record_failed_recv()
    s.record_kv_expired_req()
    assert s.data["num_failed_transfers"] == [1, 1]
    assert s.data["num_failed_recvs"] == [1]
    assert s.data["num_kv_expired_reqs"] == [1]
    assert not s.is_empty()


def test_is_empty_checks_every_failure_column():
    """Failures alone, without successes, still make the container non-empty."""
    s = _stats()
    s.record_failed_recvs = s.record_failed_recv
    s.record_failed_recv()
    assert s.num_successful_transfers == 0
    assert not s.is_empty()


def test_clone_and_reset_snapshots_and_clears():
    """clone_and_reset returns the recorded data and zeroes the source."""
    s = _stats()
    s.record_transfer(0.5, 2**20, 3)
    s.record_failed_transfer()
    snap = s.clone_and_reset()
    assert snap.data["transfer_duration"] == [0.5]
    assert snap.data["bytes_transferred"] == [2**20]
    assert snap.data["num_descriptors"] == [3]
    assert snap.data["num_failed_transfers"] == [1]
    assert s.data["transfer_duration"] == []
    assert s.data["num_failed_transfers"] == []
    assert s.is_empty()
    assert not snap.is_empty()
    # The snapshot owns its lists: mutating it cannot resurrect the source.
    snap.data["transfer_duration"].append(9.0)
    assert s.data["transfer_duration"] == []


def test_reduce_zero_successes_returns_zeroed_metrics():
    """With no successful transfers reduce reports zeros for the rate
    series while still surfacing the failure counts."""
    s = _stats()
    s.record_failed_transfer()
    s.record_failed_transfer()
    s.record_failed_recv()
    s.record_kv_expired_req()
    assert s.reduce() == {
        "Num successful transfers": 0,
        "Avg xfer time (ms)": 0,
        "P90 xfer time (ms)": 0,
        "Avg MB per transfer": 0,
        "Throughput (MB/s)": 0,
        "Avg number of descriptors": 0,
        "Num failed transfers": 2,
        "Num failed recvs": 1,
        "Num KV expired reqs": 1,
    }


def test_reduce_success_path_closed_form():
    """reduce() reports n, mean/P90 latency, MB stats and throughput.

    Transfers: 1.0 s / 2 MiB / 10 descs and 3.0 s / 6 MiB / 30 descs.
    Derived: avg xfer 2000 ms, P90 of [1, 3] = 2.8 s -> 2800 ms,
    avg 4 MiB, total 8 MiB / 4 s = 2 MiB/s, avg 20 descs.
    """
    s = _stats()
    s.record_transfer(1.0, 2 * 2**20, 10)
    s.record_transfer(3.0, 6 * 2**20, 30)
    s.record_failed_recv()
    r = s.reduce()
    assert r["Num successful transfers"] == 2
    assert r["Avg xfer time (ms)"] == pytest.approx(2000.0)
    assert r["P90 xfer time (ms)"] == pytest.approx(2800.0)
    assert r["Avg MB per transfer"] == pytest.approx(4.0)
    assert r["Throughput (MB/s)"] == pytest.approx(2.0)
    assert r["Avg number of descriptors"] == pytest.approx(20.0)
    assert r["Num failed transfers"] == 0
    assert r["Num failed recvs"] == 1
    assert r["Num KV expired reqs"] == 0


def test_reduce_zero_total_time_reports_zero_throughput():
    """A zero-duration transfer window yields 0 throughput, not a crash."""
    s = _stats()
    s.record_transfer(0.0, 2**20, 1)
    r = s.reduce()
    assert r["Throughput (MB/s)"] == 0
    assert r["Num successful transfers"] == 1


def test_aggregate_merges_other_into_self():
    """aggregate() extends this container's columns with the other's."""
    a = _stats()
    a.record_transfer(0.5, 2**20, 2)
    b = _stats()
    b.record_transfer(1.0, 2 * 2**20, 4)
    b.record_failed_transfer()
    ret = a.aggregate(b)
    assert ret is a
    assert a.data["transfer_duration"] == [0.5, 1.0]
    assert a.data["bytes_transferred"] == [2**20, 2 * 2**20]
    assert a.data["num_descriptors"] == [2, 4]
    assert a.data["num_failed_transfers"] == [1]


def test_aggregate_skips_empty_other():
    """Aggregating an empty container leaves the columns untouched."""
    a = _stats()
    a.record_transfer(0.5, 2**20, 2)
    a.aggregate(_stats())
    assert a.data["transfer_duration"] == [0.5]


def test_pickle_roundtrip_rebuilds_lock():
    """Pickling strips the lock and unpickling rebuilds a working one."""
    s = _stats()
    s.record_transfer(0.5, 2**20, 3)
    s.record_failed_recv()
    state = pickle.dumps(s)
    loaded = pickle.loads(state)
    assert loaded.data["transfer_duration"] == [0.5]
    assert loaded.data["num_failed_recvs"] == [1]
    # The rebuilt lock guards writes: post-load recording still works.
    loaded.record_transfer(0.25, 512, 1)
    assert loaded.data["transfer_duration"] == [0.5, 0.25]
    # The lock itself never crosses the wire.
    assert b"_lock" not in pickle.dumps(loaded)


def test_build_kv_connector_stats_wraps_data(monkeypatch):
    """The connector classmethod wraps a plain dict, defaulting to empty."""
    _flagcx_env.install_fake_flagcx_wheel(monkeypatch)
    _flagcx_env.ensure_vllm_api_compat()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as conn

    empty = conn.FlagCXConnector.build_kv_connector_stats()
    assert empty is not None
    assert empty.data["transfer_duration"] == []

    wrapped = conn.FlagCXConnector.build_kv_connector_stats(
        {"transfer_duration": [0.5]}
    )
    assert wrapped.data["transfer_duration"] == [0.5]

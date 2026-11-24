# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for vllm_fl.distributed.kv_transfer.flagcx_stats.

CPU-only, pure-logic tests:

* ``FlagCXKVConnectorStats``: post-init reset semantics, pickle/deepcopy wire
  form (lock stripped and rebuilt), record paths, snapshot isolation of
  ``clone_and_reset``, ``is_empty`` transitions, ``aggregate`` and the exact
  ``reduce`` output for empty / populated / zero-duration containers.
* ``FlagCXPromMetrics``: counter and histogram registration (names, buckets,
  label plumbing through ``create_metric_per_engine``) and ``observe``'s
  per-key advance, the ``num_transfers`` special case, and the missing-engine
  guard.
"""

from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.supplementary]

from vllm_fl.distributed.kv_transfer.flagcx_stats import (  # noqa: E402
    FlagCXKVConnectorStats,
    FlagCXPromMetrics,
)

ALL_KEYS = (
    "transfer_duration",
    "bytes_transferred",
    "num_descriptors",
    "num_failed_transfers",
    "num_failed_recvs",
    "num_kv_expired_reqs",
)


class TestFlagCXKVConnectorStatsContainer:
    def test_post_init_resets_when_data_missing(self):
        stats = FlagCXKVConnectorStats()
        assert set(stats.data) == set(ALL_KEYS)
        assert all(stats.data[k] == [] for k in ALL_KEYS)
        assert stats.is_empty()

    def test_post_init_keeps_prefilled_data(self):
        stats = FlagCXKVConnectorStats(data={"transfer_duration": [0.5]})
        assert stats.data == {"transfer_duration": [0.5]}
        assert stats.num_successful_transfers == 1
        assert not stats.is_empty()

    def test_getstate_strips_lock(self):
        stats = FlagCXKVConnectorStats()
        state = stats.__getstate__()
        assert "_lock" not in state
        assert set(state["data"]) == set(ALL_KEYS)

    def test_pickle_roundtrip_rebuilds_lock_and_keeps_data(self):
        stats = FlagCXKVConnectorStats()
        stats.record_transfer(0.25, 1024, 2)
        stats.record_failed_transfer()

        restored = pickle.loads(pickle.dumps(stats))

        assert restored.data == stats.data
        assert restored.data is not stats.data
        # The rebuilt lock is a fresh, working per-process lock.
        assert isinstance(restored._lock, type(stats._lock))
        restored.record_failed_recv()
        assert restored.data["num_failed_recvs"] == [1]
        # __setstate__ re-ran reset() only when data was empty.
        assert stats.data["num_failed_recvs"] == []

    def test_deepcopy_snapshot_is_independent(self):
        stats = FlagCXKVConnectorStats()
        stats.record_transfer(0.1, 8, 1)
        clone = copy.deepcopy(stats)
        clone.record_failed_recv()
        assert stats.data["num_failed_recvs"] == []
        assert clone.data["num_failed_recvs"] == [1]

    def test_record_transfer_appends_three_columns(self):
        stats = FlagCXKVConnectorStats()
        stats.record_transfer(duration_s=0.5, total_bytes=2048, num_descs=4)
        stats.record_transfer(duration_s=1.5, total_bytes=4096, num_descs=6)
        assert stats.data["transfer_duration"] == [0.5, 1.5]
        assert stats.data["bytes_transferred"] == [2048, 4096]
        assert stats.data["num_descriptors"] == [4, 6]
        assert stats.num_successful_transfers == 2

    def test_failure_counters_append_unit_markers(self):
        stats = FlagCXKVConnectorStats()
        stats.record_failed_transfer()
        stats.record_failed_transfer()
        stats.record_failed_recv()
        stats.record_kv_expired_req()
        assert stats.data["num_failed_transfers"] == [1, 1]
        assert stats.data["num_failed_recvs"] == [1]
        assert stats.data["num_kv_expired_reqs"] == [1]
        assert not stats.is_empty()
        assert stats.num_successful_transfers == 0

    def test_clone_and_reset_returns_independent_snapshot(self):
        stats = FlagCXKVConnectorStats()
        stats.record_transfer(0.1, 8, 1)
        stats.record_failed_recv()

        snapshot = stats.clone_and_reset()

        assert snapshot is not stats
        assert snapshot.data["transfer_duration"] == [0.1]
        assert snapshot.data["num_failed_recvs"] == [1]
        # The source was reset under the lock.
        assert stats.is_empty()
        # Snapshot lists are copies: later writes do not leak in.
        stats.record_transfer(9.9, 1, 1)
        assert snapshot.data["transfer_duration"] == [0.1]

    def test_is_empty_transitions(self):
        stats = FlagCXKVConnectorStats()
        assert stats.is_empty()
        stats.record_kv_expired_req()
        assert not stats.is_empty()

    def test_aggregate_with_empty_other_is_noop(self):
        stats = FlagCXKVConnectorStats()
        stats.record_transfer(0.1, 8, 1)
        empty = FlagCXKVConnectorStats()

        result = stats.aggregate(empty)

        assert result is stats
        assert stats.data["transfer_duration"] == [0.1]

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


class TestFlagCXKVConnectorStatsReduce:
    def test_reduce_empty_container_returns_zeroed_series(self):
        stats = FlagCXKVConnectorStats()
        stats.record_failed_transfer()
        stats.record_failed_transfer()
        stats.record_failed_recv()
        stats.record_kv_expired_req()

        reduced = stats.reduce()

        assert reduced == {
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

    def test_reduce_populated_container_exact_values(self):
        stats = FlagCXKVConnectorStats()
        stats.record_transfer(duration_s=0.1, total_bytes=2**21, num_descs=3)
        stats.record_transfer(duration_s=0.2, total_bytes=2**22, num_descs=5)
        stats.record_failed_recv()

        reduced = stats.reduce()

        # 2 transfers, 0.1s/0.2s; 2 MiB and 4 MiB; total time 0.3 s.
        assert reduced["Num successful transfers"] == 2
        assert reduced["Avg xfer time (ms)"] == 150.0
        assert reduced["P90 xfer time (ms)"] == 190.0
        assert reduced["Avg MB per transfer"] == 3.0
        assert reduced["Throughput (MB/s)"] == 20.0
        assert reduced["Avg number of descriptors"] == 4.0
        assert reduced["Num failed transfers"] == 0
        assert reduced["Num failed recvs"] == 1
        assert reduced["Num KV expired reqs"] == 0

    def test_reduce_zero_total_time_reports_zero_throughput(self):
        stats = FlagCXKVConnectorStats()
        stats.record_transfer(duration_s=0.0, total_bytes=2**21, num_descs=1)

        reduced = stats.reduce()

        assert reduced["Throughput (MB/s)"] == 0.0
        assert reduced["Num successful transfers"] == 1
        assert reduced["Avg xfer time (ms)"] == 0.0


class _FakeChild:
    """Recording stand-in for a labeled prometheus_client child."""

    def __init__(self):
        self.incs = []
        self.observes = []

    def inc(self, value=1):
        self.incs.append(value)

    def observe(self, value):
        self.observes.append(value)


class _FakeMetric:
    """Stand-in for a prometheus_client metric class instance.

    FlagCXPromMetrics stores only the ``create_metric_per_engine`` result
    (a dict of labeled children), so the factory keeps a per-instance
    registry of created metrics (for name/bucket assertions) and the
    labels() calls (for per-engine plumbing assertions).
    """

    created: list["_FakeMetric"] = []

    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.documentation = kwargs.get("documentation")
        self.labelnames = kwargs.get("labelnames")
        self.buckets = kwargs.get("buckets")
        self.labelvalues = []
        _FakeMetric.created.append(self)

    def labels(self, *values):
        self.labelvalues.append(values)
        return _FakeChild()


class _FakeGauge(_FakeMetric):
    pass


class _FakeCounter(_FakeMetric):
    pass


class _FakeHistogram(_FakeMetric):
    pass


def _make_prom_metrics(per_engine=None):
    """Build FlagCXPromMetrics with recording fake metric classes.

    Returns ``(pm, created, children)``:

    * ``pm``: the FlagCXPromMetrics instance.
    * ``created``: every _FakeMetric constructed during registration (10:
      7 counters + 3 histograms), carrying name/buckets/labelvalues.
    * ``children``: ``{"counters": {data_key: {engine_idx: child}},
      "histograms": ...}`` — the labeled children the module retained.
    """
    from prometheus_client import Counter, Gauge, Histogram

    _FakeMetric.created = []
    vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace())
    metric_types = {
        Gauge: _FakeGauge,
        Counter: _FakeCounter,
        Histogram: _FakeHistogram,
    }
    per_engine = per_engine if per_engine is not None else {0: ["m1"], 1: ["m2"]}
    pm = FlagCXPromMetrics(
        vllm_config, metric_types, ["model_name"], per_engine
    )
    counters = {key: child_dict for child_dict, key in pm._counters}
    histograms = {key: child_dict for child_dict, key in pm._histograms}
    return pm, list(_FakeMetric.created), {
        "counters": counters,
        "histograms": histograms,
    }


EXPECTED_COUNTERS = {
    "vllm:flagcx_total_bytes_transferred": "bytes_transferred",
    "vllm:flagcx_total_xfer_time_seconds": "transfer_duration",
    "vllm:flagcx_transfers": "num_transfers",
    "vllm:flagcx_total_descriptors": "num_descriptors",
    "vllm:flagcx_num_failed_transfers": "num_failed_transfers",
    "vllm:flagcx_num_failed_recvs": "num_failed_recvs",
    "vllm:flagcx_num_kv_expired_reqs": "num_kv_expired_reqs",
}

EXPECTED_HISTOGRAMS = {
    "vllm:flagcx_xfer_time_seconds": "transfer_duration",
    "vllm:flagcx_bytes_transferred": "bytes_transferred",
    "vllm:flagcx_num_descriptors": "num_descriptors",
}


class TestFlagCXPromMetrics:
    def test_registers_seven_counters_then_three_histograms(self):
        _, created, _ = _make_prom_metrics()

        # All 10 metrics in spec order: 7 counters then 3 histograms.
        assert [m.name for m in created] == [
            "vllm:flagcx_total_bytes_transferred",
            "vllm:flagcx_total_xfer_time_seconds",
            "vllm:flagcx_transfers",
            "vllm:flagcx_total_descriptors",
            "vllm:flagcx_num_failed_transfers",
            "vllm:flagcx_num_failed_recvs",
            "vllm:flagcx_num_kv_expired_reqs",
            "vllm:flagcx_xfer_time_seconds",
            "vllm:flagcx_bytes_transferred",
            "vllm:flagcx_num_descriptors",
        ]

    def test_histogram_buckets_and_label_plumbing(self):
        _, created, _ = _make_prom_metrics()

        by_name = {m.name: m for m in created}
        assert by_name["vllm:flagcx_xfer_time_seconds"].buckets == [
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
        assert by_name["vllm:flagcx_bytes_transferred"].buckets == [
            2 ** (10 + i) for i in range(1, 25, 2)
        ]
        assert by_name["vllm:flagcx_num_descriptors"].buckets == [
            10,
            20,
            30,
            50,
            75,
            100,
            200,
            400,
            1000,
            2000,
            4000,
            10000,
            20000,
            50000,
        ]
        for metric in created:
            assert metric.labelnames == ["model_name"]
            # One labels() call per engine index, in per_engine order.
            assert metric.labelvalues == [("m1",), ("m2",)]

    def test_observe_advances_counters_and_histograms_per_value(self):
        pm, _, kids = _make_prom_metrics()
        data = {
            "transfer_duration": [0.1, 0.2],
            "bytes_transferred": [2**21, 2**22],
            "num_descriptors": [3, 5],
            "num_failed_transfers": [1, 1],
            "num_failed_recvs": [1],
            "num_kv_expired_reqs": [1],
        }

        pm.observe(data, engine_idx=0)

        counters = kids["counters"]
        assert counters["num_transfers"][0].incs == [2]
        assert counters["bytes_transferred"][0].incs == [2**21, 2**22]
        assert counters["transfer_duration"][0].incs == [0.1, 0.2]
        assert counters["num_descriptors"][0].incs == [3, 5]
        assert counters["num_failed_transfers"][0].incs == [1, 1]
        assert counters["num_failed_recvs"][0].incs == [1]
        assert counters["num_kv_expired_reqs"][0].incs == [1]
        # The other engine's children are untouched.
        assert counters["num_transfers"][1].incs == []

        histograms = kids["histograms"]
        assert histograms["transfer_duration"][0].observes == [0.1, 0.2]
        assert histograms["bytes_transferred"][0].observes == [2**21, 2**22]
        assert histograms["num_descriptors"][0].observes == [3, 5]
        assert histograms["transfer_duration"][1].observes == []

    def test_observe_with_missing_series_advances_only_num_transfers(self):
        pm, _, kids = _make_prom_metrics()

        pm.observe({}, engine_idx=0)

        counters = kids["counters"]
        assert counters["num_transfers"][0].incs == [0]
        for key, child in counters.items():
            if key != "num_transfers":
                assert child[0].incs == []
        for child in kids["histograms"].values():
            assert child[0].observes == []

    def test_observe_for_unknown_engine_index_is_skipped(self):
        pm, _, kids = _make_prom_metrics()

        pm.observe(
            {"transfer_duration": [0.1], "bytes_transferred": [8]}, engine_idx=7
        )

        for child in kids["counters"].values():
            for engine_child in child.values():
                assert engine_child.incs == []
        for child in kids["histograms"].values():
            for engine_child in child.values():
                assert engine_child.observes == []

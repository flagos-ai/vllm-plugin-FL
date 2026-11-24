# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Stats container and Prometheus metric coverage for FlagCX.

``FlagCXKVConnectorStats`` is a lock-guarded column store; every expected
reduction is derived in closed form.  ``FlagCXPromMetrics`` registers and
feeds counters/histograms; the metric classes and the vLLM base
initializer are replaced with recorders so the registration and observe
logic in the module under test runs without a global Prom registry.
"""

from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import test_flagcx_connector_support as support


@pytest.fixture()
def stats_mod():
    import vllm_fl.distributed.kv_transfer.flagcx_stats as stats

    return stats


def _stats(stats_mod):
    return stats_mod.FlagCXKVConnectorStats()


def test_post_init_resets_all_columns(stats_mod):
    stats = _stats(stats_mod)
    assert stats.data == {
        "transfer_duration": [],
        "bytes_transferred": [],
        "num_descriptors": [],
        "num_failed_transfers": [],
        "num_failed_recvs": [],
        "num_kv_expired_reqs": [],
    }
    assert stats.is_empty()


def test_post_init_keeps_passed_data(stats_mod):
    stats = stats_mod.FlagCXKVConnectorStats(data={"transfer_duration": [0.5]})
    assert stats.data == {"transfer_duration": [0.5]}
    assert stats.num_successful_transfers == 1


def test_record_transfer_aligns_columns(stats_mod):
    stats = _stats(stats_mod)
    stats.record_transfer(duration_s=0.25, total_bytes=1024, num_descs=4)
    stats.record_transfer(duration_s=0.5, total_bytes=2048, num_descs=8)
    assert stats.data["transfer_duration"] == [0.25, 0.5]
    assert stats.data["bytes_transferred"] == [1024, 2048]
    assert stats.data["num_descriptors"] == [4, 8]
    assert stats.num_successful_transfers == 2
    assert not stats.is_empty()


def test_failure_counters_append_ones(stats_mod):
    stats = _stats(stats_mod)
    stats.record_failed_transfer()
    stats.record_failed_transfer()
    stats.record_failed_recv()
    stats.record_kv_expired_req()
    assert stats.data["num_failed_transfers"] == [1, 1]
    assert stats.data["num_failed_recvs"] == [1]
    assert stats.data["num_kv_expired_reqs"] == [1]
    assert not stats.is_empty()


def test_clone_and_reset(stats_mod):
    stats = _stats(stats_mod)
    stats.record_transfer(0.5, 2**20, 3)
    stats.record_failed_transfer()
    snapshot = stats.clone_and_reset()
    assert snapshot.data["transfer_duration"] == [0.5]
    assert snapshot.data["bytes_transferred"] == [2**20]
    assert snapshot.data["num_descriptors"] == [3]
    assert snapshot.data["num_failed_transfers"] == [1]
    assert stats.data["transfer_duration"] == []
    assert stats.is_empty()
    assert not snapshot.is_empty()
    snapshot.data["transfer_duration"].append(9.0)
    assert stats.data["transfer_duration"] == []


def test_reduce_zero_successes(stats_mod):
    stats = _stats(stats_mod)
    stats.record_failed_transfer()
    stats.record_failed_transfer()
    stats.record_failed_recv()
    stats.record_kv_expired_req()
    assert stats.reduce() == {
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


def test_reduce_success_closed_form(stats_mod):
    stats = _stats(stats_mod)
    stats.record_transfer(1.0, 2 * 2**20, 10)
    stats.record_transfer(3.0, 6 * 2**20, 30)
    stats.record_failed_recv()
    result = stats.reduce()
    assert result["Num successful transfers"] == 2
    assert result["Avg xfer time (ms)"] == pytest.approx(2000.0)
    assert result["P90 xfer time (ms)"] == pytest.approx(2800.0)
    assert result["Avg MB per transfer"] == pytest.approx(4.0)
    assert result["Throughput (MB/s)"] == pytest.approx(2.0)
    assert result["Avg number of descriptors"] == pytest.approx(20.0)
    assert result["Num failed transfers"] == 0
    assert result["Num failed recvs"] == 1
    assert result["Num KV expired reqs"] == 0


def test_reduce_zero_time_throughput(stats_mod):
    stats = _stats(stats_mod)
    stats.record_transfer(0.0, 2**20, 1)
    result = stats.reduce()
    assert result["Throughput (MB/s)"] == 0
    assert result["Num successful transfers"] == 1


def test_aggregate_merges(stats_mod):
    first = _stats(stats_mod)
    first.record_transfer(0.5, 2**20, 2)
    other = _stats(stats_mod)
    other.record_transfer(1.0, 2 * 2**20, 4)
    other.record_failed_transfer()
    assert first.aggregate(other) is first
    assert first.data["transfer_duration"] == [0.5, 1.0]
    assert first.data["bytes_transferred"] == [2**20, 2 * 2**20]
    assert first.data["num_descriptors"] == [2, 4]
    assert first.data["num_failed_transfers"] == [1]


def test_aggregate_skips_empty(stats_mod):
    first = _stats(stats_mod)
    first.record_transfer(0.5, 2**20, 2)
    first.aggregate(_stats(stats_mod))
    assert first.data["transfer_duration"] == [0.5]


def test_pickle_roundtrip_rebuilds_lock(stats_mod):
    stats = _stats(stats_mod)
    stats.record_transfer(0.5, 2**20, 3)
    stats.record_failed_recv()
    loaded = pickle.loads(pickle.dumps(stats))
    assert loaded.data["transfer_duration"] == [0.5]
    assert loaded.data["num_failed_recvs"] == [1]
    loaded.record_transfer(0.25, 512, 1)
    assert loaded.data["transfer_duration"] == [0.5, 0.25]
    assert b"_lock" not in pickle.dumps(loaded)


def test_build_kv_connector_stats(monkeypatch):
    conn, _ = support.load_connector(monkeypatch)
    try:
        empty = conn.FlagCXConnector.build_kv_connector_stats()
        assert empty.data["transfer_duration"] == []
        wrapped = conn.FlagCXConnector.build_kv_connector_stats(
            {"transfer_duration": [0.5]}
        )
        assert wrapped.data["transfer_duration"] == [0.5]
    finally:
        sys.modules.pop(
            "vllm_fl.distributed.kv_transfer.flagcx_connector", None
        )


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


class _Harness:
    def __init__(self):
        self.created = []
        self.children = []

    def counter_cls(self):
        harness = self

        class _Counter:
            def __init__(self, *, name, documentation, labelnames, **kwargs):
                self.spec = {
                    "kind": "counter",
                    "name": name,
                    "doc": documentation,
                    "labelnames": tuple(labelnames),
                }
                harness.created.append(self.spec)

            def labels(self, *labelvalues):
                return harness._child(self.spec, labelvalues)

        return _Counter

    def histogram_cls(self):
        harness = self

        class _Histogram:
            def __init__(
                self, *, name, documentation, labelnames, buckets=None, **kwargs
            ):
                self.spec = {
                    "kind": "histogram",
                    "name": name,
                    "doc": documentation,
                    "labelnames": tuple(labelnames),
                    "buckets": tuple(buckets or ()),
                }
                harness.created.append(self.spec)

            def labels(self, *labelvalues):
                return harness._child(self.spec, labelvalues)

        return _Histogram

    def _child(self, spec, labelvalues):
        child = _Child(self, spec, labelvalues)
        self.children.append(child)
        return child


class _Child:
    def __init__(self, harness, spec, labelvalues):
        self.spec = spec
        self.labelvalues = tuple(labelvalues)
        self.incs = []
        self.observations = []

    def inc(self, value=1):
        self.incs.append(value)

    def observe(self, value):
        self.observations.append(value)


def _make_prom(monkeypatch, stats_mod):
    harness = _Harness()

    def fake_base_init(self, vllm_config, metric_types, labelnames, per_engine):
        self.per_engine_labelvalues = per_engine
        self._counter_cls = metric_types["counter"]
        self._histogram_cls = metric_types["histogram"]
        self.labelnames = labelnames

    def fake_create(metric, per_engine):
        return {
            index: metric.labels(*values) for index, values in per_engine.items()
        }

    monkeypatch.setattr(stats_mod.KVConnectorPromMetrics, "__init__", fake_base_init)
    monkeypatch.setattr(stats_mod, "create_metric_per_engine", fake_create)
    metrics = stats_mod.FlagCXPromMetrics(
        None,
        {"counter": harness.counter_cls(), "histogram": harness.histogram_cls()},
        ["model_name"],
        {0: ["mymodel"], 1: ["other"]},
    )
    return metrics, harness


def _child(harness, name, labelvalue):
    matches = [
        child
        for child in harness.children
        if child.spec["name"] == name and child.labelvalues == (labelvalue,)
    ]
    assert len(matches) == 1, (name, labelvalue, len(matches))
    return matches[0]


def test_prom_registers_specs(stats_mod, monkeypatch):
    _, harness = _make_prom(monkeypatch, stats_mod)
    counters = [s for s in harness.created if s["kind"] == "counter"]
    histograms = [s for s in harness.created if s["kind"] == "histogram"]
    assert {c["name"] for c in counters} == {
        "vllm:flagcx_total_bytes_transferred",
        "vllm:flagcx_total_xfer_time_seconds",
        "vllm:flagcx_transfers",
        "vllm:flagcx_total_descriptors",
        "vllm:flagcx_num_failed_transfers",
        "vllm:flagcx_num_failed_recvs",
        "vllm:flagcx_num_kv_expired_reqs",
    }
    assert all(c["labelnames"] == ("model_name",) for c in counters)
    assert {h["name"] for h in histograms} == {
        "vllm:flagcx_xfer_time_seconds",
        "vllm:flagcx_bytes_transferred",
        "vllm:flagcx_num_descriptors",
    }
    bytes_hist = next(
        h for h in histograms if h["name"] == "vllm:flagcx_bytes_transferred"
    )
    assert bytes_hist["buckets"] == tuple(2 ** (10 + i) for i in range(1, 25, 2))
    desc_hist = next(
        h for h in histograms if h["name"] == "vllm:flagcx_num_descriptors"
    )
    assert desc_hist["buckets"][0] == 10
    assert desc_hist["buckets"][-1] == 50000


def test_prom_observe(stats_mod, monkeypatch):
    metrics, harness = _make_prom(monkeypatch, stats_mod)
    data = {
        "transfer_duration": [0.5, 1.5],
        "bytes_transferred": [2**20, 3 * 2**20],
        "num_descriptors": [2, 6],
        "num_failed_transfers": [1],
        "num_failed_recvs": [],
        "num_kv_expired_reqs": [1, 1],
    }
    metrics.observe(data, engine_idx=0)

    def child(name):
        return _child(harness, name, "mymodel")

    assert child("vllm:flagcx_transfers").incs == [2]
    assert child("vllm:flagcx_total_bytes_transferred").incs == [2**20, 3 * 2**20]
    assert child("vllm:flagcx_total_xfer_time_seconds").incs == [0.5, 1.5]
    assert child("vllm:flagcx_total_descriptors").incs == [2, 6]
    assert child("vllm:flagcx_num_failed_transfers").incs == [1]
    assert child("vllm:flagcx_num_failed_recvs").incs == []
    assert child("vllm:flagcx_num_kv_expired_reqs").incs == [1, 1]
    assert child("vllm:flagcx_xfer_time_seconds").observations == [0.5, 1.5]
    assert child("vllm:flagcx_bytes_transferred").observations == [
        2**20,
        3 * 2**20,
    ]
    assert child("vllm:flagcx_num_descriptors").observations == [2, 6]
    assert _child(harness, "vllm:flagcx_transfers", "other").incs == []


def test_prom_observe_unknown_engine_skipped(stats_mod, monkeypatch):
    metrics, harness = _make_prom(monkeypatch, stats_mod)
    metrics.observe(
        {
            "transfer_duration": [0.5],
            "bytes_transferred": [2**20],
            "num_descriptors": [1],
        },
        engine_idx=7,
    )
    assert all(not child.incs for child in harness.children)
    assert all(not child.observations for child in harness.children)


def test_prom_observe_empty(stats_mod, monkeypatch):
    metrics, harness = _make_prom(monkeypatch, stats_mod)
    metrics.observe({}, engine_idx=0)
    assert _child(harness, "vllm:flagcx_transfers", "mymodel").incs == [0]
    assert all(not child.observations for child in harness.children)


def test_build_prom_metrics_classmethod(monkeypatch, stats_mod):
    conn, _ = support.load_connector(monkeypatch)
    harness = _Harness()

    def fake_base_init(self, vllm_config, metric_types, labelnames, per_engine):
        self.per_engine_labelvalues = per_engine
        self._counter_cls = metric_types["counter"]
        self._histogram_cls = metric_types["histogram"]

    def fake_create(metric, per_engine):
        return {index: metric.labels(*values) for index, values in per_engine.items()}

    monkeypatch.setattr(stats_mod.KVConnectorPromMetrics, "__init__", fake_base_init)
    monkeypatch.setattr(stats_mod, "create_metric_per_engine", fake_create)
    try:
        metrics = conn.FlagCXConnector.build_prom_metrics(
            None,
            {"counter": harness.counter_cls(), "histogram": harness.histogram_cls()},
            ["model_name"],
            {0: ["mymodel"]},
        )
        assert isinstance(metrics, stats_mod.FlagCXPromMetrics)
        assert any(
            spec["name"] == "vllm:flagcx_transfers" and spec["kind"] == "counter"
            for spec in harness.created
        )
    finally:
        sys.modules.pop(
            "vllm_fl.distributed.kv_transfer.flagcx_connector", None
        )

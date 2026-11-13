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
"""Prometheus-metrics coverage tests for flagcx_stats.py.

FlagCXPromMetrics registers seven counters and three histograms and
maps recorded stats data onto them. prometheus_client's global registry
rejects duplicate timeseries across tests, so the metric classes are
replaced with recording substitutes and every registered name, bucket
and label wiring plus every inc/observe call is asserted in closed
form.
"""

from __future__ import annotations

from . import _flagcx_env  # noqa: F401  (installs inert flag_gems if triton is absent)

_flagcx_env.ensure_importable_env()


class _MetricTypesHarness:
    """Recording metric classes bound to an isolated registry.

    prometheus_client's global registry rejects duplicate timeseries
    across tests, and the metrics under test are constructed by name
    inside FlagCXPromMetrics. Passing classes whose constructor records
    the registration and whose label children record inc/observe lets us
    assert the exact names, buckets and label wiring (R4) without any
    global state.
    """

    def __init__(self):
        self.created: list[dict] = []
        self.children: list = []

    def _labels(self, spec, labelvalues):
        child = _RecordingChild(self, spec, labelvalues)
        self.children.append(child)
        return child

    def counter_cls(self):
        harness = self

        class _Counter:
            def __init__(self, *, name, documentation, labelnames, **kw):
                harness.created.append(
                    {
                        "kind": "counter",
                        "name": name,
                        "doc": documentation,
                        "labelnames": tuple(labelnames),
                    }
                )
                self._spec = harness.created[-1]

            def labels(self, *labelvalues):
                return harness._labels(self._spec, labelvalues)

        return _Counter

    def histogram_cls(self):
        harness = self

        class _Histogram:
            def __init__(self, *, name, documentation, labelnames, buckets=None, **kw):
                harness.created.append(
                    {
                        "kind": "histogram",
                        "name": name,
                        "doc": documentation,
                        "labelnames": tuple(labelnames),
                        "buckets": tuple(buckets),
                    }
                )
                self._spec = harness.created[-1]

            def labels(self, *labelvalues):
                return harness._labels(self._spec, labelvalues)

        return _Histogram

    def gauge_cls(self):
        harness = self

        class _Gauge:
            def __init__(self, **kw):
                harness.created.append({"kind": "gauge"})

        return _Gauge


class _RecordingChild:
    """One labeled metric child; records inc/observe calls."""

    def __init__(self, harness, spec, labelvalues):
        self.harness = harness
        self.spec = spec
        self.labelvalues = tuple(labelvalues)
        self.incs: list = []
        self.observations: list = []

    def inc(self, value=1):
        self.incs.append(value)

    def observe(self, value):
        self.observations.append(value)


class _FakeTransferConfig:
    kv_role = "kv_both"
    kv_connector = "FlagCX"


class _FakeVllmConfig:
    kv_transfer_config = _FakeTransferConfig()


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


def _make_prom_metrics():
    from vllm_fl.distributed.kv_transfer.flagcx_stats import FlagCXPromMetrics

    harness = _MetricTypesHarness()
    from prometheus_client import Counter, Gauge, Histogram

    metric_types = {
        Gauge: harness.gauge_cls(),
        Counter: harness.counter_cls(),
        Histogram: harness.histogram_cls(),
    }
    metrics = FlagCXPromMetrics(
        _FakeVllmConfig(),
        metric_types,
        ["model_name"],
        {0: ["mymodel"], 1: ["other"]},
    )
    return metrics, harness


def _engine_child(harness, name, labelvalue):
    """Return the labeled child of metric ``name`` for one engine."""
    matches = [
        c
        for c in harness.children
        if c.spec["name"] == name and c.labelvalues == (labelvalue,)
    ]
    assert len(matches) == 1, (name, labelvalue, len(matches))
    return matches[0]


def test_prom_metrics_registers_counter_and_histogram_specs():
    """build_prom_metrics registers the seven counters and three
    histograms with the documented names and label wiring."""
    metrics, harness = _make_prom_metrics()
    counters = [c for c in harness.created if c["kind"] == "counter"]
    histograms = [h for h in harness.created if h["kind"] == "histogram"]
    assert {c["name"]: c["labelnames"] for c in counters} == {
        name: ("model_name",) for name in EXPECTED_COUNTERS
    }
    assert {h["name"]: h["labelnames"] for h in histograms} == {
        name: ("model_name",) for name in EXPECTED_HISTOGRAMS
    }
    # The bytes histogram uses the uniform 2KB..16GB NIXL buckets.
    bytes_hist = next(
        h for h in histograms if h["name"] == "vllm:flagcx_bytes_transferred"
    )
    assert bytes_hist["buckets"] == tuple(2 ** (10 + i) for i in range(1, 25, 2))
    # Descriptors histogram starts at 10 and ends at 50000.
    desc_hist = next(
        h for h in histograms if h["name"] == "vllm:flagcx_num_descriptors"
    )
    assert desc_hist["buckets"][0] == 10
    assert desc_hist["buckets"][-1] == 50000


def test_prom_metrics_observe_advances_counters_and_histograms():
    """observe() advances num_transfers by the duration column length,
    sums per-record values into the other counters, and feeds the
    histograms with the per-transfer observations."""
    metrics, harness = _make_prom_metrics()
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
        return _engine_child(harness, name, "mymodel")

    # num_transfers is advanced once, by the duration column length.
    assert child("vllm:flagcx_transfers").incs == [2]
    # Every other counter incs once per recorded value.
    assert child("vllm:flagcx_total_bytes_transferred").incs == [2**20, 3 * 2**20]
    assert child("vllm:flagcx_total_xfer_time_seconds").incs == [0.5, 1.5]
    assert child("vllm:flagcx_total_descriptors").incs == [2, 6]
    assert child("vllm:flagcx_num_failed_transfers").incs == [1]
    assert child("vllm:flagcx_num_failed_recvs").incs == []
    assert child("vllm:flagcx_num_kv_expired_reqs").incs == [1, 1]
    # Histograms observe each per-transfer value.
    assert child("vllm:flagcx_xfer_time_seconds").observations == [0.5, 1.5]
    assert child("vllm:flagcx_bytes_transferred").observations == [2**20, 3 * 2**20]
    assert child("vllm:flagcx_num_descriptors").observations == [2, 6]
    # The other engine's children stay untouched.
    assert _engine_child(harness, "vllm:flagcx_transfers", "other").incs == []


def test_prom_metrics_observe_unknown_engine_is_skipped():
    """An engine_idx with no labeled child leaves every metric untouched."""
    metrics, harness = _make_prom_metrics()
    data = {
        "transfer_duration": [0.5],
        "bytes_transferred": [2**20],
        "num_descriptors": [1],
    }
    metrics.observe(data, engine_idx=7)
    assert all(not child.incs for child in harness.children)
    assert all(not child.observations for child in harness.children)


def test_prom_metrics_observe_empty_data():
    """Observing empty columns records a zero transfer count only."""
    metrics, harness = _make_prom_metrics()
    metrics.observe({}, engine_idx=0)
    assert _engine_child(harness, "vllm:flagcx_transfers", "mymodel").incs == [0]
    assert all(not c.observations for c in harness.children)

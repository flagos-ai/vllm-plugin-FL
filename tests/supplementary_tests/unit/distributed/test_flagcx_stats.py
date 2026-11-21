# Copyright (c) 2026 BAAI. All rights reserved.
"""Supplementary CPU tests for ``vllm_fl.distributed.kv_transfer.flagcx_stats``.

Covers the KV transfer stats bookkeeping container (record/clone/reset,
pickling, reduction/aggregation math) and the Prometheus metrics registration
and observation fan-out, with vLLM and the metrics factory stubbed at the
module boundary.
"""

import pickle
import threading

import pytest

from tests.supplementary_tests.unit.distributed._flagcx_stubs import (
    FlagcxStubEnv,
)

pytest.importorskip("numpy")
pytest.importorskip("msgspec")
pytest.importorskip("zmq")


@pytest.fixture
def stats_env(monkeypatch):
    """Install the vLLM boundary stubs and import the real stats module."""
    FlagcxStubEnv(monkeypatch).install()
    import vllm_fl.distributed.kv_transfer.flagcx_stats as flagcx_stats

    return flagcx_stats


def _recorded(flagcx_stats, *, transfers=(), failures=0, failed_recvs=0, expired=0):
    """Build a stats container with the given observations recorded."""
    stats = flagcx_stats.FlagCXKVConnectorStats()
    for duration, size, descs in transfers:
        stats.record_transfer(duration, size, descs)
    for _ in range(failures):
        stats.record_failed_transfer()
    for _ in range(failed_recvs):
        stats.record_failed_recv()
    for _ in range(expired):
        stats.record_kv_expired_req()
    return stats


# ---------------------------------------------------------------------------
# FlagCXKVConnectorStats container
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_post_init_resets_when_constructed_without_data(stats_env):
    """A container constructed with no data gets all six empty columns."""
    stats = stats_env.FlagCXKVConnectorStats()
    assert stats.data == {
        "transfer_duration": [],
        "bytes_transferred": [],
        "num_descriptors": [],
        "num_failed_transfers": [],
        "num_failed_recvs": [],
        "num_kv_expired_reqs": [],
    }
    assert stats.is_empty()


@pytest.mark.supplementary
def test_post_init_keeps_existing_data(stats_env):
    """Pre-seeded data (e.g. from the wire) is not cleared by __post_init__."""
    stats = stats_env.FlagCXKVConnectorStats(data={"num_failed_recvs": [1, 1]})
    assert stats.data["num_failed_recvs"] == [1, 1]
    assert "transfer_duration" not in stats.data  # untouched, no reset


@pytest.mark.supplementary
def test_partial_preseeded_data_keyerrors_is_empty(stats_env):
    """Latent edge: is_empty() assumes the full column set exists, so a
    container built from a partial data dict raises KeyError.  In the real
    flow the dict is always {} (full reset) or a full clone, so this only
    fires if build_kv_connector_stats ever receives a partial dict.
    Documented as a defect pin; see the report."""
    stats = stats_env.FlagCXKVConnectorStats(data={"num_failed_recvs": [1]})
    with pytest.raises(KeyError, match="transfer_duration"):
        stats.is_empty()


@pytest.mark.supplementary
def test_record_transfer_appends_aligned_columns(stats_env):
    """record_transfer appends one entry to each of its three columns."""
    stats = stats_env.FlagCXKVConnectorStats()
    stats.record_transfer(0.5, 1024, 2)
    stats.record_transfer(0.25, 2048, 4)
    assert stats.data["transfer_duration"] == [0.5, 0.25]
    assert stats.data["bytes_transferred"] == [1024, 2048]
    assert stats.data["num_descriptors"] == [2, 4]
    assert stats.num_successful_transfers == 2
    assert not stats.is_empty()


@pytest.mark.supplementary
def test_failure_counters_append_ones(stats_env):
    """Each failure counter appends a 1 so counters can .inc(list_item)."""
    stats = stats_env.FlagCXKVConnectorStats()
    stats.record_failed_transfer()
    stats.record_failed_transfer()
    stats.record_failed_recv()
    stats.record_kv_expired_req()
    assert stats.data["num_failed_transfers"] == [1, 1]
    assert stats.data["num_failed_recvs"] == [1]
    assert stats.data["num_kv_expired_reqs"] == [1]
    assert not stats.is_empty()


@pytest.mark.supplementary
def test_is_empty_true_only_when_all_counters_drained(stats_env):
    """is_empty requires zero transfers AND zero failure/expiry records."""
    stats = _recorded(stats_env, failed_recvs=1)
    assert not stats.is_empty()
    empty = stats_env.FlagCXKVConnectorStats()
    assert empty.is_empty()


@pytest.mark.supplementary
def test_clone_and_reset_snapshots_and_clears(stats_env):
    """clone_and_reset copies the columns under the lock and resets in place."""
    stats = _recorded(stats_env, transfers=[(0.1, 512, 1)], failures=1)
    snapshot = stats.clone_and_reset()
    assert snapshot.data["transfer_duration"] == [0.1]
    assert snapshot.data["num_failed_transfers"] == [1]
    assert stats.data["transfer_duration"] == []
    assert stats.data["num_failed_transfers"] == []
    # The snapshot owns a fresh lock, distinct from the original's.
    assert snapshot._lock is not stats._lock
    # Copy-on-clone: mutating the snapshot's list does not touch the source.
    snapshot.data["transfer_duration"].append(9.9)
    assert stats.data["transfer_duration"] == []


@pytest.mark.supplementary
def test_concurrent_records_stay_column_aligned(stats_env):
    """Threads recording transfers concurrently keep all columns aligned."""
    stats = stats_env.FlagCXKVConnectorStats()
    n_threads, per_thread = 4, 25

    def worker():
        for _ in range(per_thread):
            stats.record_transfer(0.001, 8, 1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = n_threads * per_thread
    assert stats.num_successful_transfers == total
    assert len(stats.data["bytes_transferred"]) == total
    assert len(stats.data["num_descriptors"]) == total


@pytest.mark.supplementary
def test_pickling_strips_lock_and_rebuilds_it(stats_env):
    """Locks are dropped on pickling; the receiver rebuilds a fresh lock."""
    stats = _recorded(stats_env, transfers=[(0.2, 64, 3)], expired=1)
    restored = pickle.loads(pickle.dumps(stats))
    assert restored.data["transfer_duration"] == [0.2]
    assert restored.data["num_kv_expired_reqs"] == [1]
    assert isinstance(restored._lock, type(threading.Lock()))
    state = stats.__getstate__()
    assert "_lock" not in state
    # The restored instance is fully usable.
    restored.record_failed_recv()
    assert restored.data["num_failed_recvs"] == [1]


@pytest.mark.supplementary
def test_aggregate_merges_other_into_self(stats_env):
    """aggregate extends self's columns with the other's and returns self."""
    target = _recorded(stats_env, transfers=[(0.1, 100, 1)])
    other = _recorded(stats_env, transfers=[(0.2, 200, 2)], failed_recvs=1)
    result = target.aggregate(other)
    assert result is target
    assert target.data["transfer_duration"] == [0.1, 0.2]
    assert target.data["bytes_transferred"] == [100, 200]
    assert target.data["num_failed_recvs"] == [1]


@pytest.mark.supplementary
def test_aggregate_skips_empty_other(stats_env):
    """An empty other leaves self untouched."""
    target = _recorded(stats_env, transfers=[(0.1, 100, 1)])
    empty = stats_env.FlagCXKVConnectorStats()
    before = {k: list(v) for k, v in target.data.items()}
    target.aggregate(empty)
    assert target.data == before


@pytest.mark.supplementary
def test_reduce_zero_transfers_returns_zeroed_metrics(stats_env):
    """With no successful transfers, reduce reports zeroed success metrics
    while preserving the raw failure counts."""
    stats = _recorded(stats_env, failures=2, failed_recvs=3, expired=4)
    reduced = stats.reduce()
    assert reduced == {
        "Num successful transfers": 0,
        "Avg xfer time (ms)": 0,
        "P90 xfer time (ms)": 0,
        "Avg MB per transfer": 0,
        "Throughput (MB/s)": 0,
        "Avg number of descriptors": 0,
        "Num failed transfers": 2,
        "Num failed recvs": 3,
        "Num KV expired reqs": 4,
    }


@pytest.mark.supplementary
def test_reduce_computes_transfer_statistics(stats_env):
    """reduce derives count/latency/size/throughput/descriptor aggregates."""
    # 1 MiB + 3 MiB in 0.5s + 1.5s = 4 MiB over 2s -> 2 MiB/s (source
    # converts bytes with 2**20 exactly, then rounds to 3 decimals).
    stats = _recorded(
        stats_env,
        transfers=[(0.5, 2**20, 10), (1.5, 3 * 2**20, 30)],
    )
    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == 2
    assert reduced["Avg xfer time (ms)"] == 1000.0
    assert reduced["Avg MB per transfer"] == 2.0
    assert reduced["Throughput (MB/s)"] == 2.0
    assert reduced["Avg number of descriptors"] == 20.0
    assert reduced["Num failed transfers"] == 0


@pytest.mark.supplementary
def test_reduce_zero_total_time_reports_zero_throughput(stats_env):
    """A zero-duration accumulation reports 0 throughput, not a ZeroDivision."""
    stats = _recorded(stats_env, transfers=[(0.0, 2**20, 1)])
    reduced = stats.reduce()
    assert reduced["Throughput (MB/s)"] == 0
    assert reduced["Num successful transfers"] == 1


@pytest.mark.supplementary
def test_reduce_p90_rounds_to_milliseconds(stats_env):
    """P90 latency is converted to milliseconds with 3 decimals."""
    stats = _recorded(
        stats_env,
        transfers=[(0.1 * i / 100, 2**10, 1) for i in range(1, 11)],
    )
    reduced = stats.reduce()
    assert reduced["P90 xfer time (ms)"] == pytest.approx(9.1, abs=0.01)


# ---------------------------------------------------------------------------
# FlagCXPromMetrics
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_prom_metrics_registers_all_counters_and_histograms(stats_env, monkeypatch):
    """The constructor registers 7 counters and 3 histograms with the
    documented names, each fanned out per engine via create_metric_per_engine."""
    created = {"counters": [], "histograms": []}

    def fake_create(metric, engines):
        if hasattr(metric, "buckets"):
            created["histograms"].append(metric)
        else:
            created["counters"].append(metric)
        return {engine: metric for engine in engines}

    # flagcx_stats binds create_metric_per_engine at import time; patch the
    # name on the already-imported module under test.
    monkeypatch.setattr(stats_env, "create_metric_per_engine", fake_create)

    per_engine = {0: ["model_name=e1"], 1: ["model_name=e2"]}
    prom = stats_env.FlagCXPromMetrics(
        object(),  # vllm_config (unused by registration)
        {"counter": object, "histogram": object},
        ["model_name"],
        per_engine,
    )

    counter_names = {c.name for c in created["counters"]}
    histogram_names = {h.name for h in created["histograms"]}
    assert counter_names == {
        "vllm:flagcx_total_bytes_transferred",
        "vllm:flagcx_total_xfer_time_seconds",
        "vllm:flagcx_transfers",
        "vllm:flagcx_total_descriptors",
        "vllm:flagcx_num_failed_transfers",
        "vllm:flagcx_num_failed_recvs",
        "vllm:flagcx_num_kv_expired_reqs",
    }
    assert histogram_names == {
        "vllm:flagcx_xfer_time_seconds",
        "vllm:flagcx_bytes_transferred",
        "vllm:flagcx_num_descriptors",
    }
    # Every metric is fanned out to both engines.
    assert all(
        set(entry.keys()) == {0, 1} for entry, _ in prom._counters + prom._histograms
    )
    # The data keys pair each counter/histogram with its stats column.
    assert [key for _, key in prom._counters] == [
        "bytes_transferred",
        "transfer_duration",
        "num_transfers",
        "num_descriptors",
        "num_failed_transfers",
        "num_failed_recvs",
        "num_kv_expired_reqs",
    ]
    assert [key for _, key in prom._histograms] == [
        "transfer_duration",
        "bytes_transferred",
        "num_descriptors",
    ]


@pytest.mark.supplementary
def test_prom_metrics_observes_counters_and_histograms(stats_env):
    """observe() advances the transfers counter by the transfer_duration
    column length, every other counter by each recorded value, and each
    histogram per value; a missing engine idx observes nothing."""
    metrics = {}
    per_engine = {0: ["model_name=e1"]}
    prom = stats_env.FlagCXPromMetrics(
        object(),
        {"counter": object, "histogram": object},
        ["model_name"],
        per_engine,
    )
    # Swap the per-engine maps for single shared recorders so assertions can
    # see what observe() dispatches.
    for engine_map, key in prom._counters + prom._histograms:
        metrics[key] = engine_map[0]

    data = {
        "transfer_duration": [0.1, 0.2],
        "bytes_transferred": [1024, 2048],
        "num_descriptors": [2, 3],
        "num_failed_transfers": [1],
        "num_failed_recvs": [1, 1],
        "num_kv_expired_reqs": [],
    }
    prom.observe(data, engine_idx=0)

    assert metrics["num_transfers"].values == [2]  # column length, not values
    assert metrics["bytes_transferred"].values == [1024, 2048]
    assert metrics["transfer_duration"].values == [0.1, 0.2]
    assert metrics["num_descriptors"].values == [2, 3]
    assert metrics["num_failed_transfers"].values == [1]
    assert metrics["num_failed_recvs"].values == [1, 1]
    assert metrics["num_kv_expired_reqs"].values == []

    # Unknown engine idx: every map lookup misses -> nothing is observed.
    before = {key: list(metric.values) for key, metric in metrics.items()}
    prom.observe({"transfer_duration": [0.5], "bytes_transferred": [8]}, engine_idx=99)
    assert {k: v for k, v in metrics.items() if v.values != before[k]} == {}


@pytest.mark.supplementary
def test_prom_metrics_observe_handles_missing_columns(stats_env):
    """observe() tolerates a data dict that only carries some columns."""
    prom = stats_env.FlagCXPromMetrics(
        object(),
        {"counter": object, "histogram": object},
        ["model_name"],
        {0: ["model_name=e1"]},
    )
    prom.observe({"num_failed_transfers": [1, 1]}, engine_idx=0)
    counters = {key: entry[0] for entry, key in prom._counters}
    assert counters["num_failed_transfers"].values == [1, 1]
    assert counters["num_transfers"].values == [0]

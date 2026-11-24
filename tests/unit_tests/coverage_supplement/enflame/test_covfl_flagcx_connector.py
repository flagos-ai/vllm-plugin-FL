"""Coverage-focused, hardware-free tests for the FlagCX KV-transfer connector.

Target files (commit 030b452f, read-only source):
  vllm_fl/distributed/kv_transfer/flagcx_connector.py
  vllm_fl/distributed/kv_transfer/flagcx_stats.py

These tests deliberately avoid every real dependency: no GCU, no NCCL/FlagCX
C library, no ZMQ sockets on the wire, no spawned processes and no asyncio
loops that never return. The FlagCX wrapper module is stubbed only when it is
not importable, and FLAGCXLibrary / transport primitives are mocked per test.
"""

import asyncio
import contextlib
import importlib
import pickle
import sys
import threading
import types
from types import SimpleNamespace
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Import bootstrap: make the connector importable without the FlagCX C stack.
#
# If the real wrapper is available we use it. Otherwise we temporarily install
# a stub for plugin.interservice.flagcx_wrapper, import the connector (which
# binds FLAGCXLibrary at module scope), and then restore sys.modules so this
# test module does not leak the stub into the rest of the pytest session.
# ---------------------------------------------------------------------------
_CONN_MODULE = "vllm_fl.distributed.kv_transfer.flagcx_connector"
_STATS_MODULE = "vllm_fl.distributed.kv_transfer.flagcx_stats"
_STUB_KEYS = (
    "plugin",
    "plugin.interservice",
    "plugin.interservice.flagcx_wrapper",
)
_MISSING = object()


def _install_wrapper_stub():
    saved = {name: sys.modules.get(name, _MISSING) for name in _STUB_KEYS}

    pkg = types.ModuleType("plugin")
    pkg.__path__ = []
    sub = types.ModuleType("plugin.interservice")
    sub.__path__ = []
    wrapper = types.ModuleType("plugin.interservice.flagcx_wrapper")

    class FLAGCXLibrary:  # noqa: N801 - mirror upstream name
        def __init__(self, *args, **kwargs):
            self.args = args

    wrapper.FLAGCXLibrary = FLAGCXLibrary
    pkg.interservice = sub
    sub.flagcx_wrapper = wrapper
    sys.modules["plugin"] = pkg
    sys.modules["plugin.interservice"] = sub
    sys.modules["plugin.interservice.flagcx_wrapper"] = wrapper
    return saved


def _restore_modules(saved):
    for name, value in saved.items():
        if value is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def _ensure_vllm_compat():
    """Patch vLLM symbols this build renamed, so the connector is importable."""
    try:
        import vllm.v1.attention.backends.utils as _u
    except Exception:
        return
    if not hasattr(_u, "get_kv_cache_layout"):
        _cb = getattr(_u, "record_kv_cache_layout", None)
        _u.get_kv_cache_layout = (
            _cb if callable(_cb) else (lambda *a, **k: "NHD")
        )


def _load_modules():
    _ensure_vllm_compat()
    try:
        return (
            importlib.import_module(_CONN_MODULE),
            importlib.import_module(_STATS_MODULE),
        )
    except Exception:
        pass

    saved = _install_wrapper_stub()
    try:
        return (
            importlib.import_module(_CONN_MODULE),
            importlib.import_module(_STATS_MODULE),
        )
    finally:
        _restore_modules(saved)


_IMPORT_ERROR = None
try:
    connector, stats = _load_modules()
except Exception as exc:  # pragma: no cover - environment guard
    connector = None
    stats = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    connector is None,
    reason=f"cannot import FlagCX connector in this environment: {_IMPORT_ERROR!r}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _region(layer="l0", idx=0, base=1000, block_len=10, kv_block_len=10, group=0):
    return connector.TransferRegion(
        layer_name=layer,
        layer_index=idx,
        base_addr=base,
        block_len=block_len,
        kv_block_len=kv_block_len,
        group_index=group,
    )


def _agent_meta(req_blocks, remote_tp_rank=0, remote_tp_size=1, host="h", port=9000):
    return connector.FlagCXAgentMetadata(
        remote_hostname=host,
        remote_port=port,
        remote_tp_size=remote_tp_size,
        remote_tp_rank=remote_tp_rank,
        req_blocks=req_blocks,
        kv_caches_base_addr=[2000],
        block_lens=[10],
        kv_block_lens=[10],
        registered_layer_names=["l0"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )


def _send_meta(p_req="p", tid="t", blocks=None, ready=None):
    return connector.SendBlockMeta(
        p_req_id=p_req,
        transfer_id=tid,
        local_block_ids=[] if blocks is None else blocks,
        ready=ready if ready is not None else threading.Event(),
    )


class _FakeMetric:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.incs = []
        self.observes = []

    def inc(self, value=1):
        self.incs.append(value)

    def observe(self, value):
        self.observes.append(value)


def _fake_prom_init(self, vllm_config, metric_types, labelnames, per_engine):
    self.vllm_config = vllm_config
    self.metric_types = metric_types
    self.labelnames = labelnames
    self.per_engine_labelvalues = per_engine
    self._counter_cls = metric_types["counter"]
    self._histogram_cls = metric_types["histogram"]


def _make_prom(engine=0):
    def factory(metric, per_engine):
        return {engine: metric}

    with (
        mock.patch.object(stats.KVConnectorPromMetrics, "__init__", _fake_prom_init),
        mock.patch.object(stats, "create_metric_per_engine", side_effect=factory),
    ):
        pm = stats.FlagCXPromMetrics(
            None,
            {"counter": _FakeMetric, "histogram": _FakeMetric},
            ["engine"],
            {engine: ["label"]},
        )
    return pm


def _metric_map(pm, kind, engine):
    coll = pm._counters if kind == "counters" else pm._histograms
    return {key: holder.get(engine) for holder, key in coll}


# ===========================================================================
# flagcx_stats.FlagCXKVConnectorStats
# ===========================================================================
class TestFlagCXKVConnectorStats:
    def test_default_init_creates_full_columns(self):
        s = stats.FlagCXKVConnectorStats()
        assert s.data["transfer_duration"] == []
        assert s.data["num_failed_recvs"] == []
        assert s.is_empty() is True
        assert s.num_successful_transfers == 0

    def test_record_transfer_and_counters(self):
        s = stats.FlagCXKVConnectorStats()
        s.record_transfer(0.5, 1024, 3)
        assert s.num_successful_transfers == 1
        assert s.data["transfer_duration"] == [0.5]
        assert s.data["bytes_transferred"] == [1024]
        assert s.data["num_descriptors"] == [3]
        assert s.is_empty() is False

    def test_record_failures_and_expiry_make_non_empty(self):
        for method, key in (
            ("record_failed_transfer", "num_failed_transfers"),
            ("record_failed_recv", "num_failed_recvs"),
            ("record_kv_expired_req", "num_kv_expired_reqs"),
        ):
            s = stats.FlagCXKVConnectorStats()
            getattr(s, method)()
            assert s.data[key] == [1]
            assert s.is_empty() is False

    def test_clone_and_reset_snapshots_then_clears(self):
        s = stats.FlagCXKVConnectorStats()
        s.record_transfer(0.1, 2, 1)
        s.record_failed_transfer()
        snap = s.clone_and_reset()
        assert isinstance(snap, stats.FlagCXKVConnectorStats)
        assert snap.data["transfer_duration"] == [0.1]
        assert snap.data["num_failed_transfers"] == [1]
        assert s.is_empty() is True
        assert s.data["transfer_duration"] == []
        assert snap._lock is not s._lock

    def test_aggregate_extends_and_ignores_empty(self):
        a = stats.FlagCXKVConnectorStats()
        a.record_transfer(1.0, 10, 1)
        b = stats.FlagCXKVConnectorStats()
        b.record_transfer(2.0, 20, 2)
        b.record_failed_recv()
        assert a.aggregate(b) is a
        assert a.data["transfer_duration"] == [1.0, 2.0]
        assert a.data["num_failed_recvs"] == [1]

        empty = stats.FlagCXKVConnectorStats()
        before = dict(a.data)
        assert a.aggregate(empty) is a
        assert a.data == before

    def test_reduce_zero_success_branch(self):
        s = stats.FlagCXKVConnectorStats()
        s.record_failed_transfer()
        s.record_failed_recv()
        s.record_kv_expired_req()
        out = s.reduce()
        assert out["Num successful transfers"] == 0
        assert out["Num failed transfers"] == 1
        assert out["Num failed recvs"] == 1
        assert out["Num KV expired reqs"] == 1

    def test_reduce_success_branch(self):
        s = stats.FlagCXKVConnectorStats()
        s.record_transfer(0.1, 2**20, 1)
        s.record_transfer(0.2, 2**20, 3)
        out = s.reduce()
        assert out["Num successful transfers"] == 2
        assert out["Avg MB per transfer"] == 1.0
        assert out["Throughput (MB/s)"] > 0
        assert out["Avg number of descriptors"] == 2.0
        assert out["P90 xfer time (ms)"] > 0

    def test_reduce_zero_duration_throttles_throughput(self):
        s = stats.FlagCXKVConnectorStats()
        s.record_transfer(0.0, 0, 0)
        out = s.reduce()
        assert out["Throughput (MB/s)"] == 0.0

    def test_pickle_roundtrip_rebuilds_lock(self):
        s = stats.FlagCXKVConnectorStats()
        s.record_transfer(0.3, 5, 1)
        restored = pickle.loads(pickle.dumps(s))
        assert restored.data == s.data
        assert restored._lock is not s._lock
        assert restored._lock is not None


# ===========================================================================
# flagcx_stats.FlagCXPromMetrics
# ===========================================================================
class TestFlagCXPromMetrics:
    def test_construct_wires_counters_and_histograms(self):
        pm = _make_prom(engine=0)
        counter_keys = {key for _, key in pm._counters}
        assert "num_transfers" in counter_keys
        assert "num_failed_recvs" in counter_keys
        hist_keys = {key for _, key in pm._histograms}
        assert hist_keys == {"transfer_duration", "bytes_transferred", "num_descriptors"}

    def test_observe_advances_counters_and_histograms(self):
        pm = _make_prom(engine=0)
        data = {
            "transfer_duration": [0.1, 0.2],
            "bytes_transferred": [10, 20],
            "num_descriptors": [1, 2],
            "num_failed_transfers": [1],
            "num_failed_recvs": [1, 1],
            "num_kv_expired_reqs": [1],
        }
        pm.observe(data, engine_idx=0)

        counters = _metric_map(pm, "counters", 0)
        assert counters["num_transfers"].incs == [2]
        assert counters["num_failed_recvs"].incs == [1, 1]
        assert counters["bytes_transferred"].incs == [10, 20]

        hist = _metric_map(pm, "histograms", 0)
        assert hist["transfer_duration"].observes == [0.1, 0.2]
        assert hist["num_descriptors"].observes == [1, 2]

    def test_observe_missing_engine_and_missing_keys(self):
        pm = _make_prom(engine=0)
        pm.observe({}, engine_idx=7)
        for metric in _metric_map(pm, "counters", 0).values():
            assert metric.incs == []
        all_metrics = [m for d, _ in pm._counters for m in d.values()]
        all_metrics += [m for d, _ in pm._histograms for m in d.values()]
        assert all(not m.incs and not m.observes for m in all_metrics)


# ===========================================================================
# Pure module-level helpers
# ===========================================================================
class TestPureHelpers:
    def test_group_concurrent_contiguous(self):
        assert connector.group_concurrent_contiguous([], []) == ([], [])
        both = connector.group_concurrent_contiguous([1, 2, 3], [4, 5, 6])
        assert both == ([[1, 2, 3]], [[4, 5, 6]])
        split = connector.group_concurrent_contiguous([1, 3, 4], [1, 2, 3])
        assert split == ([[1], [3, 4]], [[1], [2, 3]])

    def test_get_tp_ratio_equal_and_divisible(self):
        assert connector._get_tp_ratio(1, 1) == 1
        assert connector._get_tp_ratio(4, 2) == 2
        assert connector._get_tp_ratio(1, 4) == -4

    def test_get_tp_ratio_rejects_indivisible(self):
        with pytest.raises(AssertionError):
            connector._get_tp_ratio(3, 2)
        with pytest.raises(AssertionError):
            connector._get_tp_ratio(2, 3)

    def test_expand_transfer_regions_default(self):
        regions = connector._expand_transfer_regions(
            base_addrs=[100],
            block_lens=[10],
            kv_block_lens=[8],
            layer_names=["l"],
            layer_indices=[0],
            is_kv_layout_blocks_first=False,
        )
        assert len(regions) == 1
        assert regions[0].base_addr == 100
        assert regions[0].group_index == 0

    def test_expand_transfer_regions_split_blocks_first(self):
        regions = connector._expand_transfer_regions(
            base_addrs=[100, 200],
            block_lens=[10, 10],
            kv_block_lens=[4, 4],
            layer_names=["a", "b"],
            layer_indices=[0, 1],
            is_kv_layout_blocks_first=True,
            group_indices=[0, 1],
        )
        assert len(regions) == 4
        assert regions[0].base_addr == 100
        assert regions[1].base_addr == 104  # V half at base + kv_block_len
        assert regions[2].group_index == 1

    def test_expand_transfer_regions_explicit_no_split(self):
        regions = connector._expand_transfer_regions(
            base_addrs=[7],
            block_lens=[3],
            kv_block_lens=[3],
            layer_names=["x"],
            layer_indices=[2],
            is_kv_layout_blocks_first=True,
            group_indices=[5],
            split_kv_regions=[False],
        )
        assert len(regions) == 1
        assert regions[0].group_index == 5

    def test_expand_transfer_regions_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            connector._expand_transfer_regions(
                [1], [1], [1], ["a"], [], False
            )
        with pytest.raises(AssertionError):
            connector._expand_transfer_regions(
                [1], [1], [1], ["a"], [0], False, group_indices=[0, 1]
            )
        with pytest.raises(AssertionError):
            connector._expand_transfer_regions(
                [1], [1], [1], ["a"], [0], False, split_kv_regions=[True, False]
            )

    def test_compute_sender_transfer_plan_variants(self):
        # tp_ratio == 1 -> full copy
        assert connector._compute_sender_transfer_plan(0, 1, 0, 1, 64, 64, False) == (
            True, 0, 0, 64,
        )
        # producer wider, not replicated -> write own shard into dst slot
        assert connector._compute_sender_transfer_plan(1, 2, 0, 1, 32, 64, False) == (
            True, 0, 32, 32,
        )
        # producer wider, replicated -> only rank 0 sends
        assert connector._compute_sender_transfer_plan(
            0, 2, 0, 1, 32, 64, True
        ) == (True, 0, 0, 32)
        assert connector._compute_sender_transfer_plan(
            1, 2, 0, 1, 32, 64, True
        ) == (False, 0, 0, 32)
        # producer narrower, replicated -> full copy
        assert connector._compute_sender_transfer_plan(
            0, 1, 0, 4, 16, 4, True
        ) == (True, 0, 0, 16)
        # producer narrower, not replicated -> gather from remote shard
        assert connector._compute_sender_transfer_plan(
            0, 1, 1, 4, 16, 4, False
        ) == (True, 4, 0, 4)

    def test_can_coalesce_block_transfers(self):
        assert connector._can_coalesce_block_transfers(10, 10, 0, 0, 10) is True
        assert connector._can_coalesce_block_transfers(10, 10, 1, 0, 10) is False
        assert connector._can_coalesce_block_transfers(10, 10, 0, 1, 10) is False
        assert connector._can_coalesce_block_transfers(20, 10, 0, 0, 10) is False
        assert connector._can_coalesce_block_transfers(10, 20, 0, 0, 10) is False

    def test_validate_asymmetric_region_lengths(self):
        a = [_region(kv_block_len=10)]
        b = [_region(kv_block_len=10)]
        assert connector._validate_asymmetric_region_lengths(a, [], 1, 1, False)
        assert connector._validate_asymmetric_region_lengths(a, b, 4, 2, True) is None
        assert connector._validate_asymmetric_region_lengths(a, b, 1, 1, False) is None
        assert connector._validate_asymmetric_region_lengths(
            [_region(kv_block_len=10)], [_region(kv_block_len=20)], 1, 1, False
        )
        assert connector._validate_asymmetric_region_lengths(
            [_region(kv_block_len=10)], [_region(kv_block_len=20)], 2, 1, False
        ) is None
        assert connector._validate_asymmetric_region_lengths(
            [_region(kv_block_len=10)], [_region(kv_block_len=10)], 2, 1, False
        )
        assert connector._validate_asymmetric_region_lengths(
            [_region(kv_block_len=20)], [_region(kv_block_len=10)], 1, 2, False
        ) is None
        assert connector._validate_asymmetric_region_lengths(
            [_region(kv_block_len=10)], [_region(kv_block_len=10)], 1, 2, False
        )

    def test_align_transfer_regions(self):
        local = [_region(layer="a", group=0), _region(layer="a", group=0)]
        remote = [_region(layer="a", group=0), _region(layer="a", group=0)]
        aligned_l, aligned_r, err = connector._align_transfer_regions(local, remote)
        assert err is None
        assert len(aligned_l) == len(aligned_r) == 2

    def test_align_transfer_regions_errors(self):
        local = [_region(layer="a", group=0)]
        _, _, err = connector._align_transfer_regions(
            local, [_region(layer="b", group=0)]
        )
        assert err is not None and "no matching consumer occurrence" in err

        _, _, err = connector._align_transfer_regions(
            [_region(layer="a", idx=1)], [_region(layer="a", idx=2)]
        )
        assert err is not None and "layer index mismatch" in err

        _, _, err = connector._align_transfer_regions(
            [_region(layer="a", group=1)], [_region(layer="a", group=2)]
        )
        assert err is not None and "group index mismatch" in err

    def test_get_side_channel_port(self, monkeypatch):
        monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)
        cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_rank=2, tensor_parallel_size=3
            )
        )
        assert connector._get_side_channel_port(cfg) == 8998 + 2 * 3

        monkeypatch.setenv("FLAGCX_BOOTSTRAP_PORT", "7000")
        cfg.parallel_config.data_parallel_rank = 0
        cfg.parallel_config.tensor_parallel_size = 4
        assert connector._get_side_channel_port(cfg) == 7000


# ===========================================================================
# Metadata / dataclasses
# ===========================================================================
class TestMetadata:
    def test_agent_metadata_msgspec_roundtrip(self):
        meta = _agent_meta({"d1": ("t1", [[1, 2]])})
        raw = connector.msgspec.msgpack.encode(meta)
        decoded = connector.msgspec.msgpack.decode(raw, type=connector.FlagCXAgentMetadata)
        assert decoded.remote_hostname == "h"
        assert decoded.req_blocks["d1"] == ("t1", [[1, 2]])

    def test_connector_metadata_add_recv_req(self):
        md = connector.FlagCXConnectorMetadata()
        md.add_new_req(
            "r1",
            [[1, 2]],
            {
                "transfer_id": "tid",
                "remote_host": "h",
                "remote_port": 9,
                "remote_tp_size": 4,
            },
        )
        recv = md.reqs_to_recv["r1"]
        assert recv.transfer_id == "tid"
        assert recv.remote_port == 9
        assert recv.remote_tp_size == 4

    def test_connector_metadata_add_send_req_and_default_tp(self):
        md = connector.FlagCXConnectorMetadata()
        md.add_new_req(
            "r2",
            [[3]],
            {"transfer_id": "tid2", "remote_host": "h", "remote_port": 8},
        )
        assert md.reqs_to_recv["r2"].remote_tp_size == 0

        md.add_new_req(
            "r3",
            [[4]],
            {"transfer_id": "tid3"},
            load_remote_cache=False,
        )
        assert md.reqs_to_send["r3"] == ("tid3", [[4]])

    def test_dataclasses_defaults(self):
        recv = connector.RecvReqMeta([], "h", 1, "tid")
        assert recv.remote_tp_size == 0
        sb = _send_meta()
        assert sb.expire_time == float("inf")
        assert sb.need_send == 0 and sb.sent == 0
        assert connector.SendReqMeta({}, threading.Lock()).reqs == {}
        assert connector.FinishedSendReqSet(set(), threading.Lock()).set == set()
        assert connector.FinishedReceiveReqSet(set(), asyncio.Lock()).set == set()


# ===========================================================================
# FlagCXConnector (delegating facade + classmethods)
# ===========================================================================
class TestFlagCXConnectorFacade:
    def _bare(self, scheduler=None, worker=None):
        inst = object.__new__(connector.FlagCXConnector)
        inst.connector_scheduler = scheduler
        inst.connector_worker = worker
        return inst

    def test_get_required_kvcache_layout(self):
        none_cfg = SimpleNamespace(model_config=None)
        assert connector.FlagCXConnector.get_required_kvcache_layout(none_cfg) is None

        mla_cfg = SimpleNamespace(model_config=SimpleNamespace(use_mla=True))
        assert connector.FlagCXConnector.get_required_kvcache_layout(mla_cfg) is None

        dense_cfg = SimpleNamespace(model_config=SimpleNamespace(use_mla=False))
        assert connector.FlagCXConnector.get_required_kvcache_layout(dense_cfg) == "HND"

    def test_build_kv_connector_stats(self):
        built = connector.FlagCXConnector.build_kv_connector_stats()
        assert isinstance(built, stats.FlagCXKVConnectorStats)
        built2 = connector.FlagCXConnector.build_kv_connector_stats(
            {"transfer_duration": [0.1]}
        )
        assert built2.data["transfer_duration"] == [0.1]

    def test_build_prom_metrics(self):
        sentinel = object()
        with mock.patch.object(
            connector, "FlagCXPromMetrics", return_value=sentinel
        ) as prom:
            out = connector.FlagCXConnector.build_prom_metrics(
                "cfg", {}, ["e"], {0: ["l"]}
            )
        assert out is sentinel
        prom.assert_called_once_with("cfg", {}, ["e"], {0: ["l"]})

    def test_scheduler_delegates(self):
        sched = mock.MagicMock()
        inst = self._bare(scheduler=sched)
        req = object()
        inst.get_num_new_matched_tokens(req, 5)
        sched.get_num_new_matched_tokens.assert_called_once_with(req, 5)

        blocks = object()
        inst.update_state_after_alloc(req, blocks, 3)
        sched.update_state_after_alloc.assert_called_once_with(req, blocks, 3)

        out = inst.build_connector_meta("sched_out")
        assert out is sched.build_connector_meta.return_value
        sched.build_connector_meta.assert_called_once_with("sched_out")

        inst.request_finished(req, [1, 2])
        sched.request_finished.assert_called_once_with(req, ([1, 2],))

        inst.request_finished_all_groups(req, ([1], [2]))
        sched.request_finished.assert_called_with(req, ([1], [2]))

    def test_worker_delegates(self):
        worker = mock.MagicMock()
        inst = self._bare(worker=worker)

        inst.register_kv_caches({"l": "tensor"})
        worker.register_kv_caches.assert_called_once_with({"l": "tensor"})

        out = inst.get_finished({"a"})
        assert out is worker.get_finished.return_value

        md = connector.FlagCXConnectorMetadata()
        inst._connector_metadata = md
        inst.start_load_kv(object())
        worker.start_load_kv.assert_called_once_with(md)

        inst.wait_for_layer_load("layer0")
        worker.wait_for_layer_load.assert_called_once()

        assert inst.save_kv_layer("layer0", object(), object()) is None
        assert inst.wait_for_save() is None

        stats_out = inst.get_kv_connector_stats()
        assert stats_out is worker.get_kv_connector_stats.return_value

    def test_facade_without_worker(self):
        inst = self._bare(scheduler=None, worker=None)
        assert inst.get_kv_connector_stats() is None
        inst.wait_for_layer_load("layer")  # None worker: no-op
        assert inst.save_kv_layer("l", object(), object()) is None
        assert inst.wait_for_save() is None

    def test_start_load_kv_requires_metadata_type(self):
        worker = mock.MagicMock()
        inst = self._bare(worker=worker)
        inst._connector_metadata = object()
        with pytest.raises(AssertionError):
            inst.start_load_kv(object())


# ===========================================================================
# FlagCXConnectorScheduler (bare instance, pure logic)
# ===========================================================================
def _make_scheduler(**overrides):
    s = object.__new__(connector.FlagCXConnectorScheduler)
    s.block_size = 16
    s.engine_id = "eng"
    s.side_channel_host = "127.0.0.1"
    s.side_channel_port = 8998
    s.kv_role = "kv_both"
    s._is_hma_required = False
    s._has_mamba = False
    s.blocks_per_sw = []
    s._reqs_need_recv = {}
    s._reqs_need_send = {}
    s.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1)
    )
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


class TestSchedulerLogic:
    def test_get_sw_clipped_blocks(self):
        s = _make_scheduler(_is_hma_required=True, blocks_per_sw=[2, 0])
        assert _make_scheduler().get_sw_clipped_blocks([]) == []
        assert s.get_sw_clipped_blocks([[1, 2, 3], [4, 5, 6]]) == [
            [2, 3],
            [4, 5, 6],
        ]
        not_hma = _make_scheduler(_is_hma_required=False, blocks_per_sw=[1])
        assert not_hma.get_sw_clipped_blocks([[1, 2]]) == [[1, 2]]

    def test_remote_prefill_token_count(self):
        assert _make_scheduler(_has_mamba=True)._get_remote_prefill_token_count(5) == 4
        assert _make_scheduler(_has_mamba=True)._get_remote_prefill_token_count(1) == 1
        assert _make_scheduler(_has_mamba=False)._get_remote_prefill_token_count(5) == 5

    def _request(self, **over):
        req = SimpleNamespace(
            request_id="r",
            kv_transfer_params={},
            num_prompt_tokens=4,
            prompt_token_ids=[1, 2, 3, 4],
            prompt_embeds=None,
            _all_token_ids=[1, 2, 3, 4],
            max_tokens=8,
            status=connector.RequestStatus.FINISHED_LENGTH_CAPPED,
        )
        for key, value in over.items():
            setattr(req, key, value)
        return req

    def test_truncate_mamba_request_branches(self):
        s = _make_scheduler(_has_mamba=True)

        none_params = self._request(kv_transfer_params=None)
        s._truncate_mamba_request_for_prefill(none_params)
        assert none_params.num_prompt_tokens == 4

        already = self._request(kv_transfer_params={"_p_side_truncated": True})
        s._truncate_mamba_request_for_prefill(already)
        assert already.num_prompt_tokens == 4

        single = self._request(num_prompt_tokens=1)
        s._truncate_mamba_request_for_prefill(single)
        assert single.num_prompt_tokens == 1

        token_req = self._request()
        s._truncate_mamba_request_for_prefill(token_req)
        assert token_req.prompt_token_ids == [1, 2, 3]
        assert token_req._all_token_ids == [1, 2, 3]
        assert token_req.num_prompt_tokens == 3
        assert token_req.max_tokens == 1
        assert token_req.kv_transfer_params["_p_side_truncated"] is True

        embed_req = self._request(prompt_token_ids=None, prompt_embeds=[1, 2, 3, 4])
        s._truncate_mamba_request_for_prefill(embed_req)
        assert embed_req.prompt_embeds == [1, 2, 3]

        neither = self._request(prompt_token_ids=None, prompt_embeds=None)
        s._truncate_mamba_request_for_prefill(neither)
        assert neither.num_prompt_tokens == 4

    def test_get_num_new_matched_tokens_remote_prefill(self):
        s = _make_scheduler(_has_mamba=False)
        req = self._request(
            kv_transfer_params={"do_remote_prefill": True},
            prompt_token_ids=[1, 2, 3, 4],
        )
        assert s.get_num_new_matched_tokens(req, 1) == (3, True)
        assert s.get_num_new_matched_tokens(req, 10) == (0, False)

    def test_get_num_new_matched_tokens_truncates_for_remote_decode(self):
        s = _make_scheduler(_has_mamba=True)
        req = self._request(kv_transfer_params={"do_remote_decode": True})
        assert s.get_num_new_matched_tokens(req, 0) == (0, False)
        assert req.num_prompt_tokens == 3

    def test_get_num_new_matched_tokens_default(self):
        s = _make_scheduler()
        req = self._request(kv_transfer_params=None)
        assert s.get_num_new_matched_tokens(req, 0) == (0, False)

    def test_update_state_after_alloc_no_params(self):
        s = _make_scheduler()
        req = self._request(kv_transfer_params=None)
        s.update_state_after_alloc(req, object(), 0)
        assert s._reqs_need_recv == {}

    def test_update_state_after_alloc_remote_prefill(self):
        s = _make_scheduler()
        blocks = mock.MagicMock()
        blocks.get_unhashed_block_ids_all_groups.return_value = [[1, 2]]
        req = self._request(
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_host": "h",
                "remote_port": 9,
                "transfer_id": "t",
            }
        )
        s.update_state_after_alloc(req, blocks, 2)
        assert "r" in s._reqs_need_recv
        assert req.kv_transfer_params["do_remote_prefill"] is False

        # num_external_tokens == 0 -> empty block ids, no call to get_unhashed
        blocks2 = mock.MagicMock()
        req2 = self._request(
            request_id="r2",
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_host": "h",
                "remote_port": 9,
                "transfer_id": "t2",
            },
        )
        s.update_state_after_alloc(req2, blocks2, 0)
        assert s._reqs_need_recv["r2"][1] == []
        blocks2.get_unhashed_block_ids_all_groups.assert_not_called()

    def test_update_state_after_alloc_invalid_prefill_params(self):
        s = _make_scheduler()
        req = self._request(
            kv_transfer_params={"do_remote_prefill": True, "remote_host": "h"}
        )
        s.update_state_after_alloc(req, mock.MagicMock(), 1)
        assert s._reqs_need_recv == {}
        assert req.kv_transfer_params["do_remote_prefill"] is False

    def test_update_state_after_alloc_remote_decode(self):
        s = _make_scheduler()
        missing = self._request(
            request_id="rm", kv_transfer_params={"do_remote_decode": True}
        )
        s.update_state_after_alloc(missing, object(), 0)
        assert s._reqs_need_send == {}

        ok = self._request(
            request_id="ro",
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "t"},
        )
        s.update_state_after_alloc(ok, object(), 0)
        assert s._reqs_need_send["ro"] == (ok, [])

    def test_build_connector_meta_both_roles(self):
        s = _make_scheduler(kv_role="kv_both")
        recv_req = self._request(
            request_id="r1",
            kv_transfer_params={
                "transfer_id": "t1",
                "remote_host": "h",
                "remote_port": 9,
            },
        )
        send_req = self._request(
            request_id="r2",
            kv_transfer_params={"transfer_id": "t2"},
        )
        s._reqs_need_recv["r1"] = (recv_req, [[1]])
        s._reqs_need_send["r2"] = (send_req, [[2]])
        md = s.build_connector_meta(object())
        assert "r1" in md.reqs_to_recv
        assert md.reqs_to_send["r2"] == ("t2", [[2]])
        assert s._reqs_need_recv == {} and s._reqs_need_send == {}

    def test_build_connector_meta_producer_only_sends(self):
        s = _make_scheduler(kv_role="kv_producer")
        req = self._request(
            request_id="r", kv_transfer_params={"transfer_id": "t"}
        )
        s._reqs_need_recv["skip"] = (req, [])
        s._reqs_need_send["r"] = (req, [[1]])
        md = s.build_connector_meta(object())
        assert md.reqs_to_recv == {}
        assert "r" in md.reqs_to_send
        # For a producer the recv branch is skipped, so its stale entry stays.
        assert "skip" in s._reqs_need_recv

    def test_build_connector_meta_consumer_only_recvs(self):
        s = _make_scheduler(kv_role="kv_consumer")
        req = self._request(
            request_id="r",
            kv_transfer_params={
                "transfer_id": "t",
                "remote_host": "h",
                "remote_port": 9,
            },
        )
        s._reqs_need_recv["r"] = (req, [[1]])
        s._reqs_need_send["skip"] = (req, [])
        md = s.build_connector_meta(object())
        assert "r" in md.reqs_to_recv
        assert md.reqs_to_send == {}
        assert "skip" in s._reqs_need_send

    def test_request_finished_early_returns(self):
        s = _make_scheduler(kv_role="kv_producer")
        assert s.request_finished(
            self._request(kv_transfer_params=None), ([],)
        ) == (False, None)
        assert s.request_finished(
            self._request(kv_transfer_params={}), ([],)
        ) == (False, None)
        assert s.request_finished(
            self._request(kv_transfer_params={"transfer_id": None}), ([],)
        ) == (False, None)

    def test_request_finished_remote_prefill(self):
        s = _make_scheduler(kv_role="kv_consumer")
        req = self._request(
            kv_transfer_params={"transfer_id": "t", "do_remote_prefill": True}
        )
        out = s.request_finished(req, ([],))
        assert out == (False, None)
        assert s._reqs_need_recv["r"] == (req, [])
        assert req.kv_transfer_params["do_remote_prefill"] is False

    def test_request_finished_non_capped_returns(self):
        s = _make_scheduler(kv_role="kv_producer")
        not_decode = self._request(kv_transfer_params={"transfer_id": "t"})
        assert s.request_finished(not_decode, ([],)) == (False, None)

        wrong_status = self._request(
            kv_transfer_params={"transfer_id": "t", "do_remote_decode": True},
            status="RUNNING",
        )
        assert s.request_finished(wrong_status, ([1],)) == (False, None)

    def test_request_finished_success_delays_blocks(self):
        s = _make_scheduler(kv_role="kv_producer")
        req = self._request(
            kv_transfer_params={"transfer_id": "t", "do_remote_decode": True}
        )
        out, payload = s.request_finished(req, ([1, 2],))
        assert out is True
        assert payload["do_remote_prefill"] is True
        assert payload["do_remote_decode"] is False
        assert payload["remote_host"] == "127.0.0.1"
        assert payload["remote_port"] == 8998
        assert payload["transfer_id"] == "t"
        assert "r" in s._reqs_need_send

    def test_request_finished_success_no_blocks(self):
        s = _make_scheduler(kv_role="kv_producer")
        req = self._request(
            kv_transfer_params={"transfer_id": "t", "do_remote_decode": True}
        )
        out, payload = s.request_finished(req, ([],))
        assert out is False
        assert payload is not None and payload["transfer_id"] == "t"
        assert s._reqs_need_send == {}


# ===========================================================================
# FlagCXConnectorWorker (no threads, no sockets)
# ===========================================================================
class _Noop:
    pass


def _make_worker(**overrides):
    w = object.__new__(connector.FlagCXConnectorWorker)
    w.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(use_mla=False, get_total_num_kv_heads=lambda: 8),
        kv_transfer_config=SimpleNamespace(
            engine_id="eng", kv_role="kv_both", kv_connector_extra_config={}
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0, tensor_parallel_size=1),
    )
    w.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=_Noop())],
        has_mamba_layers=False,
    )
    w.engine_id = "eng"
    w.hostname = "host-1"
    w.kv_role = "kv_both"
    w.tp_rank = 0
    w.tp_size = 1
    w.block_size = 16
    w.use_mla = False
    w._physical_blocks_per_logical_kv_block = 1
    w._abort_request_timeout = 1
    w._layer_specs = {}
    w.kv_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        is_kv_layout_blocks_first=False,
        handshake_target_ranks=lambda size: [0],
    )
    w.reqs_need_send = connector.SendReqMeta(reqs={}, lock=threading.Lock())
    w.finished_sending_reqs = connector.FinishedSendReqSet(set(), threading.Lock())
    w.finished_recving_reqs = connector.FinishedReceiveReqSet(set(), asyncio.Lock())
    w.xfer_stats = stats.FlagCXKVConnectorStats()
    w._pull_pending = {}
    w.session_conns = {}
    w.conn_lock = threading.Lock()
    w.engine = object()
    w.flagcx = mock.MagicMock()
    w.rpc_port = 4321
    w.kv_caches_base_addr = [1000]
    w.block_len_per_layer = [10]
    w.kv_block_len_per_layer = [10]
    w.registered_layer_names = ["l0"]
    w.registered_layer_indices = [0]
    w.registered_group_indices = [0]
    w._encoder = connector.msgspec.msgpack.Encoder()
    w.zmq_ctx = mock.MagicMock()
    w.async_zmq_ctx = mock.MagicMock()
    for key, value in overrides.items():
        setattr(w, key, value)
    return w


class TestWorkerLogic:
    def test_sync_block_size_no_change(self):
        w = _make_worker()
        backend = object()
        with (
            mock.patch.object(
                connector, "get_current_attn_backends", return_value=[backend]
            ),
            mock.patch.object(
                connector, "select_common_block_size", return_value=16
            ),
        ):
            w._sync_block_size_with_kernel()
        assert w.block_size == 16
        assert w._physical_blocks_per_logical_kv_block == 1

    def test_sync_block_size_reduces_to_kernel(self):
        w = _make_worker(block_size=32)
        with (
            mock.patch.object(
                connector, "get_current_attn_backends", return_value=[object()]
            ),
            mock.patch.object(
                connector, "select_common_block_size", return_value=8
            ),
        ):
            w._sync_block_size_with_kernel()
        assert w.block_size == 8
        assert w._physical_blocks_per_logical_kv_block == 4

    def test_get_conn_caches_handle(self):
        conn = object()
        w = _make_worker()
        w.flagcx = mock.MagicMock()
        w.flagcx.flagcxP2pGetConn.return_value = conn
        first = w._get_conn("h:1")
        second = w._get_conn("h:1")
        assert first is conn and second is conn
        w.flagcx.flagcxP2pGetConn.assert_called_once_with(w.engine, "h:1")
        assert w.session_conns == {"h:1": conn}

    def test_producer_cache_is_replicated(self):
        w = _make_worker(
            kv_topo=SimpleNamespace(
                local_replicates_kv_cache=True,
                is_kv_layout_blocks_first=False,
                handshake_target_ranks=lambda size: [0],
            )
        )
        assert w._producer_cache_is_replicated() is True

    def test_get_transfer_regions_no_split(self):
        w = _make_worker()
        regions = w._get_transfer_regions(
            [10], [4], [2], ["l"], [0], [0]
        )
        assert len(regions) == 1

    def test_get_transfer_regions_blocks_first_split(self):
        w = _make_worker()
        w.kv_topo = SimpleNamespace(
            local_replicates_kv_cache=False,
            is_kv_layout_blocks_first=True,
            handshake_target_ranks=lambda size: [0],
        )
        w._layer_specs = {"l": _Noop()}
        fake_a, fake_b, fake_c = type("A", (), {}), type("B", (), {}), type("C", (), {})
        with mock.patch.multiple(
            connector,
            MambaSpec=fake_a,
            MLAAttentionSpec=fake_b,
            SlidingWindowMLASpec=fake_c,
        ):
            regions = w._get_transfer_regions(
                [10], [4], [2], ["l"], [0], [0]
            )
        assert len(regions) == 2

    def test_get_sender_transfer_plan(self):
        w = _make_worker()
        assert w._get_sender_transfer_plan(8, 8, 0, 1) == (True, 0, 0, 8)

    def test_logical_to_kernel_block_ids_identity(self):
        w = _make_worker()
        blocks = [[1, 2]]
        assert w._logical_to_kernel_block_ids(blocks) is blocks

    def test_logical_to_kernel_block_ids_expands(self):
        w = _make_worker()
        w._physical_blocks_per_logical_kv_block = 2
        w.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=_Noop())]
        )
        mapped = SimpleNamespace(tolist=lambda: [4, 5])
        with mock.patch.object(
            connector.BlockTable,
            "map_to_kernel_blocks",
            return_value=mapped,
        ) as mapper:
            out = w._logical_to_kernel_block_ids([[1, 2]])
        assert out == [[4, 5]]
        assert mapper.call_count == 1

    def test_logical_to_kernel_block_ids_keeps_mamba(self):
        w = _make_worker()
        w._physical_blocks_per_logical_kv_block = 2

        class FakeMamba:
            pass

        spec = FakeMamba()
        w.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
        )
        with mock.patch.object(connector, "MambaSpec", FakeMamba):
            out = w._logical_to_kernel_block_ids([[7, 8]])
        assert out == [[7, 8]]

    def _xfer_worker(self, **over):
        w = _make_worker()
        w.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=_Noop())]
        )
        for key, value in over.items():
            setattr(w, key, value)
        return w

    def test_build_transfer_params_coalesced(self):
        w = self._xfer_worker()
        region = _region(base=1000, block_len=10, kv_block_len=10, group=0)
        sm = _send_meta(blocks=[[1, 2]], tid="t")
        meta = _agent_meta({"d": ("t", [[1, 2]])}, remote_tp_rank=0, remote_tp_size=1)
        src, dst, sizes = w._build_transfer_params(
            [("d", sm)], meta, [region], [region]
        )
        assert sizes == [20]
        assert src == [1000 + 1 * 10]
        assert dst == [1000 + 1 * 10]

    def test_build_transfer_params_non_coalesced_per_block(self):
        w = self._xfer_worker()
        region = _region(base=1000, block_len=20, kv_block_len=10, group=0)
        sm = _send_meta(blocks=[[1, 2]], tid="t")
        meta = _agent_meta({"d": ("t", [[1, 2]])}, remote_tp_rank=0, remote_tp_size=1)
        src, dst, sizes = w._build_transfer_params(
            [("d", sm)], meta, [region], [region]
        )
        assert sizes == [10, 10]
        assert src == [1000 + 1 * 20, 1000 + 2 * 20]

    def test_build_transfer_params_trims_extra_local_blocks(self):
        w = self._xfer_worker()
        region = _region(base=1000, block_len=10, kv_block_len=10, group=0)
        sm = _send_meta(blocks=[[1, 2, 3]], tid="t")
        meta = _agent_meta({"d": ("t", [[1, 2]])}, remote_tp_rank=0, remote_tp_size=1)
        _, _, sizes = w._build_transfer_params(
            [("d", sm)], meta, [region], [region]
        )
        assert sizes == [20]  # trimmed to blocks [2, 3]

    def test_build_transfer_params_skips_empty_and_mismatched(self):
        w = self._xfer_worker()
        region = _region(group=0)

        # empty remote blocks -> continue
        sm = _send_meta(blocks=[[1]], tid="t")
        meta = _agent_meta({"d": ("t", [[]])})
        assert w._build_transfer_params([("d", sm)], meta, [region], [region]) == (
            [],
            [],
            [],
        )

        # group count mismatch -> continue
        meta2 = _agent_meta({"d": ("t", [[1], [2]])})
        assert w._build_transfer_params([("d", sm)], meta2, [region], [region]) == (
            [],
            [],
            [],
        )

        # local fewer than remote -> error, continue
        sm3 = _send_meta(blocks=[[1]], tid="t")
        meta3 = _agent_meta({"d": ("t", [[1, 2]])})
        assert w._build_transfer_params([("d", sm3)], meta3, [region], [region]) == (
            [],
            [],
            [],
        )

        # region group index out of range -> continue
        region_big = _region(group=5)
        sm4 = _send_meta(blocks=[[1]], tid="t")
        meta4 = _agent_meta({"d": ("t", [[1]])})
        assert w._build_transfer_params(
            [("d", sm4)], meta4, [region_big], [region_big]
        ) == ([], [], [])

    def test_build_transfer_params_replicated_skips(self):
        w = self._xfer_worker()
        w.tp_rank = 1
        w.tp_size = 2
        w.kv_topo = SimpleNamespace(
            local_replicates_kv_cache=True,
            is_kv_layout_blocks_first=False,
            handshake_target_ranks=lambda size: [0],
        )
        region = _region(group=0)
        sm = _send_meta(blocks=[[1]], tid="t")
        meta = _agent_meta({"d": ("t", [[1]])}, remote_tp_size=1)
        assert w._build_transfer_params([("d", sm)], meta, [region], [region]) == (
            [],
            [],
            [],
        )

    def test_build_transfer_params_strips_mamba_null_blocks(self):
        w = self._xfer_worker()

        class FakeMamba:
            pass

        spec = FakeMamba()
        w.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
        )
        region = _region(base=1000, block_len=10, kv_block_len=10, group=0)
        null_id = connector.NULL_BLOCK_ID
        sm = _send_meta(blocks=[[null_id, 1]], tid="t")
        meta = _agent_meta({"d": ("t", [[null_id, 1]])}, remote_tp_size=1)
        with mock.patch.object(connector, "MambaSpec", FakeMamba):
            _, _, sizes = w._build_transfer_params(
                [("d", sm)], meta, [region], [region]
            )
        assert sizes == [10]

    def test_start_load_kv_arms_send(self):
        w = _make_worker(kv_role="kv_producer")
        md = connector.FlagCXConnectorMetadata()
        md.reqs_to_send["p1"] = ("t1", [[0, 1]])
        w.start_load_kv(md)
        sm = w.reqs_need_send.reqs["t1"]
        assert sm.p_req_id == "p1"
        assert sm.local_block_ids == [[0, 1]]
        assert sm.ready.is_set()

        # Second call hits the existing-entry branch.
        md2 = connector.FlagCXConnectorMetadata()
        md2.reqs_to_send["p1"] = ("t1", [[2]])
        w.start_load_kv(md2)
        assert w.reqs_need_send.reqs["t1"].p_req_id == "p1"

        # Empty block ids: entry created but not armed.
        md3 = connector.FlagCXConnectorMetadata()
        md3.reqs_to_send["p2"] = ("t2", [])
        w.start_load_kv(md3)
        assert w.reqs_need_send.reqs["t2"].ready.is_set() is False

    def test_wait_for_layer_load_and_get_finished_producer(self):
        w = _make_worker(kv_role="kv_producer")
        assert w.wait_for_layer_load() is None
        w.finished_sending_reqs.set.add("p1")
        expired = _send_meta(p_req="p2", tid="expired")
        expired.expire_time = 0.0
        no_req = _send_meta(p_req="", tid="expired_empty")
        no_req.expire_time = 0.0
        alive = _send_meta(p_req="p3", tid="alive")
        alive.expire_time = float("inf")
        w.reqs_need_send.reqs.update(
            {"expired": expired, "expired_empty": no_req, "alive": alive}
        )
        sending, recving = w.get_finished()
        assert sending == {"p1", "p2"}
        assert recving is None
        assert "expired" not in w.reqs_need_send.reqs
        assert "expired_empty" not in w.reqs_need_send.reqs
        assert "alive" in w.reqs_need_send.reqs
        assert w.xfer_stats.data["num_kv_expired_reqs"] == [1, 1]

    def test_get_kv_connector_stats_empty_then_clone(self):
        w = _make_worker(kv_role="kv_producer")
        assert w.get_kv_connector_stats() is None
        w.xfer_stats.record_transfer(0.5, 100, 1)
        got = w.get_kv_connector_stats()
        assert isinstance(got, stats.FlagCXKVConnectorStats)
        assert got.data["transfer_duration"] == [0.5]
        assert w.xfer_stats.is_empty() is True

    def test_fetch_finished_recving(self):
        w = _make_worker()
        w.finished_recving_reqs.set.update({"a", "b"})
        out = asyncio.run(w._fetch_finished_recving())
        assert out == {"a", "b"}
        assert w.finished_recving_reqs.set == set()

    def test_receiver_loop_fn(self):
        w = _make_worker()
        loop = mock.MagicMock()
        with mock.patch.object(connector.asyncio, "set_event_loop") as set_loop:
            w._receiver_loop_fn(loop)
        set_loop.assert_called_once_with(loop)
        loop.run_forever.assert_called_once()

    def test_group_kv_pull_fans_out(self):
        w = _make_worker()
        w.tp_size = 2
        w.kv_topo = SimpleNamespace(
            local_replicates_kv_cache=False,
            is_kv_layout_blocks_first=False,
            handshake_target_ranks=lambda size: [0, 1],
        )
        recv = {
            "r1": connector.RecvReqMeta(
                local_block_ids=[[1]], remote_host="h", remote_port=9000,
                transfer_id="t1", remote_tp_size=2,
            )
        }
        closed = []

        def fake_ensure(coro):
            closed.append(coro)
            coro.close()
            return mock.MagicMock()

        with mock.patch.object(
            connector.asyncio, "ensure_future", side_effect=fake_ensure
        ):
            asyncio.run(w._group_kv_pull(recv))
        assert w._pull_pending["r1"] == 2
        assert len(closed) == 2

    def test_receive_kv_success(self):
        w = _make_worker()
        w.kv_caches_base_addr = []
        w.block_len_per_layer = []
        w.kv_block_len_per_layer = []
        w.registered_layer_names = []
        w.registered_layer_indices = []
        w.registered_group_indices = []
        w.tp_size = 1
        w.tp_rank = 0
        sock = _FakeAsyncSock(recv_value=connector.TRANS_DONE)
        with mock.patch.object(connector, "make_zmq_socket", return_value=sock):
            asyncio.run(w._receive_kv("tcp://h:1", {"r1": ("t", [[0]])}))
        assert "r1" in w.finished_recving_reqs.set
        assert w._pull_pending == {}
        assert sock.closed is True

    def test_receive_kv_trans_error(self):
        w = _make_worker()
        sock = _FakeAsyncSock(recv_value=connector.TRANS_ERROR)
        with mock.patch.object(connector, "make_zmq_socket", return_value=sock):
            asyncio.run(w._receive_kv("tcp://h:1", {"r1": ("t", [[0]])}))
        assert w.xfer_stats.data["num_failed_recvs"] == [1]
        assert w.finished_recving_reqs.set == set()

    def test_receive_kv_exception_and_context_terminated(self):
        w = _make_worker()
        sock = _FakeAsyncSock(recv_exc=RuntimeError("boom"))
        with mock.patch.object(connector, "make_zmq_socket", return_value=sock):
            asyncio.run(w._receive_kv("tcp://h:1", {"r1": ("t", [[0]])}))
        assert w.xfer_stats.data["num_failed_recvs"] == [1]

        w2 = _make_worker()
        sock2 = _FakeAsyncSock(recv_exc=connector.zmq.ContextTerminated())
        with mock.patch.object(connector, "make_zmq_socket", return_value=sock2):
            asyncio.run(w2._receive_kv("tcp://h:1", {"r1": ("t", [[0]])}))
        assert w2.xfer_stats.data["num_failed_recvs"] == []

    def test_sender_worker_success_and_error(self):
        w = _make_worker()
        w._decoder = mock.MagicMock()
        w._decoder.decode.return_value = "metadata"
        w._send_kv_to_decode = mock.MagicMock()
        pusher = mock.MagicMock()
        with mock.patch.object(connector, "make_zmq_socket", return_value=pusher):
            w._sender_worker(b"id", b"raw", "inproc://chan")
        w._send_kv_to_decode.assert_called_once_with("metadata")
        pusher.send_multipart.assert_called_once_with((b"id", connector.TRANS_DONE))

        w._decoder.decode.side_effect = ValueError("bad")
        pusher2 = mock.MagicMock()
        with mock.patch.object(connector, "make_zmq_socket", return_value=pusher2):
            w._sender_worker(b"id", b"raw", "inproc://chan")
        pusher2.send_multipart.assert_called_once_with(
            (b"id", connector.TRANS_ERROR)
        )

        pusher3 = mock.MagicMock()
        pusher3.send_multipart.side_effect = connector.zmq.ZMQError("closed")
        w._decoder.decode.side_effect = None
        with mock.patch.object(connector, "make_zmq_socket", return_value=pusher3):
            w._sender_worker(b"id", b"raw", "inproc://chan")
        pusher3.close.assert_called_once()

    def test_sender_thread_dispatch_and_shutdown(self):
        w = _make_worker()
        w._sender_executor = mock.MagicMock()
        frontend = _FakeMsock()
        backend = _FakeMsock()
        backend.recv_multipart = lambda: (b"id", b"done")

        class FakePoller:
            def __init__(self):
                self.seq = [
                    [(frontend, connector.zmq.POLLIN)],
                    [(backend, connector.zmq.POLLIN)],
                    connector.zmq.ContextTerminated(),
                ]
                self.registered = []

            def register(self, sock, event):
                self.registered.append(sock)

            def poll(self, timeout=None):
                item = self.seq.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        poller = FakePoller()
        ready = mock.MagicMock()
        with (
            mock.patch.object(
                connector, "make_zmq_socket", side_effect=[frontend, backend]
            ),
            mock.patch.object(connector.zmq, "Poller", lambda: poller),
        ):
            w._sender_thread(ready, 9000, 0)
        ready.set.assert_called_once()
        w._sender_executor.submit.assert_called_once()
        assert frontend.sent == [(b"id", b"", b"done")]
        assert frontend.closed is True and backend.closed is True

    def test_shutdown_with_engine_and_without(self):
        w = _make_worker(kv_role="kv_producer")
        w._sender_executor = mock.MagicMock()
        w._sender_t = None
        w.engine = object()
        w.flagcx = mock.MagicMock()
        w.shutdown()
        w.zmq_ctx.term.assert_called_once()
        w.async_zmq_ctx.term.assert_called_once()
        w._sender_executor.shutdown.assert_called_once_with(wait=False)
        w.flagcx.flagcxP2pEngineDestroy.assert_called_once()
        assert w.engine is None

        w2 = _make_worker(kv_role="kv_producer")
        w2._sender_executor = mock.MagicMock()
        w2._sender_t = None
        w2.engine = None
        w2.flagcx = mock.MagicMock()
        w2.shutdown()
        w2.flagcx.flagcxP2pEngineDestroy.assert_not_called()

    def test_shutdown_engine_destroy_failure_is_logged(self):
        w = _make_worker(kv_role="kv_producer")
        w._sender_executor = mock.MagicMock()
        w._sender_t = None
        w.engine = object()
        w.flagcx = mock.MagicMock()
        w.flagcx.flagcxP2pEngineDestroy.side_effect = RuntimeError("destroy")
        w.shutdown()
        assert w.engine is None

    def test_shutdown_receiver_loop_stop(self):
        w = _make_worker(kv_role="kv_consumer")
        w.receiver_loop = mock.MagicMock()
        w.receiver_loop.is_running.return_value = True
        w._receiver_t = mock.MagicMock()
        w.engine = None
        w.shutdown()
        w.receiver_loop.call_soon_threadsafe.assert_called_once()
        w._receiver_t.join.assert_called_once_with(timeout=2)

    def test_del_calls_shutdown(self):
        w = object.__new__(connector.FlagCXConnectorWorker)
        w.shutdown = mock.MagicMock()
        w.__del__()
        w.shutdown.assert_called_once()


class _FakeAsyncSock:
    def __init__(self, recv_value=None, recv_exc=None):
        self.recv_value = recv_value
        self.recv_exc = recv_exc
        self.sent = []
        self.opts = []
        self.closed = False

    def setsockopt(self, *args):
        self.opts.append(args)

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if self.recv_exc is not None:
            raise self.recv_exc
        return self.recv_value

    def close(self):
        self.closed = True


class _FakeMsock:
    def __init__(self):
        self.sent = []
        self.closed = False

    def recv_multipart(self):
        return (b"id", b"", b"meta")

    def send_multipart(self, parts):
        self.sent.append(parts)

    def close(self):
        self.closed = True


# ===========================================================================
# FlagCXConnectorWorker.__init__ (fully mocked, no threads / no ZMQ / no GCU)
# ===========================================================================
def _make_vllm_config(block_size=16, kv_role="kv_both"):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(
            use_mla=False, get_total_num_kv_heads=lambda: 8
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id="eng",
            kv_role=kv_role,
            kv_connector_extra_config={"num_workers": 3},
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0, tensor_parallel_size=1
        ),
    )


def _make_kv_cache_config():
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=_Noop(),
            )
        ],
        has_mamba_layers=False,
    )


def _init_worker(monkeypatch, kv_role="kv_both", lib_env=None):
    if lib_env is None:
        monkeypatch.delenv("FLAGCX_LIB_PATH", raising=False)
    else:
        monkeypatch.setenv("FLAGCX_LIB_PATH", lib_env)
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)

    flagcx_cls = mock.MagicMock(name="FLAGCXLibrary")
    flagcx = flagcx_cls.return_value
    fake_thread = mock.MagicMock(name="thread")
    fake_loop = mock.MagicMock(name="loop")
    backend = mock.MagicMock()
    backend.get_name.return_value = "flash"

    patches = [
        mock.patch.multiple(
            connector,
            FLAGCXLibrary=flagcx_cls,
            get_ip=mock.MagicMock(return_value="host-1"),
            get_tensor_model_parallel_rank=mock.MagicMock(return_value=2),
            get_tensor_model_parallel_world_size=mock.MagicMock(return_value=4),
            get_current_attn_backends=mock.MagicMock(return_value=[backend]),
            select_common_block_size=mock.MagicMock(
                side_effect=lambda bs, be: bs
            ),
            get_kv_cache_layout=mock.MagicMock(return_value="NHD"),
            TransferTopology=mock.MagicMock(name="TransferTopology"),
            ThreadPoolExecutor=mock.MagicMock(name="TPE"),
        ),
        mock.patch.object(
            connector.threading, "Thread", mock.MagicMock(return_value=fake_thread)
        ),
        mock.patch.object(
            connector.asyncio, "new_event_loop", mock.MagicMock(return_value=fake_loop)
        ),
        mock.patch.object(connector.zmq, "Context", mock.MagicMock(name="ZmqCtx")),
        mock.patch.object(
            connector.zmq.asyncio, "Context", mock.MagicMock(name="AsyncZmqCtx")
        ),
    ]
    with contextlib.ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        worker = connector.FlagCXConnectorWorker(
            _make_vllm_config(kv_role=kv_role), "eng", _make_kv_cache_config()
        )
    return worker, flagcx_cls, flagcx, fake_thread, fake_loop, backend


class TestWorkerInit:
    def test_init_both_roles_default_lib_path(self, monkeypatch):
        worker, flagcx_cls, flagcx, fake_thread, fake_loop, _ = _init_worker(
            monkeypatch, kv_role="kv_both", lib_env=None
        )
        assert worker.flagcx is flagcx
        assert worker.engine is flagcx.flagcxP2pEngineCreate.return_value
        assert worker.hostname == "host-1"
        assert worker.tp_rank == 2
        assert worker.tp_size == 4
        assert worker.num_workers == 3
        assert worker.rpc_port == 0
        assert worker.kv_cache_device is None
        assert worker.kv_role == "kv_both"
        assert worker.receiver_loop is fake_loop
        assert fake_thread.start.called
        # library path branch: no FLAGCX_LIB_PATH -> derived from FLAGCX_PATH=""
        called_path = flagcx_cls.call_args.args[0]
        assert called_path.endswith("build/lib/libflagcx.so")
        # clean up the mocked receiver/sender resources
        worker.shutdown()

    def test_init_producer_with_explicit_lib_path(self, monkeypatch):
        worker, flagcx_cls, flagcx, fake_thread, _, _ = _init_worker(
            monkeypatch, kv_role="kv_producer", lib_env="/opt/libflagcx.so"
        )
        assert worker.kv_role == "kv_producer"
        assert flagcx_cls.call_args.args[0] == "/opt/libflagcx.so"
        assert worker._sender_executor is not None
        # Producer must not start the receiver loop/thread.
        assert not hasattr(worker, "receiver_loop")
        worker.shutdown()

    def test_init_consumer_skips_sender(self, monkeypatch):
        worker, _, _, _, _, _ = _init_worker(
            monkeypatch, kv_role="kv_consumer", lib_env="/opt/libflagcx.so"
        )
        assert worker.kv_role == "kv_consumer"
        assert not hasattr(worker, "_sender_executor")
        worker.shutdown()

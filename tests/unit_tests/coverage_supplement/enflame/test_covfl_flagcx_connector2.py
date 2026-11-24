"""Supplementary, hardware-free coverage tests for the FlagCX connector.

Target (commit 030b452f, read-only):
  vllm_fl/distributed/kv_transfer/flagcx_connector.py

Complements ``test_covfl_flagcx_connector.py`` by exercising paths that file
leaves uncovered:

  * FlagCXConnector.__init__ (both roles + guard asserts)
  * FlagCXConnectorScheduler.__init__ (HMA / sliding-window / mamba detection)
  * FlagCXConnectorWorker.register_kv_caches (MLA / blocks-first / Mamba /
    storage dedup / empty-registry error / role-specific side effects)
  * FlagCXConnectorWorker._send_kv_to_decode (success, rank warning, timeout,
    alignment / length validation errors, write failure, empty plan, partial)
  * start_load_kv / get_finished receive branches, _receive_kv pending decrement,
    _build_transfer_params empty-group skip, shutdown sender join,
    _sender_thread generic-exception branch.

No GCU, no FlagCX C library, no real ZMQ, no spawned processes, no pytest run.
All communication primitives are mocked. Only line coverage is intended.
"""

import asyncio
import importlib
import sys
import threading
import types
from types import SimpleNamespace
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Import bootstrap (same approach as the first file): patch the renamed vLLM
# helper and temporarily install a minimal flagcx_wrapper stub.
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
# Shared helpers
# ---------------------------------------------------------------------------
_DEV = object()


class _Storage:
    def __init__(self, addr, nbytes):
        self._addr = addr
        self._nbytes = nbytes

    def data_ptr(self):
        return self._addr

    def nbytes(self):
        return self._nbytes


class _Cache:
    def __init__(self, device, base, stride0, element_size, storage):
        self.device = device
        self._base = base
        self._stride0 = stride0
        self._element_size = element_size
        self._storage = storage

    def data_ptr(self):
        return self._base

    def stride(self, _dim):
        return self._stride0

    def element_size(self):
        return self._element_size

    def untyped_storage(self):
        return self._storage


def _meta(
    req_blocks,
    host="h",
    port=9000,
    remote_tp_rank=0,
    remote_tp_size=1,
    base=(2000,),
    block_lens=(10,),
    kv_block_lens=(10,),
    names=("l0",),
    indices=(0,),
    groups=(0,),
):
    return connector.FlagCXAgentMetadata(
        remote_hostname=host,
        remote_port=port,
        remote_tp_size=remote_tp_size,
        remote_tp_rank=remote_tp_rank,
        req_blocks=req_blocks,
        kv_caches_base_addr=list(base),
        block_lens=list(block_lens),
        kv_block_lens=list(kv_block_lens),
        registered_layer_names=list(names),
        registered_layer_indices=list(indices),
        registered_group_indices=list(groups),
    )


def _smeta(tid="t", p="p", blocks=None, ready=True):
    ev = threading.Event()
    if ready:
        ev.set()
    return connector.SendBlockMeta(
        p_req_id=p,
        transfer_id=tid,
        local_block_ids=[] if blocks is None else blocks,
        ready=ev,
    )


def _bare_worker(**over):
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
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())],
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
    w._abort_request_timeout = 5
    w._layer_specs = {}
    w._layer_group_indices = {}
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
    w.rpc_port = 0
    w.kv_cache_device = None
    w.kv_caches_base_addr = []
    w.device_kv_caches = {}
    w.block_len_per_layer = []
    w.kv_block_len_per_layer = []
    w.registered_layer_names = []
    w.registered_layer_indices = []
    w.registered_group_indices = []
    w._sender_executor = mock.MagicMock()
    w._sender_t = None
    w.receiver_loop = mock.MagicMock()
    w._receiver_t = mock.MagicMock()
    w._encoder = connector.msgspec.msgpack.Encoder()
    w._decoder = connector.msgspec.msgpack.Decoder(
        connector.FlagCXAgentMetadata
    )
    w.zmq_ctx = mock.MagicMock()
    w.async_zmq_ctx = mock.MagicMock()
    for key, value in over.items():
        setattr(w, key, value)
    return w


def _send_worker(**over):
    w = _bare_worker(
        tp_size=1,
        tp_rank=0,
        kv_caches_base_addr=[1000],
        block_len_per_layer=[10],
        kv_block_len_per_layer=[10],
        registered_layer_names=["l0"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
        ),
    )
    for key, value in over.items():
        setattr(w, key, value)
    return w


# ===========================================================================
# FlagCXConnector.__init__ and role guards
# ===========================================================================
class TestConnectorInit:
    def _cfg(self, engine_id="eng"):
        return SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id=engine_id)
        )

    def test_scheduler_role_builds_scheduler_only(self):
        cfg = self._cfg()
        with (
            mock.patch.object(
                connector.KVConnectorBase_V1, "__init__", return_value=None
            ),
            mock.patch.object(connector, "FlagCXConnectorScheduler") as sched_cls,
        ):
            inst = connector.FlagCXConnector(
                cfg, connector.KVConnectorRole.SCHEDULER, kv_cache_config="kcc"
            )
        sched_cls.assert_called_once_with(cfg, "eng", "kcc")
        assert inst.connector_scheduler is sched_cls.return_value
        assert inst.connector_worker is None
        assert inst.engine_id == "eng"

    def test_worker_role_builds_worker_only(self):
        cfg = self._cfg()
        with (
            mock.patch.object(
                connector.KVConnectorBase_V1, "__init__", return_value=None
            ),
            mock.patch.object(connector, "FlagCXConnectorWorker") as worker_cls,
        ):
            inst = connector.FlagCXConnector(
                cfg, connector.KVConnectorRole.WORKER, kv_cache_config="kcc"
            )
        worker_cls.assert_called_once_with(cfg, "eng", "kcc")
        assert inst.connector_scheduler is None
        assert inst.connector_worker is worker_cls.return_value

    def test_scheduler_role_requires_kv_cache_config(self):
        with mock.patch.object(
            connector.KVConnectorBase_V1, "__init__", return_value=None
        ):
            with pytest.raises(AssertionError):
                connector.FlagCXConnector(
                    self._cfg(),
                    connector.KVConnectorRole.SCHEDULER,
                    kv_cache_config=None,
                )

    def test_requires_kv_transfer_config(self):
        cfg = SimpleNamespace(kv_transfer_config=None)
        with mock.patch.object(
            connector.KVConnectorBase_V1, "__init__", return_value=None
        ):
            with pytest.raises(AssertionError):
                connector.FlagCXConnector(
                    cfg, connector.KVConnectorRole.WORKER, kv_cache_config="kcc"
                )

    def test_requires_engine_id(self):
        cfg = self._cfg(engine_id=None)
        with mock.patch.object(
            connector.KVConnectorBase_V1, "__init__", return_value=None
        ):
            with pytest.raises(AssertionError):
                connector.FlagCXConnector(
                    cfg, connector.KVConnectorRole.WORKER, kv_cache_config="kcc"
                )


# ===========================================================================
# FlagCXConnectorScheduler.__init__
# ===========================================================================
class FakeFull:
    pass


class FakeSW:
    def __init__(self, sliding_window, block_size=16):
        self.sliding_window = sliding_window
        self.block_size = block_size


def _scheduler_cfg(kv_role="kv_both", disable_hybrid=True):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        kv_transfer_config=SimpleNamespace(kv_role=kv_role),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hybrid
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0, tensor_parallel_size=1
        ),
    )


class TestSchedulerInit:
    def test_init_homogeneous_no_hma(self, monkeypatch):
        monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)
        cfg = _scheduler_cfg(disable_hybrid=True)
        kcc = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=FakeFull())],
            has_mamba_layers=False,
        )
        with (
            mock.patch.object(connector, "get_ip", return_value="10.0.0.1"),
            mock.patch.object(connector, "FullAttentionSpec", FakeFull),
            mock.patch.object(connector, "SlidingWindowSpec", FakeSW),
        ):
            s = connector.FlagCXConnectorScheduler(cfg, "eng", kcc)
        assert s.block_size == 16
        assert s.engine_id == "eng"
        assert s.side_channel_host == "10.0.0.1"
        assert s.side_channel_port == 8998
        assert s.kv_role == "kv_both"
        assert s._is_hma_required is False
        assert s._has_mamba is False
        assert s.blocks_per_sw == [0]
        assert s._reqs_need_recv == {}
        assert s._reqs_need_send == {}

    def test_init_heterogeneous_hma_and_sliding_window(self, monkeypatch):
        monkeypatch.setenv("FLAGCX_BOOTSTRAP_PORT", "7000")
        cfg = _scheduler_cfg(disable_hybrid=False)
        kcc = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=FakeFull()),
                SimpleNamespace(kv_cache_spec=FakeSW(64)),
                SimpleNamespace(kv_cache_spec=FakeSW(0)),
            ],
            has_mamba_layers=True,
        )
        with (
            mock.patch.object(connector, "get_ip", return_value="10.0.0.2"),
            mock.patch.object(connector, "FullAttentionSpec", FakeFull),
            mock.patch.object(connector, "SlidingWindowSpec", FakeSW),
        ):
            s = connector.FlagCXConnectorScheduler(cfg, "eng", kcc)
        assert s.side_channel_port == 7000
        assert s._is_hma_required is True
        assert s._has_mamba is True
        # 64/16 -> cdiv == 4, +1 == 5 ; zero-token group stays 0.
        assert s.blocks_per_sw == [0, 5, 0]


# ===========================================================================
# FlagCXConnectorWorker.register_kv_caches
# ===========================================================================
class FakeMLA:
    def __init__(self, page_size_bytes=5):
        self.page_size_bytes = page_size_bytes


class FakeSWMLA:
    def __init__(self, page_size_bytes=7):
        self.page_size_bytes = page_size_bytes


class FakeMamba:
    pass


class FakeOther:
    pass


class TestRegisterKvCaches:
    def test_register_blocks_first_mla_mamba_and_dedup(self):
        w = _bare_worker(kv_role="kv_both", side_channel_port=8998, tp_rank=0)
        w.kv_cache_device = None
        w._layer_specs = {
            "model.layers.0.self_attn": FakeMLA(page_size_bytes=5),
            "model.layers.1.self_attn": FakeOther(),
            "model.layers.2.self_attn": FakeMamba(),
        }
        w._layer_group_indices = {
            "model.layers.0.self_attn": 0,
            "model.layers.1.self_attn": 1,
            "model.layers.2.self_attn": 2,
        }
        w.kv_topo = SimpleNamespace(
            local_replicates_kv_cache=False,
            is_kv_layout_blocks_first=True,
            get_transfer_cache_regions=lambda caches, spec: caches,
            handshake_target_ranks=lambda size: [0],
        )

        c_a = _Cache(_DEV, 1000, 10, 1, _Storage(9000, 800))
        c_k = _Cache(_DEV, 2000, 16, 1, _Storage(9001, 400))
        c_v = _Cache(_DEV, 2016, 16, 1, _Storage(9001, 400))  # same MR as K
        c_m = _Cache(_DEV, 3000, 12, 1, _Storage(9000, 800))  # same MR as A
        kv_caches = {
            "model.layers.0.self_attn": [c_a],
            "model.layers.1.self_attn": [c_k, c_v],
            "model.layers.2.self_attn": [c_m],
            "model.layers.3.self_attn": [c_a],  # no spec -> skipped
        }

        w.flagcx.flagcxP2pGetRpcPort.return_value = 5050
        fake_thread = mock.MagicMock()
        fake_event = mock.MagicMock()
        with (
            mock.patch.multiple(
                connector,
                MLAAttentionSpec=FakeMLA,
                SlidingWindowMLASpec=FakeSWMLA,
                MambaSpec=FakeMamba,
            ),
            mock.patch.object(
                connector,
                "extract_layer_index",
                side_effect=lambda n: int(n.split(".")[2]),
            ),
            mock.patch.object(
                connector.threading, "Thread", return_value=fake_thread
            ) as thread_cls,
            mock.patch.object(
                connector.threading, "Event", return_value=fake_event
            ),
        ):
            w.register_kv_caches(kv_caches)

        assert w.kv_caches_base_addr == [1000, 2000, 2016, 3000]
        assert w.block_len_per_layer == [10, 16, 16, 12]
        # MLA -> page_size_bytes; blocks-first non-mamba -> half; mamba -> full
        assert w.kv_block_len_per_layer == [5, 8, 8, 12]
        assert w.registered_layer_names == [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
            "model.layers.1.self_attn",
            "model.layers.2.self_attn",
        ]
        assert w.registered_layer_indices == [0, 1, 1, 2]
        assert w.registered_group_indices == [0, 1, 1, 2]
        assert w.device_kv_caches is kv_caches
        assert w.rpc_port == 5050
        assert w.kv_cache_device is _DEV

        # Deduplicated memory registrations: (9000, 800) and (9001, 400).
        reg_calls = w.flagcx.flagcxP2pRegister.call_args_list
        assert [c.args[1:] for c in reg_calls] == [(9000, 800), (9001, 400)]
        # kv_both: start RPC server, then launch sender thread and wait ready.
        w.flagcx.flagcxP2pStartRpcServer.assert_called_once_with(w.engine)
        thread_cls.assert_called_once()
        fake_thread.start.assert_called_once()
        fake_event.wait.assert_called_once()

    def test_register_consumer_non_blocks_first_uses_full_block_len(self):
        w = _bare_worker(kv_role="kv_consumer", tp_rank=0)
        w._layer_specs = {"model.layers.0.self_attn": FakeOther()}
        w._layer_group_indices = {"model.layers.0.self_attn": 0}
        w.kv_topo = SimpleNamespace(
            local_replicates_kv_cache=False,
            is_kv_layout_blocks_first=False,
            get_transfer_cache_regions=lambda caches, spec: caches,
            handshake_target_ranks=lambda size: [0],
        )
        cache = _Cache(_DEV, 4000, 20, 1, _Storage(9100, 120))
        w.flagcx.flagcxP2pGetRpcPort.return_value = 6060
        with (
            mock.patch.multiple(
                connector,
                MLAAttentionSpec=FakeMLA,
                SlidingWindowMLASpec=FakeSWMLA,
                MambaSpec=FakeMamba,
            ),
            mock.patch.object(
                connector,
                "extract_layer_index",
                side_effect=lambda n: int(n.split(".")[2]),
            ),
            mock.patch.object(
                connector.threading, "Thread", return_value=mock.MagicMock()
            ) as thread_cls,
        ):
            w.register_kv_caches({"model.layers.0.self_attn": [cache]})

        assert w.kv_block_len_per_layer == [20]  # else branch: full block len
        assert w.kv_caches_base_addr == [4000]
        assert w.rpc_port == 6060
        # Decode/consumer starts the RPC accept daemon but never a sender thread.
        w.flagcx.flagcxP2pStartRpcServer.assert_called_once_with(w.engine)
        thread_cls.assert_not_called()

    def test_register_raises_when_nothing_registered(self):
        w = _bare_worker()
        w._layer_specs = {}
        w.kv_topo = SimpleNamespace(
            local_replicates_kv_cache=False,
            is_kv_layout_blocks_first=False,
            get_transfer_cache_regions=lambda caches, spec: caches,
            handshake_target_ranks=lambda size: [0],
        )
        cache = _Cache(_DEV, 5000, 10, 1, _Storage(9200, 10))
        with mock.patch.object(
            connector, "extract_layer_index", side_effect=lambda n: 0
        ):
            with pytest.raises(RuntimeError):
                w.register_kv_caches({"unregistered.layer": [cache]})
        w.flagcx.flagcxP2pRegister.assert_not_called()


# ===========================================================================
# FlagCXConnectorWorker._send_kv_to_decode
# ===========================================================================
class TestSendKvToDecode:
    def _prepared(self, blocks=None, tid="t", p="p", ready=True, **over):
        w = _send_worker(**over)
        sm = _smeta(
            tid=tid,
            p=p,
            blocks=[[1, 2]] if blocks is None else blocks,
            ready=ready,
        )
        w.reqs_need_send.reqs[tid] = sm
        return w

    def test_success_records_transfer_and_finishes(self):
        w = self._prepared()
        w._send_kv_to_decode(_meta({"d": ("t", [[1, 2]])}))

        write = w.flagcx.flagcxP2pBatchWriteSync
        write.assert_called_once()
        _, src, dst, sizes = write.call_args.args
        assert sizes == [20]
        assert src == [1000 + 1 * 10]
        assert dst == [2000 + 1 * 10]
        assert "t" not in w.reqs_need_send.reqs
        assert w.finished_sending_reqs.set == {"p"}
        assert w.xfer_stats.data["num_descriptors"] == [1]
        assert w.session_conns  # connection memoized

    def test_invalid_target_rank_warns_but_proceeds(self):
        w = self._prepared(blocks=[[1]])
        meta = _meta({"d": ("t", [[1]])}, remote_tp_rank=1, remote_tp_size=1)
        w._send_kv_to_decode(meta)
        w.flagcx.flagcxP2pBatchWriteSync.assert_called_once()

    def test_missing_send_meta_times_out(self):
        w = _send_worker()
        w._abort_request_timeout = 0.01
        with pytest.raises(RuntimeError):
            w._send_kv_to_decode(_meta({"d": ("newt", [[1]])}))
        # Entry was created before the timeout.
        assert "newt" in w.reqs_need_send.reqs

    def test_existing_unready_send_meta_times_out(self):
        w = self._prepared(blocks=[[1]], ready=False)
        w._abort_request_timeout = 0.01
        with pytest.raises(RuntimeError):
            w._send_kv_to_decode(_meta({"d": ("t", [[1]])}))
        w.flagcx.flagcxP2pBatchWriteSync.assert_not_called()

    def test_region_alignment_error_raises(self):
        w = self._prepared(blocks=[[1]])
        meta = _meta({"d": ("t", [[1]])}, names=("other",))
        with pytest.raises(RuntimeError):
            w._send_kv_to_decode(meta)
        w.flagcx.flagcxP2pBatchWriteSync.assert_not_called()

    def test_region_length_validation_error_raises(self):
        w = self._prepared(blocks=[[1]])
        meta = _meta({"d": ("t", [[1]])}, kv_block_lens=(20,))
        with pytest.raises(RuntimeError):
            w._send_kv_to_decode(meta)
        w.flagcx.flagcxP2pBatchWriteSync.assert_not_called()
        # Setup-side failures don't count as failed transfers.
        assert w.xfer_stats.data["num_failed_transfers"] == []

    def test_write_failure_records_failed_transfer(self):
        w = self._prepared(blocks=[[1]])
        w.flagcx.flagcxP2pBatchWriteSync.side_effect = RuntimeError("write")
        with pytest.raises(RuntimeError):
            w._send_kv_to_decode(_meta({"d": ("t", [[1]])}))
        assert w.xfer_stats.data["num_failed_transfers"] == [1]

    def test_empty_plan_skips_write_but_finishes(self):
        w = self._prepared(blocks=[[1]])
        w._send_kv_to_decode(_meta({"d": ("t", [[]])}))
        w.flagcx.flagcxP2pBatchWriteSync.assert_not_called()
        assert w.finished_sending_reqs.set == {"p"}
        assert w.reqs_need_send.reqs == {}

    def test_partial_send_keeps_request_until_all_targets(self):
        w = self._prepared(blocks=[[1, 2]])
        w.kv_topo = SimpleNamespace(
            local_replicates_kv_cache=False,
            is_kv_layout_blocks_first=False,
            handshake_target_ranks=lambda size: [0, 1],
        )
        meta = _meta(
            {"d": ("t", [[1, 2]])},
            remote_tp_rank=0,
            remote_tp_size=2,
            kv_block_lens=(5,),
        )
        w._send_kv_to_decode(meta)
        w.flagcx.flagcxP2pBatchWriteSync.assert_called_once()
        assert "t" in w.reqs_need_send.reqs
        assert w.reqs_need_send.reqs["t"].sent == 1
        assert w.finished_sending_reqs.set == set()


# ===========================================================================
# Small remaining branches
# ===========================================================================
class TestRemainingBranches:
    def test_start_load_kv_receive_branch(self):
        w = _bare_worker(kv_role="kv_consumer")
        md = connector.FlagCXConnectorMetadata()
        md.reqs_to_recv["r"] = connector.RecvReqMeta(
            local_block_ids=[[1]],
            remote_host="h",
            remote_port=9,
            transfer_id="t",
            remote_tp_size=1,
        )
        with mock.patch.object(
            connector.asyncio, "run_coroutine_threadsafe"
        ) as rcf:
            w.start_load_kv(md)
        rcf.assert_called_once()
        rcf.call_args.args[0].close()

    def test_get_finished_receive_branch(self):
        w = _bare_worker(kv_role="kv_consumer")
        fut = mock.MagicMock()
        fut.result.return_value = {"r1"}
        with mock.patch.object(
            connector.asyncio, "run_coroutine_threadsafe", return_value=fut
        ) as rcf:
            sending, recving = w.get_finished()
        assert sending is None
        assert recving == {"r1"}
        rcf.call_args.args[0].close()

    def test_receive_kv_pull_pending_decrement(self):
        w = _bare_worker()
        w._pull_pending = {"r1": 2}
        sock = _AsyncSock(recv_value=connector.TRANS_DONE)
        with mock.patch.object(connector, "make_zmq_socket", return_value=sock):
            asyncio.run(w._receive_kv("tcp://h:1", {"r1": ("t", [[0]])}))
        assert w._pull_pending["r1"] == 1
        assert w.finished_recving_reqs.set == set()
        assert sock.closed is True

    def test_build_transfer_params_skips_empty_group(self):
        w = _bare_worker()
        w.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=object()),
                SimpleNamespace(kv_cache_spec=object()),
            ]
        )
        r0 = connector.TransferRegion(
            layer_name="l0",
            layer_index=0,
            base_addr=1000,
            block_len=10,
            kv_block_len=10,
            group_index=0,
        )
        r1 = connector.TransferRegion(
            layer_name="l1",
            layer_index=1,
            base_addr=1000,
            block_len=10,
            kv_block_len=10,
            group_index=1,
        )
        sm = _smeta(blocks=[[1], []])
        meta = _meta(
            {"d": ("t", [[1], []])},
            base=(2000, 2000),
            block_lens=(10, 10),
            kv_block_lens=(10, 10),
            names=("l0", "l1"),
            indices=(0, 1),
            groups=(0, 1),
        )
        src, dst, sizes = w._build_transfer_params(
            [("d", sm)], meta, [r0, r1], [r0, r1]
        )
        assert sizes == [10]
        assert src == [1000 + 1 * 10]
        assert dst == [2000 + 1 * 10]

    def test_shutdown_joins_sender_thread(self):
        w = _bare_worker(kv_role="kv_producer")
        w._sender_executor = mock.MagicMock()
        w._sender_t = mock.MagicMock()
        w.engine = None
        w.shutdown()
        w._sender_executor.shutdown.assert_called_once_with(wait=False)
        w._sender_t.join.assert_called_once_with(timeout=2)

    def test_sender_thread_generic_exception_closes_sockets(self):
        w = _bare_worker()
        w._sender_executor = mock.MagicMock()
        frontend = _Msock()
        backend = _Msock()

        class FakePoller:
            def register(self, sock, event):
                pass

            def poll(self, timeout=None):
                raise RuntimeError("poll boom")

        with (
            mock.patch.object(
                connector, "make_zmq_socket", side_effect=[frontend, backend]
            ),
            mock.patch.object(connector.zmq, "Poller", lambda: FakePoller()),
        ):
            w._sender_thread(mock.MagicMock(), 9000, 0)
        assert frontend.closed is True
        assert backend.closed is True


class _AsyncSock:
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


class _Msock:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send_multipart(self, parts):
        self.sent.append(parts)

    def close(self):
        self.closed = True


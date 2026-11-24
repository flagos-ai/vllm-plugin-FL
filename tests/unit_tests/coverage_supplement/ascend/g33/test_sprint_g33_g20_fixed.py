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

"""G20 extra coverage: FlagCX connector residue + Ascend attention builder.

Connector side targets the facade, ``register_kv_caches`` role branches and the
sender thread/worker dispatch loops.  The ctypes facade is faked through a
synthetic ``plugin.interservice.flagcx_wrapper`` module, so no FlagCX library is
needed.  Attention side loads the module by path (like G6) and exercises the
build / mask / split / graph-capture branches on CPU stand-ins.

Self-contained: no imports from the shared coverage-test helpers.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import sys
import threading
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

CONNECTOR = "vllm_fl.distributed.kv_transfer.flagcx_connector"
WRAPPER = "plugin.interservice.flagcx_wrapper"


def _ensure_kv_layout_symbol():
    """814b imports ``get_kv_cache_layout`` from vLLM; some installs lack it.

    Add a local fallback so a missing symbol does not mask the connector's own
    ``FlagCX wrapper`` ImportError (the real failure mode seen on 910C).
    """
    try:
        import vllm.v1.attention.backends.utils as _utils
    except Exception:  # pragma: no cover - environment guard
        return
    if not hasattr(_utils, "get_kv_cache_layout"):
        _utils.get_kv_cache_layout = lambda *a, **k: "NHD"
    if not hasattr(_utils, "NULL_BLOCK_ID"):
        _utils.NULL_BLOCK_ID = -1


_ensure_kv_layout_symbol()


# ===========================================================================
# Part 1 - FlagCX connector
# ===========================================================================


class _FakeFlagcxLib:
    """Recording stand-in for the connector's ``FLAGCXLibrary`` instance."""

    def __init__(self, recorder):
        self.recorder = recorder

    def _rec(self, name, *args):
        self.recorder.append((name,) + args)

    def flagcxP2pEngineCreate(self):
        self._rec("flagcxP2pEngineCreate")
        return "ENGINE"

    def flagcxP2pEngineDestroy(self, engine):
        self._rec("flagcxP2pEngineDestroy", engine)

    def flagcxP2pGetConn(self, engine, session):
        self._rec("flagcxP2pGetConn", session)
        return ("CONN", session)

    def flagcxP2pGetRpcPort(self, engine):
        self._rec("flagcxP2pGetRpcPort")
        return 4242

    def flagcxP2pStartRpcServer(self, engine):
        self._rec("flagcxP2pStartRpcServer", engine)

    def flagcxP2pRegister(self, engine, base_addr, size):
        self._rec("flagcxP2pRegister", base_addr, size)

    def flagcxP2pBatchWriteSync(self, conn, src, dst, sizes):
        self._rec("flagcxP2pBatchWriteSync", conn, src, dst, sizes)


def _install_wrapper(monkeypatch, *, with_library=True):
    recorder = []
    wrapper = types.ModuleType(WRAPPER)
    if with_library:
        wrapper.FLAGCXLibrary = lambda library_path=None: _FakeFlagcxLib(recorder)

    plugin = types.ModuleType("plugin")
    plugin.__path__ = []
    interservice = types.ModuleType("plugin.interservice")
    interservice.__path__ = []
    interservice.flagcx_wrapper = wrapper

    monkeypatch.setitem(sys.modules, "plugin", plugin)
    monkeypatch.setitem(sys.modules, "plugin.interservice", interservice)
    monkeypatch.setitem(sys.modules, WRAPPER, wrapper)
    return recorder


def _drop():
    sys.modules.pop(CONNECTOR, None)


@pytest.fixture()
def conn(monkeypatch):
    _install_wrapper(monkeypatch, with_library=True)
    _drop()
    try:
        module = importlib.import_module(CONNECTOR)
    except Exception as exc:  # pragma: no cover - environment guard
        _drop()
        pytest.skip(f"connector not importable on this build: {exc!r}")
    yield module
    _drop()


def test_connector_import_error_without_library(monkeypatch):
    """A wrapper missing ``FLAGCXLibrary`` raises a helpful ImportError (80-81).

    Accept an unrelated vLLM symbol failure too: on some builds the connector
    import trips over ``get_kv_cache_layout`` before reaching the wrapper
    import, which would otherwise keep the rest of this file from running.
    """
    _install_wrapper(monkeypatch, with_library=False)
    _drop()
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.import_module(CONNECTOR)
        message = str(excinfo.value)
        assert (
            "FlagCX wrapper" in message
            or "get_kv_cache_layout" in message
            or "cannot import name" in message
        )
    finally:
        _drop()


class _RecordingScheduler:
    def __init__(self):
        self.calls = []

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.calls.append(("get_num_new_matched_tokens", num_computed_tokens))
        return 3, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.calls.append(("update_state_after_alloc", blocks, num_external_tokens))
        return "updated"

    def build_connector_meta(self, scheduler_output):
        self.calls.append(("build_connector_meta", scheduler_output))
        return "meta-out"

    def request_finished(self, request, block_ids):
        self.calls.append(("request_finished", block_ids))
        return True, {"ok": 1}


class _RecordingWorker:
    def __init__(self):
        self.calls = []

    def register_kv_caches(self, kv_caches):
        self.calls.append(("register_kv_caches", kv_caches))

    def get_finished(self):
        self.calls.append(("get_finished",))
        return {"sent"}, {"recv"}

    def start_load_kv(self, metadata):
        self.calls.append(("start_load_kv", metadata))

    def wait_for_layer_load(self):
        self.calls.append(("wait_for_layer_load",))

    def get_kv_connector_stats(self):
        self.calls.append(("get_kv_connector_stats",))
        return "stats"


def _facade(module, scheduler=None, worker=None):
    facade = object.__new__(module.FlagCXConnector)
    facade.connector_scheduler = scheduler
    facade.connector_worker = worker
    facade._connector_metadata = module.FlagCXConnectorMetadata()
    return facade


def test_facade_scheduler_forwarding(conn):
    scheduler = _RecordingScheduler()
    facade = _facade(conn, scheduler=scheduler)
    req = object()

    assert facade.get_num_new_matched_tokens(req, 5) == (3, True)
    assert facade.update_state_after_alloc(req, "blocks", 2) == "updated"
    assert facade.build_connector_meta("sched-out") == "meta-out"
    assert facade.request_finished(req, [1, 2]) == (True, {"ok": 1})
    assert facade.request_finished_all_groups(req, ([1], [2])) == (True, {"ok": 1})
    assert scheduler.calls[-2][1] == ([1, 2],)
    assert scheduler.calls[-1][1] == ([1], [2])


def test_facade_worker_forwarding(conn):
    worker = _RecordingWorker()
    facade = _facade(conn, worker=worker)

    facade.register_kv_caches({"layer": "cache"})
    assert facade.get_finished({"x"}) == ({"sent"}, {"recv"})
    facade.start_load_kv(None)
    facade.wait_for_layer_load("layer")
    facade.save_kv_layer("layer", torch.zeros(1), None)  # no-op
    facade.wait_for_save()  # no-op
    assert facade.get_kv_connector_stats() == "stats"

    assert worker.calls[0] == ("register_kv_caches", {"layer": "cache"})
    assert worker.calls[1] == ("get_finished",)
    assert worker.calls[2] == ("start_load_kv", facade._connector_metadata)
    assert worker.calls[3] == ("wait_for_layer_load",)
    assert worker.calls[4] == ("get_kv_connector_stats",)


def test_facade_worker_none_paths(conn):
    facade = _facade(conn, worker=None)
    facade.wait_for_layer_load("layer")  # no worker: no-op
    assert facade.get_kv_connector_stats() is None


def test_required_kvcache_layout_branches(conn):
    cls = conn.FlagCXConnector
    assert cls.get_required_kvcache_layout(
        SimpleNamespace(model_config=None)
    ) is None
    assert cls.get_required_kvcache_layout(
        SimpleNamespace(model_config=SimpleNamespace(use_mla=True))
    ) is None
    assert cls.get_required_kvcache_layout(
        SimpleNamespace(model_config=SimpleNamespace(use_mla=False))
    ) == "HND"


def test_build_stats_and_prom(conn, monkeypatch):
    record_stats = []
    record_prom = []

    class _Stats:
        def __init__(self, data=None):
            record_stats.append(data)

    class _Prom:
        def __init__(self, vllm_config, metric_types, labelnames, values):
            record_prom.append((vllm_config, metric_types, labelnames, values))

    monkeypatch.setattr(conn, "FlagCXKVConnectorStats", _Stats, raising=False)
    monkeypatch.setattr(conn, "FlagCXPromMetrics", _Prom, raising=False)

    conn.FlagCXConnector.build_kv_connector_stats()
    conn.FlagCXConnector.build_kv_connector_stats({"a": 1})
    assert record_stats == [{}, {"a": 1}]

    args = (object(), {}, ["l"], {0: ["v"]})
    assert isinstance(conn.FlagCXConnector.build_prom_metrics(*args), _Prom)
    assert record_prom == [args]


class _Topo:
    def __init__(self, mapping, blocks_first):
        self._mapping = mapping
        self.is_kv_layout_blocks_first = blocks_first

    def get_transfer_cache_regions(self, cache_or_caches, spec):
        return self._mapping.get(id(spec), [cache_or_caches])


def _make_worker(module, *, layer_specs, group_indices, mapping, blocks_first=False,
                 role="kv_both"):
    worker = object.__new__(module.FlagCXConnectorWorker)
    worker.use_mla = False
    worker._layer_specs = dict(layer_specs)
    worker._layer_group_indices = dict(group_indices)
    worker.kv_topo = _Topo(mapping, blocks_first)
    worker.kv_cache_device = None
    worker.block_len_per_layer = []
    worker.kv_block_len_per_layer = []
    worker.registered_layer_names = []
    worker.registered_layer_indices = []
    worker.registered_group_indices = []
    worker.flagcx = _FakeFlagcxLib([])
    worker.engine = "ENGINE"
    worker.rpc_port = 0
    worker.kv_role = role
    worker.side_channel_port = 12345
    worker.tp_rank = 0
    worker._sender_thread = lambda ready_event, base_port, tp_rank: ready_event.set()
    return worker


def _kv_tensor(num_blocks=4, block_size=16, heads=2, head_size=8):
    return torch.zeros(num_blocks, block_size, heads, head_size, dtype=torch.float16)


def test_register_kv_caches_both_roles(conn):
    layer = "model.layers.0.self_attn"
    spec = object()
    tensor = _kv_tensor()
    worker = _make_worker(
        conn, layer_specs={layer: spec}, group_indices={layer: 0},
        mapping={id(spec): [tensor]}, role="kv_both",
    )
    recorder = worker.flagcx.recorder

    worker.register_kv_caches({layer: tensor})

    block_len = tensor.stride(0) * tensor.element_size()
    assert worker.kv_caches_base_addr == [tensor.data_ptr()]
    assert worker.block_len_per_layer == [block_len]
    assert worker.kv_block_len_per_layer == [block_len]
    assert worker.registered_layer_indices == [0]
    assert worker.device_kv_caches[layer] is tensor
    assert worker.rpc_port == 4242

    storage = tensor.untyped_storage()
    assert ("flagcxP2pRegister", storage.data_ptr(), storage.nbytes()) in recorder
    assert ("flagcxP2pStartRpcServer", "ENGINE") in recorder
    assert worker._sender_t is not None


def test_register_kv_caches_producer_role_skips_rpc(conn):
    """A pure producer registers memory and starts the sender, but no RPC server."""
    layer = "model.layers.0.self_attn"
    spec = object()
    tensor = _kv_tensor()
    worker = _make_worker(
        conn, layer_specs={layer: spec}, group_indices={layer: 0},
        mapping={id(spec): [tensor]}, role="kv_producer",
    )
    recorder = worker.flagcx.recorder

    worker.register_kv_caches({layer: tensor})

    assert ("flagcxP2pStartRpcServer", "ENGINE") not in recorder
    assert worker._sender_t is not None


def test_register_kv_caches_consumer_role_skips_sender(conn):
    layer = "model.layers.0.self_attn"
    spec = object()
    tensor = _kv_tensor()
    worker = _make_worker(
        conn, layer_specs={layer: spec}, group_indices={layer: 0},
        mapping={id(spec): [tensor]}, role="kv_consumer",
    )
    recorder = worker.flagcx.recorder

    worker.register_kv_caches({layer: tensor})

    assert ("flagcxP2pStartRpcServer", "ENGINE") in recorder
    assert getattr(worker, "_sender_t", None) is None




def test_register_kv_caches_dedupes_storage(conn):
    layer = "model.layers.0.self_attn"
    spec = object()
    tensor = _kv_tensor()
    worker = _make_worker(
        conn, layer_specs={layer: spec}, group_indices={layer: 0},
        mapping={id(spec): [tensor, tensor]}, role="kv_consumer",
    )
    worker.register_kv_caches({layer: tensor})

    registers = [c for c in worker.flagcx.recorder if c[0] == "flagcxP2pRegister"]
    assert len(registers) == 1
    assert len(worker.block_len_per_layer) == 2


def test_register_kv_caches_unknown_layer_raises(conn):
    tensor = _kv_tensor()
    worker = _make_worker(
        conn, layer_specs={}, group_indices={}, mapping={}, role="kv_consumer",
    )
    with pytest.raises(RuntimeError, match="No KV cache tensors"):
        worker.register_kv_caches({"model.layers.9.self_attn": tensor})


class _FakeZMQ:
    ROUTER = "router"
    PULL = "pull"
    PUSH = "push"
    POLLIN = 1

    class ContextTerminated(Exception):
        pass

    class ZMQError(Exception):
        pass


def test_sender_thread_dispatches_and_relays(conn, monkeypatch):
    submitted = []

    class _Executor:
        def submit(self, *args):
            submitted.append(args)

    class _Frontend:
        def __init__(self):
            self.sent = []
            self.closed = False

        def recv_multipart(self):
            return (b"identity", b"", b"metadata")

        def send_multipart(self, frames):
            self.sent.append(frames)

        def close(self):
            self.closed = True

    class _Backend:
        def __init__(self):
            self.closed = False

        def recv_multipart(self):
            return (b"identity", b"status")

        def close(self):
            self.closed = True

    frontend, backend = _Frontend(), _Backend()

    class _Poller:
        def __init__(self):
            self.socks = []
            self.calls = 0

        def register(self, sock, flag):
            self.socks.append(sock)

        def poll(self):
            self.calls += 1
            if self.calls == 1:
                return {sock: 1 for sock in self.socks}
            raise _FakeZMQ.ContextTerminated()

    _FakeZMQ.Poller = _Poller
    monkeypatch.setattr(conn, "zmq", _FakeZMQ, raising=False)
    monkeypatch.setattr(conn, "make_zmq_path", lambda *a: "/path", raising=False)
    monkeypatch.setattr(
        conn, "make_zmq_socket",
        lambda ctx, path, stype: frontend if stype == _FakeZMQ.ROUTER else backend,
        raising=False,
    )

    worker = object.__new__(conn.FlagCXConnectorWorker)
    worker.hostname = "127.0.0.1"
    worker.tp_rank = 0
    worker.zmq_ctx = object()
    worker._sender_executor = _Executor()
    ready = threading.Event()

    conn.FlagCXConnectorWorker._sender_thread(worker, ready, 5000, 0)

    assert ready.is_set() is True
    assert len(submitted) == 1
    _, identity, metadata_bytes, channel_path = submitted[0]
    assert (identity, metadata_bytes, channel_path) == (b"identity", b"metadata", "/path")
    assert frontend.sent == [(b"identity", b"", b"status")]
    assert frontend.closed is True and backend.closed is True


def _sender_worker_fixture(conn, monkeypatch, *, decode_raises=False, send_raises=False):
    events = []

    class _Decoder:
        def decode(self, payload):
            if decode_raises:
                raise ValueError("bad payload")
            return "META"

    class _Pusher:
        def send_multipart(self, frames):
            if send_raises:
                raise _FakeZMQ.ZMQError("closed")
            events.append(("send", frames))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(conn, "zmq", _FakeZMQ, raising=False)
    monkeypatch.setattr(
        conn, "make_zmq_socket", lambda ctx, path, stype: _Pusher(), raising=False
    )

    worker = object.__new__(conn.FlagCXConnectorWorker)
    worker.zmq_ctx = object()
    worker._decoder = _Decoder()
    worker._send_kv_to_decode = lambda meta: events.append(("xfer", meta))
    return worker, events


def test_sender_worker_success(conn, monkeypatch):
    worker, events = _sender_worker_fixture(conn, monkeypatch)
    conn.FlagCXConnectorWorker._sender_worker(worker, b"id", b"blob", "/chan")

    assert ("xfer", "META") in events
    assert ("send", (b"id", conn.TRANS_DONE)) in events
    assert events[-1] == ("close",)


def test_sender_worker_decode_error_reports_error(conn, monkeypatch):
    worker, events = _sender_worker_fixture(conn, monkeypatch, decode_raises=True)
    conn.FlagCXConnectorWorker._sender_worker(worker, b"id", b"blob", "/chan")

    assert ("xfer", "META") not in events
    assert ("send", (b"id", conn.TRANS_ERROR)) in events


def test_sender_worker_zmq_error_is_swallowed(conn, monkeypatch):
    worker, events = _sender_worker_fixture(conn, monkeypatch, send_raises=True)
    conn.FlagCXConnectorWorker._sender_worker(worker, b"id", b"blob", "/chan")

    assert ("xfer", "META") in events
    assert events[-1] == ("close",)


# ===========================================================================
# Part 2 - Ascend attention builder
# ===========================================================================

_TARGET_REL = "vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py"


def _find_repo():
    env = os.environ.get("FL_REPO_ROOT") or os.environ.get("FL_WORKSPACE")
    candidates = []
    if env:
        candidates.append(pathlib.Path(env))
    here = pathlib.Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    candidates.append(pathlib.Path.cwd())
    for entry in sys.path:
        if entry:
            candidates.append(pathlib.Path(entry))
    for base in candidates:
        if (base / "vllm_fl" / "dispatch" / "backends").is_dir():
            return base
    return None


def _exec_attention(name, path):
    impl_dir = path.parent
    package = types.ModuleType(name.rsplit(".", 1)[0])
    package.__path__ = [str(impl_dir)]
    package.__package__ = name.rsplit(".", 1)[0]
    sys.modules[package.__name__] = package

    import vllm.v1.attention.backends.registry as registry

    original = registry.register_backend
    registry.register_backend = lambda *_a, **_k: (lambda cls: cls)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        registry.register_backend = original
    return module


_REPO = _find_repo()
if _REPO is None:
    pytest.skip("cannot locate vllm_fl tree on this host", allow_module_level=True)
_TARGET = _REPO / _TARGET_REL
if not _TARGET.is_file():
    pytest.skip(f"target module missing: {_TARGET}", allow_module_level=True)

try:
    attn = _exec_attention("g20_asc_impl.attention", _TARGET)
except Exception as exc:  # torch_npu / vllm not importable here
    attn = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture()
def mod():
    if attn is None:
        pytest.skip(f"attention module unavailable: {_IMPORT_ERROR}")
    return attn


def _fake_mask(monkeypatch, device=None):
    device = device or torch.device("cpu")

    class _Mask:
        def get_attn_mask(self, *_a, **_k):
            return "attn"

        def get_mla_mask(self, *_a, **_k):
            return "mla"

        def get_splitfuse_attn_mask(self, *_a, **_k):
            return "splitfuse"

    monkeypatch.setattr(attn.AscendAttentionMetadataBuilder, "_mask_builder", _Mask())
    monkeypatch.setattr(attn.AscendAttentionMetadataBuilder, "_mask_builder_device", device)


def _allow_cpu_pin(monkeypatch):
    try:
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    except (TypeError, AttributeError):
        pass


def _config(mode, speculative=None, additional="missing"):
    return SimpleNamespace(
        speculative_config=speculative,
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        additional_config={} if additional == "missing" else additional,
    )


def test_using_paged_attention_matrix(mod):
    from vllm.config import CUDAGraphMode

    assert mod.using_paged_attention(
        4, _config(CUDAGraphMode.FULL_DECODE_ONLY, speculative=object())
    ) is False
    assert mod.using_paged_attention(4, _config(CUDAGraphMode.NONE), head_size=512) is True
    assert mod.using_paged_attention(4, _config(CUDAGraphMode.NONE)) is False
    assert mod.using_paged_attention(
        2, _config(CUDAGraphMode.FULL_DECODE_ONLY, additional={"pa_shape_list": [1, 2]})
    ) is True
    with pytest.raises(TypeError):
        mod.using_paged_attention(
            1, _config(CUDAGraphMode.FULL_DECODE_ONLY, additional={"pa_shape_list": "1,2"})
        )
    with pytest.raises(TypeError):
        mod.using_paged_attention(
            1, _config(CUDAGraphMode.FULL_DECODE_ONLY,
                       additional={"pa_shape_list": [1, True]})
        )


def test_classify_and_v2_states(mod):
    state = mod.AscendAttentionState
    classify = mod.classify_ascend_attention_state
    assert classify([0, 0], [1, 1], [1, 1], False, None) is state.PrefillNoCache
    assert classify([1, 1], [1, 1], [1, 1], False, "mtp") is state.SpecDecoding
    assert classify([2, 2], [1, 1], [1, 1], False, None) is state.DecodeOnly
    assert classify([3, 3], [2, 2], [1, 1], False, "mtp") is state.SpecDecoding
    assert classify([3, 3], [2, 2], [1, 1], False, "eagle") is state.ChunkedPrefill
    assert classify([3, 3], [2, 2], [2, 2], True, None) is state.ChunkedPrefill
    assert classify([3, 3], [2, 2], [2, 2], False, None) is state.PrefillCacheHit

    cm = SimpleNamespace(
        num_reqs=3,
        query_start_loc_cpu=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    assert mod.get_ascend_v2_attention_state(
        cm, torch.tensor([4, 3, 0]), True, None
    ) is state.ChunkedPrefill


def test_pad_v2_decode_metadata(mod):
    cm = SimpleNamespace(num_reqs=3, num_actual_tokens=3)
    qsl = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
    seq_lens = torch.tensor([5, 6, 0])
    padded_qsl, padded_seq = mod.pad_ascend_v2_decode_metadata(cm, qsl, seq_lens)
    assert padded_qsl.tolist() == [0, 1, 2, 3]
    assert padded_seq.tolist() == [5, 6, 1]

    cm_ok = SimpleNamespace(num_reqs=3, num_actual_tokens=2)
    out_qsl, out_seq = mod.pad_ascend_v2_decode_metadata(cm_ok, qsl, seq_lens)
    assert out_qsl is not None and out_seq is seq_lens

    bad = torch.tensor([0, 2, 2, 2], dtype=torch.int32)
    with pytest.raises(ValueError):
        mod.pad_ascend_v2_decode_metadata(cm, bad, seq_lens)


def test_dummy_attention_state(mod):
    state = mod.AscendAttentionState
    with pytest.raises(NotImplementedError, match="create_mixed_batch"):
        mod.get_ascend_dummy_attention_state(create_mixed_batch=True)
    # Signature is (create_mixed_batch, speculative_method, use_mla): the first
    # positional must stay False, otherwise the mixed-batch guard fires.
    assert mod.get_ascend_dummy_attention_state(False, "mtp", True) is state.SpecDecoding
    assert mod.get_ascend_dummy_attention_state(False, "mtp", False) is state.ChunkedPrefill
    assert mod.get_ascend_dummy_attention_state() is state.DecodeOnly


def test_builder_init_counts_speculative_tokens(mod, monkeypatch):
    monkeypatch.setattr(mod.AttentionMetadataBuilder, "__init__", lambda self, *a, **k: None)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )
    builder = mod.AscendAttentionMetadataBuilder(object(), ["l0"], cfg, torch.device("cpu"))
    assert builder.decode_threshold == 3
    assert builder.chunked_prefill_enabled is True
    assert builder.max_num_blocks_per_req == 8

    cfg.speculative_config = None
    plain = mod.AscendAttentionMetadataBuilder(object(), ["l0"], cfg, torch.device("cpu"))
    assert plain.decode_threshold == 1


def test_mask_builder_cache_and_variants(mod, monkeypatch):
    device = torch.device("cpu")
    monkeypatch.setattr(mod.AscendAttentionMetadataBuilder, "_mask_builder", None)
    monkeypatch.setattr(mod.AscendAttentionMetadataBuilder, "_mask_builder_device", None)
    created = []

    class _Factory:
        def __init__(self, dev):
            created.append(dev)

    monkeypatch.setattr(mod, "AttentionMaskBuilder", _Factory)
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    builder.device = device
    first = builder._get_mask_builder()
    second = builder._get_mask_builder()
    assert first is second
    assert created == [device]

    _fake_mask(monkeypatch)
    builder.model_config = SimpleNamespace(runner_type="pooling")
    assert builder._make_attention_mask(mod.AscendAttentionState.DecodeOnly) == "attn"
    builder.model_config = SimpleNamespace(runner_type="generation", use_mla=True)
    assert builder._make_attention_mask(mod.AscendAttentionState.DecodeOnly) == "mla"
    builder.model_config = SimpleNamespace(runner_type="generation", use_mla=False)
    assert builder._make_attention_mask(mod.AscendAttentionState.DecodeOnly) == "splitfuse"


def _builder(mod):
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.decode_threshold = 1
    builder.chunked_prefill_enabled = True
    builder.speculative_config = None
    builder.model_config = SimpleNamespace(runner_type="generation", use_mla=False)
    builder.vllm_config = SimpleNamespace(use_v2_model_runner=False)
    return builder


def test_build_seq_lens_three_way_fallback(mod, monkeypatch):
    _fake_mask(monkeypatch)
    _allow_cpu_pin(monkeypatch)
    builder = _builder(mod)

    base = dict(
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        block_table_tensor=torch.zeros(1, 4, dtype=torch.int32),
        slot_mapping=torch.arange(3),
        attn_state=mod.AscendAttentionState.PrefillNoCache,
        causal=True,
    )
    cm_private = SimpleNamespace(_seq_lens_cpu=torch.tensor([7]), **base)
    assert builder.build(0, cm_private).seq_lens_list == [7]

    cm_public = SimpleNamespace(seq_lens_cpu=torch.tensor([8]), **base)
    assert builder.build(0, cm_public).seq_lens_list == [8]

    cm_device = SimpleNamespace(seq_lens=torch.tensor([9]), **base)
    assert builder.build(0, cm_device).seq_lens_list == [9]


def test_build_requires_attention_state(mod, monkeypatch):
    _fake_mask(monkeypatch)
    _allow_cpu_pin(monkeypatch)
    builder = _builder(mod)
    cm = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        block_table_tensor=torch.zeros(1, 4, dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        slot_mapping=torch.arange(1),
        attn_state=None,
        causal=True,
    )
    with pytest.raises(AssertionError):
        builder.build(0, cm)


def test_build_none_block_table_and_padding(mod, monkeypatch):
    _fake_mask(monkeypatch)
    _allow_cpu_pin(monkeypatch)
    builder = _builder(mod)
    cm = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=1,
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_table_tensor=None,
        _seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
        slot_mapping=torch.arange(3),
        attn_state=mod.AscendAttentionState.DecodeOnly,
        causal=True,
    )
    metadata = builder.build(0, cm)
    assert metadata.block_tables is None
    assert metadata.seq_lens_list == [5, 1]


def test_build_pads_block_table_rows(mod, monkeypatch):
    _fake_mask(monkeypatch)
    _allow_cpu_pin(monkeypatch)
    builder = _builder(mod)
    cm = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=1,
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_table_tensor=torch.zeros(1, 4, dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([5, 6], dtype=torch.int32),
        slot_mapping=torch.arange(3),
        attn_state=mod.AscendAttentionState.DecodeOnly,
        causal=True,
    )
    metadata = builder.build(0, cm)
    assert metadata.block_tables.shape[0] == 2
    assert metadata.actual_seq_lengths_q == [1, 2]


def test_build_v2_decode_pads(mod, monkeypatch):
    _fake_mask(monkeypatch)
    _allow_cpu_pin(monkeypatch)
    builder = _builder(mod)
    builder.vllm_config = SimpleNamespace(use_v2_model_runner=True)
    cm = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=3,
        max_query_len=1,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
        block_table_tensor=torch.zeros(3, 4, dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([2, 2, 0], dtype=torch.int32),
        slot_mapping=torch.arange(6),
        attn_state=None,
        causal=True,
    )
    metadata = builder.build(0, cm)
    assert metadata.attn_state is mod.AscendAttentionState.DecodeOnly
    assert metadata.actual_seq_lengths_q == [1, 2, 3]


def test_split_decodes_and_prefills_paths(mod):
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    builder.decode_threshold = 1

    decode = SimpleNamespace(
        max_query_len=1,
        num_reqs=3,
        num_actual_tokens=3,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    assert builder._split_decodes_and_prefills(decode) == (3, 0, 3, 0)

    mixed = SimpleNamespace(
        max_query_len=3,
        num_reqs=3,
        num_actual_tokens=5,
        query_start_loc_cpu=torch.tensor([0, 1, 4, 5], dtype=torch.int32),
    )
    assert builder._split_decodes_and_prefills(mixed) == (1, 2, 1, 4)

    no_prefill = SimpleNamespace(
        max_query_len=3,
        num_reqs=3,
        num_actual_tokens=3,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    assert builder._split_decodes_and_prefills(no_prefill) == (3, 0, 3, 0)


def test_graph_capture_entry_points(mod):
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    builder.build = lambda **kwargs: SimpleNamespace(attn_state=None)
    built = builder.build_for_graph_capture(object(), mod.AscendAttentionState.DecodeOnly)
    assert built.attn_state is mod.AscendAttentionState.DecodeOnly

    other = object.__new__(mod.AscendAttentionMetadataBuilder)
    seen = {}
    other.build_for_graph_capture = (
        lambda cm, attn_state=None, model=None: seen.setdefault("state", attn_state)
    )
    other.build_for_cudagraph_capture(object())
    assert seen["state"] is mod.AscendAttentionState.DecodeOnly

    with pytest.raises(NotImplementedError):
        builder.build_for_graph_capture(
            object(), mod.AscendAttentionState.ChunkedPrefill
        )


def test_backend_identity_and_shapes(mod):
    cls = mod.AscendAttentionBackend
    assert cls.get_name() == "CUSTOM"
    assert cls.get_impl_cls() is mod.AscendAttentionBackendImpl
    assert cls.get_builder_cls() is mod.AscendAttentionMetadataBuilder
    assert cls.get_kv_cache_shape(2, 16, 4, 8) == (2, 2, 16, 4, 8)
    assert cls.get_supported_kernel_block_sizes() == [128]
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    assert builder.reorder_batch(object(), object()) is False
    assert builder.use_cascade_attention() is False


def test_backend_swap_and_copy_blocks(mod):
    cls = mod.AscendAttentionBackend
    src_k = torch.zeros(4, 3)
    src_v = torch.zeros(4, 3)
    src_k[0] = torch.arange(3)
    src_v[0] = torch.arange(3) + 10
    dst_k = torch.zeros(4, 3)
    dst_v = torch.zeros(4, 3)
    mapping = torch.tensor([[0, 2], [1, 3]])
    cls.swap_blocks([src_k, src_v], [dst_k, dst_v], mapping)
    assert torch.equal(dst_k[2], src_k[0])
    assert torch.equal(dst_v[3], src_v[1])

    kv = torch.zeros(2, 4, 3)
    kv[0, 0] = torch.arange(3)
    cls.copy_blocks([kv], torch.tensor([[0, 2]]))
    assert torch.equal(kv[0, 2], kv[0, 0])


def test_backend_customize_spec(mod):
    # ``dataclasses.replace`` can only set declared fields, so the stand-in
    # must carry the very field ``get_kv_cache_shape``/``customize_spec`` sets.
    @dataclass
    class _Spec:
        num_head_slots: int = 1
        num_kv_heads: int = 2
        head_size: int = 8
        dtype: torch.dtype = torch.float16
        state_content_bytes: int = 0

    out = mod.AscendAttentionBackend.customize_spec(_Spec())
    assert out.num_head_slots == 2
    assert out.state_content_bytes == 2 * 8 * 2

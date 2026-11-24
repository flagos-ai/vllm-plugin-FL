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

"""814b coverage for ``kv_transfer/flagcx_connector.py``.

The 814b connector imports its ctypes facade ONLY from the source-tree
wrapper ``plugin.interservice.flagcx_wrapper`` (there is no wheel ladder and
no ``_flagcx_is_wheel``).  Every test therefore installs a fake wrapper module
in ``sys.modules`` before importing the connector; the real FlagCX library is
never needed.

The worker is built with ``object.__new__`` plus hand-set attributes, so the
vLLM config objects, process groups, ZMQ sockets and background threads are
all replaced by recorders.  Expected values (addresses, lengths, block ids)
are derived in closed form from the tensors and the fakes.

Self-contained: no imports from the shared ``coverage_tests`` helpers.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
from types import SimpleNamespace

import pytest
import torch

CONNECTOR = "vllm_fl.distributed.kv_transfer.flagcx_connector"
WRAPPER = "plugin.interservice.flagcx_wrapper"


# ---------------------------------------------------------------------------
# Fake wrapper plumbing
# ---------------------------------------------------------------------------


class _FakeFlagcxLib:
    """Recording stand-in for the connector's ``FLAGCXLibrary`` instance."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.write_should_raise = None

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
        if self.write_should_raise is not None:
            raise self.write_should_raise


def _install_wrapper(monkeypatch, *, with_library=True):
    """Install a fake ``plugin.interservice.flagcx_wrapper`` package."""
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
    """Import the connector against the fake source-tree wrapper."""
    recorder = _install_wrapper(monkeypatch, with_library=True)
    _drop()
    try:
        module = importlib.import_module(CONNECTOR)
    except Exception as exc:  # pragma: no cover - environment guard
        _drop()
        pytest.skip(f"connector not importable on this build: {exc!r}")
    yield module, recorder
    _drop()


# ---------------------------------------------------------------------------
# Import ladder failure (target 80, 81)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Facade delegation (target 247-267, 272-273, 280-301, 310, 313, 324-326, 348)
# ---------------------------------------------------------------------------


class _RecordingScheduler:
    def __init__(self):
        self.calls = []

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.calls.append(("get_num_new_matched_tokens", request, num_computed_tokens))
        return 3, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.calls.append(
            ("update_state_after_alloc", request, blocks, num_external_tokens)
        )
        return "updated"

    def build_connector_meta(self, scheduler_output):
        self.calls.append(("build_connector_meta", scheduler_output))
        return "meta-out"

    def request_finished(self, request, block_ids):
        self.calls.append(("request_finished", request, block_ids))
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


def _facade(conn, scheduler=None, worker=None):
    facade = object.__new__(conn.FlagCXConnector)
    facade.connector_scheduler = scheduler
    facade.connector_worker = worker
    facade._connector_metadata = conn.FlagCXConnectorMetadata()
    return facade


def test_facade_scheduler_delegation(conn):
    """The scheduler-side facade methods forward verbatim."""
    module, _ = conn
    scheduler = _RecordingScheduler()
    facade = _facade(module, scheduler=scheduler)
    req = object()

    assert facade.get_num_new_matched_tokens(req, 5) == (3, True)
    assert facade.update_state_after_alloc(req, "blocks", 2) == "updated"
    assert facade.build_connector_meta("sched-out") == "meta-out"
    assert facade.request_finished(req, [1, 2]) == (True, {"ok": 1})
    assert facade.request_finished_all_groups(req, ([1], [2])) == (True, {"ok": 1})
    assert [call[0] for call in scheduler.calls] == [
        "get_num_new_matched_tokens",
        "update_state_after_alloc",
        "build_connector_meta",
        "request_finished",
        "request_finished",
    ]
    assert scheduler.calls[3][2] == ([1, 2],)
    assert scheduler.calls[4][2] == ([1], [2])


def test_facade_worker_delegation(conn):
    """The worker-side facade methods forward, including the metadata assert."""
    module, _ = conn
    worker = _RecordingWorker()
    facade = _facade(module, worker=worker)

    facade.register_kv_caches({"layer": "cache"})
    assert facade.get_finished({"x"}) == ({"sent"}, {"recv"})
    facade.start_load_kv(None)
    facade.wait_for_layer_load("layer")
    facade.save_kv_layer("layer", torch.zeros(1), None)
    facade.wait_for_save()
    assert facade.get_kv_connector_stats() == "stats"

    assert worker.calls[0] == ("register_kv_caches", {"layer": "cache"})
    assert worker.calls[1] == ("get_finished",)
    assert worker.calls[2] == ("start_load_kv", facade._connector_metadata)
    assert worker.calls[3] == ("wait_for_layer_load",)
    assert worker.calls[4] == ("get_kv_connector_stats",)


def test_facade_worker_absent_paths(conn):
    """worker-None paths return None cleanly."""
    module, _ = conn
    facade = _facade(module, worker=None)
    facade.wait_for_layer_load("layer")  # no worker: no-op
    assert facade.get_kv_connector_stats() is None


def test_facade_build_prom_metrics_returns_prom(conn, monkeypatch):
    """build_prom_metrics forwards its four arguments to FlagCXPromMetrics."""
    module, _ = conn
    recorded = []

    class _FakeProm:
        def __init__(self, vllm_config, metric_types, labelnames, labelvalues):
            recorded.append((vllm_config, metric_types, labelnames, labelvalues))

    monkeypatch.setattr(module, "FlagCXPromMetrics", _FakeProm, raising=False)

    cfg, metric_types, labels, labelvalues = object(), {}, ["a"], {0: ["x"]}
    result = module.FlagCXConnector.build_prom_metrics(
        cfg, metric_types, labels, labelvalues
    )

    assert isinstance(result, _FakeProm)
    assert recorded == [(cfg, metric_types, labels, labelvalues)]


# ---------------------------------------------------------------------------
# Mamba truncation early return (target 434)
# ---------------------------------------------------------------------------


def test_truncate_mamba_returns_without_text_or_embeds(conn):
    """A prompt with neither token ids nor embeds returns without mutation."""
    module, _ = conn
    scheduler = object.__new__(module.FlagCXConnectorScheduler)
    request = SimpleNamespace(
        kv_transfer_params={},
        num_prompt_tokens=3,
        prompt_token_ids=None,
        prompt_embeds=None,
    )

    assert (
        module.FlagCXConnectorScheduler._truncate_mamba_request_for_prefill(
            scheduler, request
        )
        is None
    )
    assert request.num_prompt_tokens == 3
    assert request.kv_transfer_params == {}


# ---------------------------------------------------------------------------
# Worker.register_kv_caches (target 764-774, 778-835)
# ---------------------------------------------------------------------------


class _Topo:
    def __init__(self, mapping, blocks_first):
        self._mapping = mapping
        self.is_kv_layout_blocks_first = blocks_first

    def get_transfer_cache_regions(self, cache_or_caches, spec):
        return self._mapping.get(id(spec), [cache_or_caches])


def _make_worker(
    module,
    *,
    layer_specs,
    group_indices,
    mapping,
    blocks_first=False,
    role="kv_both",
):
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


def _kv_tensor(num_blocks=4, block_size=16, heads=2, head_size=8, dtype=torch.float16):
    return torch.zeros(num_blocks, block_size, heads, head_size, dtype=dtype)


def test_register_kv_caches_default_layout(conn):
    """K/V-first registration records one region per tensor and starts both
    the RPC accept server and the sender thread for a both-role worker."""
    module, _ = conn
    layer = "model.layers.0.self_attn"
    spec = object()
    tensor = _kv_tensor()
    worker = _make_worker(
        module,
        layer_specs={layer: spec},
        group_indices={layer: 0},
        mapping={id(spec): [tensor]},
        blocks_first=False,
        role="kv_both",
    )
    recorder = worker.flagcx.recorder

    worker.register_kv_caches({layer: tensor})

    block_len = tensor.stride(0) * tensor.element_size()
    assert worker.kv_caches_base_addr == [tensor.data_ptr()]
    assert worker.block_len_per_layer == [block_len]
    assert worker.kv_block_len_per_layer == [block_len]
    assert worker.registered_layer_names == [layer]
    assert worker.registered_layer_indices == [0]
    assert worker.registered_group_indices == [0]
    assert worker.device_kv_caches[layer] is tensor
    assert worker.kv_cache_device == tensor.device
    assert worker.rpc_port == 4242

    storage = tensor.untyped_storage()
    assert ("flagcxP2pRegister", storage.data_ptr(), storage.nbytes()) in recorder
    assert ("flagcxP2pStartRpcServer", "ENGINE") in recorder
    assert worker._sender_t is not None


def test_register_kv_caches_blocks_first_and_mla(conn, monkeypatch):
    """MLA uses page_size_bytes; a blocks-first attention region is halved;
    Mamba is never halved (kept whole)."""
    module, _ = conn

    class _MLA:
        page_size_bytes = 123

    class _SWMA:
        pass

    class _Mamba:
        pass

    monkeypatch.setattr(module, "MLAAttentionSpec", _MLA, raising=False)
    monkeypatch.setattr(module, "SlidingWindowMLASpec", _SWMA, raising=False)
    monkeypatch.setattr(module, "MambaSpec", _Mamba, raising=False)

    mla_layer = "model.layers.0.self_attn"
    plain_layer = "model.layers.1.self_attn"
    mamba_layer = "model.layers.2.mamba"
    mla_spec = _MLA()
    plain_spec = object()
    mamba_spec = _Mamba()

    mla_tensor = _kv_tensor()
    plain_tensor = _kv_tensor()
    mamba_tensor = _kv_tensor()

    worker = _make_worker(
        module,
        layer_specs={
            mla_layer: mla_spec,
            plain_layer: plain_spec,
            mamba_layer: mamba_spec,
        },
        group_indices={mla_layer: 0, plain_layer: 1, mamba_layer: 2},
        mapping={
            id(mla_spec): [mla_tensor],
            id(plain_spec): [plain_tensor],
            id(mamba_spec): [mamba_tensor],
        },
        blocks_first=True,
        role="kv_both",
    )

    worker.register_kv_caches(
        {
            mla_layer: mla_tensor,
            plain_layer: plain_tensor,
            mamba_layer: mamba_tensor,
        }
    )

    plain_len = plain_tensor.stride(0) * plain_tensor.element_size()
    mamba_len = mamba_tensor.stride(0) * mamba_tensor.element_size()
    assert worker.kv_block_len_per_layer == [123, plain_len // 2, mamba_len]
    assert worker.registered_layer_indices == [0, 1, 2]
    assert worker.registered_group_indices == [0, 1, 2]


def test_register_kv_caches_dedupes_shared_storage(conn):
    """Two views sharing one storage register a single memory region."""
    module, _ = conn
    layer = "model.layers.0.self_attn"
    spec = object()
    tensor = _kv_tensor()
    worker = _make_worker(
        module,
        layer_specs={layer: spec},
        group_indices={layer: 0},
        mapping={id(spec): [tensor, tensor]},
        blocks_first=False,
        role="kv_consumer",
    )
    recorder = worker.flagcx.recorder

    worker.register_kv_caches({layer: tensor})

    registers = [call for call in recorder if call[0] == "flagcxP2pRegister"]
    assert len(registers) == 1
    assert len(worker.block_len_per_layer) == 2


def test_register_kv_caches_requires_a_known_tensor(conn):
    """An unknown layer with no spec leaves nothing to register."""
    module, _ = conn
    tensor = _kv_tensor()
    worker = _make_worker(
        module,
        layer_specs={},
        group_indices={},
        mapping={},
        blocks_first=False,
        role="kv_consumer",
    )

    with pytest.raises(RuntimeError, match="No KV cache tensors"):
        worker.register_kv_caches({"model.layers.9.self_attn": tensor})


# ---------------------------------------------------------------------------
# _sender_thread dispatch loop (target 842-869)
# ---------------------------------------------------------------------------


def test_sender_thread_dispatches_and_relays(conn, monkeypatch):
    """The ROUTER/PULL loop submits work and relays status back."""
    module, _ = conn

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

    frontend = _Frontend()
    backend = _Backend()

    class _FakeZMQ:
        ROUTER = "router"
        PULL = "pull"
        POLLIN = 1

        class ContextTerminated(Exception):
            pass

        class ZMQError(Exception):
            pass

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
    monkeypatch.setattr(module, "zmq", _FakeZMQ, raising=False)
    monkeypatch.setattr(module, "make_zmq_path", lambda *a: "/path", raising=False)

    def make_socket(ctx, path, socket_type):
        return frontend if socket_type == _FakeZMQ.ROUTER else backend

    monkeypatch.setattr(module, "make_zmq_socket", make_socket, raising=False)

    worker = object.__new__(module.FlagCXConnectorWorker)
    worker.hostname = "127.0.0.1"
    worker.tp_rank = 0
    worker.zmq_ctx = object()
    worker._sender_executor = _Executor()
    ready = threading.Event()

    module.FlagCXConnectorWorker._sender_thread(worker, ready, 5000, 0)

    assert ready.is_set() is True
    assert len(submitted) == 1
    _, identity, metadata_bytes, channel_path = submitted[0]
    assert identity == b"identity"
    assert metadata_bytes == b"metadata"
    assert channel_path == "/path"
    assert frontend.sent == [(b"identity", b"", b"status")]
    assert frontend.closed is True
    assert backend.closed is True


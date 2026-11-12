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
"""Transfer-path coverage tests for flagcx_connector.py.

Covers the FlagCXConnector facade constructor, cache registration, the
sender thread/worker and the per-request write-plan builder, the receiver
coroutine's failure branches, transfer-region expansion, kernel block-id
translation, and shutdown/error handling.

The flagcx wheel boundary is faked with the recording facade from
``_flagcx_env`` (the sanctioned external-resource mock); expected addresses
and lengths are derived in closed form from the tensors and configs handed
to the connector. The ZMQ side channel is exercised with real local
sockets; one receiver test substitutes an exploding socket to force the
side-channel exception branch.
"""

from __future__ import annotations

import asyncio
import importlib
import socket
import sys
import threading
import types
import uuid

import pytest
import torch
import zmq

from ..._distributed import single_rank_gloo
from . import _flagcx_env, _flagcx_helpers as helpers

# zmq.asyncio needs a selector-based loop; on Windows the default is
# Proactor, which zmq cannot multiplex. Linux (the CI image) already
# uses epoll, so this only affects local runs.
if sys.platform == "win32":  # pragma: no cover - platform guard
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_flagcx_env.ensure_importable_env()
_flagcx_env.ensure_vllm_api_compat()


@pytest.fixture()
def worker_env(monkeypatch):
    """Import the connector under the fake wheel and own a one-rank TP group.

    Yields ``(conn, recorder, build, make_connector)``; teardown shuts down
    every worker built by either factory and drops the module cache.
    """
    recorder = _flagcx_env.install_fake_flagcx_wheel(monkeypatch)
    _flagcx_env.drop_connector_module()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as conn

    monkeypatch.setattr(conn, "get_kv_cache_layout", lambda: "NHD")
    monkeypatch.setattr(conn, "get_ip", lambda: "127.0.0.1")
    layer = helpers.FakeAttnLayer(helpers.FakeAttnBackend)
    workers: list = []

    def build(
        role="kv_consumer",
        groups=None,
        has_mamba=False,
        block_size=16,
        tp_size=1,
        total_kv_heads=2,
        use_mla=False,
        extra_config=None,
    ):
        """Construct a FlagCXConnectorWorker against faked configs."""
        if groups is None:
            groups = [helpers.full_attention_group(0, block_size=block_size)]
        cfg = helpers.vllm_config(
            model=helpers.ModelInfo(
                use_mla=use_mla, total_num_kv_heads=total_kv_heads
            ),
            parallel=helpers.ParallelInfo(tp_size=tp_size),
            transfer=helpers.TransferInfo(
                kv_role=role, extra_config=extra_config or {}
            ),
            cache=helpers.CacheInfo(block_size=block_size),
            compilation=helpers.CompilationInfo({"model.layers.0.self_attn": layer}),
        )
        w = conn.FlagCXConnectorWorker(
            cfg, "engt", helpers.FakeKVCacheConfig(groups, has_mamba_layers=has_mamba)
        )
        workers.append(w)
        return w

    def make_connector(role, kv_cache_config):
        """Construct the public FlagCXConnector facade for ``role``."""
        cfg = helpers.vllm_config(
            compilation=helpers.CompilationInfo({"model.layers.0.self_attn": layer})
        )
        connector = conn.FlagCXConnector(cfg, role, kv_cache_config)
        if connector.connector_worker is not None:
            workers.append(connector.connector_worker)
        return connector

    try:
        with single_rank_gloo("tp"):
            try:
                yield conn, recorder, build, make_connector
            finally:
                for worker in workers:
                    worker.shutdown()
    finally:
        _flagcx_env.drop_connector_module()


def _mla_group(conn, layer=5, block_size=16):
    """One MLAAttentionSpec group (page_size_bytes ignores the K/V factor)."""
    spec = conn.MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    return helpers.FakeKVGroup(spec, [f"model.layers.{layer}.self_attn"])


def _kv_cache(num_blocks=4, block_size=16, heads=2, head_size=8):
    """A K/V-first cache tensor: dim 0 holds K and V halves."""
    return torch.zeros(2, num_blocks, block_size, heads, head_size, dtype=torch.float16)


def _free_tcp_port():
    """Reserve and release a local port for the sender's ROUTER socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _region(
    conn,
    base_addr,
    *,
    block_len=512,
    kv_block_len=512,
    name="l0",
    layer_index=0,
    group_index=0,
):
    return conn.TransferRegion(
        layer_name=name,
        layer_index=layer_index,
        base_addr=base_addr,
        block_len=block_len,
        kv_block_len=kv_block_len,
        group_index=group_index,
    )


def _agent_meta(conn, req_blocks, remote_tp_rank=0, remote_tp_size=1):
    """Minimal wire metadata; parallel lists are unused by the planner."""
    return conn.FlagCXAgentMetadata(
        remote_hostname="10.0.0.2",
        remote_port=4242,
        remote_tp_size=remote_tp_size,
        remote_tp_rank=remote_tp_rank,
        req_blocks=req_blocks,
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
        registered_layer_names=[],
        registered_layer_indices=[],
        registered_group_indices=[],
    )


def _send_meta(conn, transfer_id="t1", ids=None, p_req_id="p1", ready=True):
    event = threading.Event()
    if ready:
        event.set()
    return conn.SendBlockMeta(
        p_req_id=p_req_id,
        transfer_id=transfer_id,
        local_block_ids=[[0, 1]] if ids is None else [list(g) for g in ids],
        ready=event,
    )


def _seed_send_state(
    conn,
    w,
    *,
    local_bases=(0x1000, 0x1800),
    remote_bases=(0x8000, 0x8800),
    block_lens=(512, 512),
    kv_block_lens=(512, 512),
    remote_kv_block_lens=None,
    local_ids=((0, 1),),
    remote_ids=((4, 5),),
    remote_names=("l0", "l0"),
    remote_tp_rank=0,
    remote_tp_size=1,
    p_req_id="p1",
    transfer_id="t1",
    d_req_id="d1",
    ready=True,
):
    """Populate one worker and its remote agent metadata for a transfer."""
    if remote_kv_block_lens is None:
        remote_kv_block_lens = kv_block_lens
    w.kv_caches_base_addr = list(local_bases)
    w.block_len_per_layer = list(block_lens)
    w.kv_block_len_per_layer = list(kv_block_lens)
    w.registered_layer_names = ["l0", "l0"]
    w.registered_layer_indices = [0, 0]
    w.registered_group_indices = [0, 0]

    send_meta = _send_meta(
        conn, transfer_id=transfer_id, ids=local_ids, p_req_id=p_req_id, ready=ready
    )
    w.reqs_need_send.reqs[transfer_id] = send_meta

    agent = conn.FlagCXAgentMetadata(
        remote_hostname="10.0.0.2",
        remote_port=4242,
        remote_tp_size=remote_tp_size,
        remote_tp_rank=remote_tp_rank,
        req_blocks={d_req_id: (transfer_id, [list(g) for g in remote_ids])},
        kv_caches_base_addr=list(remote_bases),
        block_lens=list(block_lens),
        kv_block_lens=list(remote_kv_block_lens),
        registered_layer_names=list(remote_names),
        registered_layer_indices=[0, 0],
        registered_group_indices=[0, 0],
    )
    return agent, send_meta


def _writes(recorder):
    return [c for c in recorder if c[0] == "flagcxP2pBatchWriteSync"]


# ---------------------------------------------------------------------------
# FlagCXConnector facade constructor
# ---------------------------------------------------------------------------


def test_connector_init_scheduler_role(worker_env):
    """The scheduler role builds only the scheduler half; the worker is None."""
    conn, _, _, make_connector = worker_env

    connector = make_connector(
        conn.KVConnectorRole.SCHEDULER,
        helpers.FakeKVCacheConfig([helpers.full_attention_group(0)]),
    )
    assert connector.connector_scheduler is not None
    assert connector.connector_scheduler.engine_id == "eng-test"
    assert connector.connector_worker is None
    assert connector.role == conn.KVConnectorRole.SCHEDULER


def test_connector_init_worker_role(worker_env):
    """The worker role builds only the worker half, with the transfer config
    handed through to the worker."""
    conn, _, _, make_connector = worker_env

    connector = make_connector(
        conn.KVConnectorRole.WORKER,
        helpers.FakeKVCacheConfig([helpers.full_attention_group(0)]),
    )
    assert connector.connector_scheduler is None
    assert connector.connector_worker is not None
    assert connector.connector_worker.kv_role == "kv_both"
    assert connector.role == conn.KVConnectorRole.WORKER


def test_connector_init_scheduler_requires_kv_cache_config(worker_env):
    """The scheduler constructor asserts on a missing KV cache config instead
    of half-initializing."""
    conn, _, _, make_connector = worker_env

    with pytest.raises(AssertionError, match="kv_cache_config"):
        make_connector(conn.KVConnectorRole.SCHEDULER, None)


def test_connector_init_requires_engine_id(worker_env):
    """An unset engine id is rejected before either side is constructed."""
    conn, _, _, _ = worker_env

    cfg = helpers.vllm_config(transfer=helpers.TransferInfo(engine_id=None))
    with pytest.raises(AssertionError):
        conn.FlagCXConnector(
            cfg,
            conn.KVConnectorRole.SCHEDULER,
            helpers.FakeKVCacheConfig([helpers.full_attention_group(0)]),
        )


# ---------------------------------------------------------------------------
# Worker constructor extras: source-tree library path, kernel block sync
# ---------------------------------------------------------------------------


def test_worker_source_tree_prefers_lib_library_path(monkeypatch, tmp_path):
    """The source-tree wrapper gets an explicit .so path, preferring
    lib/ over build/lib/ when both exist."""
    _flagcx_env.ensure_importable_env()
    _flagcx_env.ensure_vllm_api_compat()

    lib_dir = tmp_path / "lib"
    build_lib_dir = tmp_path / "build" / "lib"
    lib_dir.mkdir()
    build_lib_dir.mkdir(parents=True)
    preferred = lib_dir / "libflagcx.so"
    preferred.write_bytes(b"")
    (build_lib_dir / "libflagcx.so").write_bytes(b"")

    # Block the wheel facade so the import ladder takes the source-tree path.
    monkeypatch.setitem(sys.modules, "flagcx", None)
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))

    recorder = _flagcx_env.FlagCXCallRecorder()

    def factory(library_path=None):
        recorder.record("FLAGCXLibrary", library_path)
        return _flagcx_env.FakeFLAGCXLibrary(recorder)

    wrapper = types.ModuleType("plugin.interservice.flagcx_wrapper")
    wrapper.FLAGCXLibrary = factory
    interservice = types.ModuleType("plugin.interservice")
    interservice.__path__ = []
    interservice.flagcx_wrapper = wrapper
    plugin = types.ModuleType("plugin")
    plugin.__path__ = []
    plugin.interservice = interservice
    monkeypatch.setitem(sys.modules, "plugin", plugin)
    monkeypatch.setitem(sys.modules, "plugin.interservice", interservice)
    monkeypatch.setitem(
        sys.modules, "plugin.interservice.flagcx_wrapper", wrapper
    )

    _flagcx_env.drop_connector_module()
    try:
        conn = importlib.import_module(
            "vllm_fl.distributed.kv_transfer.flagcx_connector"
        )
        monkeypatch.setattr(conn, "get_kv_cache_layout", lambda: "NHD")
        monkeypatch.setattr(conn, "get_ip", lambda: "127.0.0.1")
        layer = helpers.FakeAttnLayer(helpers.FakeAttnBackend)
        cfg = helpers.vllm_config(
            compilation=helpers.CompilationInfo({"model.layers.0.self_attn": layer})
        )
        with single_rank_gloo("tp"):
            w = conn.FlagCXConnectorWorker(
                cfg,
                "engsrc",
                helpers.FakeKVCacheConfig([helpers.full_attention_group(0)]),
            )
            try:
                assert conn._flagcx_is_wheel is False
                assert ("FLAGCXLibrary", str(preferred)) in recorder
            finally:
                w.shutdown()
    finally:
        _flagcx_env.drop_connector_module()


def test_worker_syncs_block_size_with_kernel(worker_env, monkeypatch):
    """When the kernel only supports a smaller block size, the worker adopts
    it and expands logical block ids to kernel-physical ones; Mamba groups
    stay in the logical/page-id space."""
    conn, _, build, _ = worker_env
    monkeypatch.setattr(conn, "select_common_block_size", lambda size, backends: 8)

    groups = [helpers.full_attention_group(0), helpers.mamba_group(2)]
    w = build("kv_consumer", groups=groups, block_size=16)

    assert w.block_size == 8
    assert w.kv_topo.block_size == 8
    assert w._physical_blocks_per_logical_kv_block == 2
    assert w._logical_to_kernel_block_ids([[1, 2], [7]]) == [[2, 3, 4, 5], [7]]
    assert w._logical_to_kernel_block_ids([[], []]) == [[], []]


def test_shutdown_swallows_engine_destroy_failure(worker_env, monkeypatch):
    """A failing engine destroy is logged and cleared instead of escaping
    shutdown."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")

    def explode(engine):
        raise RuntimeError("destroy failed")

    monkeypatch.setattr(w.flagcx, "flagcxP2pEngineDestroy", explode)
    w.shutdown()
    assert w.engine is None
    # Idempotent: the teardown (and __del__) sees no engine left to destroy.
    w.shutdown()


# ---------------------------------------------------------------------------
# register_kv_caches bookkeeping
# ---------------------------------------------------------------------------


def test_register_kv_caches_consumer_regions(worker_env):
    """K/V regions share one storage MR, per-region lengths come from the
    tensor strides, and the consumer starts the RPC accept server."""
    conn, recorder, build, _ = worker_env
    w = build("kv_consumer")
    cache = _kv_cache()

    w.register_kv_caches({"model.layers.0.self_attn": cache})

    k_view, v_view = cache[0], cache[1]
    block_len = k_view.stride(0) * k_view.element_size()
    assert block_len == 512
    assert w.kv_caches_base_addr == [k_view.data_ptr(), v_view.data_ptr()]
    assert w.block_len_per_layer == [block_len, block_len]
    assert w.kv_block_len_per_layer == [block_len, block_len]
    assert w.registered_layer_names == ["model.layers.0.self_attn"] * 2
    assert w.registered_layer_indices == [0, 0]
    assert w.registered_group_indices == [0, 0]
    assert w.device_kv_caches["model.layers.0.self_attn"] is cache

    storage = cache.untyped_storage()
    registers = [c for c in recorder if c[0] == "flagcxP2pRegister"]
    assert registers == [
        ("flagcxP2pRegister", storage.data_ptr(), storage.nbytes())
    ]
    assert w.rpc_port == 4242
    assert ("flagcxP2pStartRpcServer",) in recorder
    # A consumer owns no sender executor or sender thread.
    assert not hasattr(w, "_sender_executor")
    assert not hasattr(w, "_sender_t")


def test_register_kv_caches_skips_unknown_layers(worker_env):
    """Layers without a spec are dropped without affecting known ones."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")
    known = _kv_cache()
    unknown = _kv_cache()

    w.register_kv_caches(
        {
            "model.layers.0.self_attn": known,
            "model.layers.9.self_attn": unknown,
        }
    )

    assert w.registered_layer_names == ["model.layers.0.self_attn"] * 2
    assert w.kv_caches_base_addr == [known[0].data_ptr(), known[1].data_ptr()]


def test_register_kv_caches_requires_a_tensor(worker_env):
    """With no known tensor there is nothing to register and registration
    fails loudly."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")

    with pytest.raises(RuntimeError, match="No KV cache tensors"):
        w.register_kv_caches({"model.layers.9.self_attn": _kv_cache()})
    assert w.kv_caches_base_addr == []
    assert w.block_len_per_layer == []
    assert w.registered_layer_names == []


def test_register_kv_caches_mla_uses_page_size(worker_env):
    """MLA regions take their kv_block_len from the spec's page size, not
    from the tensor stride."""
    conn, _, build, _ = worker_env
    group = _mla_group(conn)
    w = build("kv_consumer", groups=[group], use_mla=True)
    cache = torch.zeros(4, 16, 8, dtype=torch.float32)

    w.register_kv_caches({"model.layers.5.self_attn": cache})

    assert w.block_len_per_layer == [512]
    assert w.kv_block_len_per_layer == [256]
    assert w.kv_caches_base_addr == [cache.data_ptr()]
    assert w.registered_layer_names == ["model.layers.5.self_attn"]
    assert w.registered_layer_indices == [5]


def test_register_kv_caches_mamba_registers_conv_only(worker_env):
    """Mamba/GDN registration keeps only the conv-state tensor."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer", groups=[helpers.mamba_group(2)])
    conv = torch.zeros(4, 8, 2, dtype=torch.float32)
    ssm = torch.zeros(4, 16, dtype=torch.float32)

    w.register_kv_caches({"model.layers.2.mamba": [conv, ssm]})

    block_len = conv.stride(0) * conv.element_size()
    assert block_len == 64
    assert w.kv_caches_base_addr == [conv.data_ptr()]
    assert w.block_len_per_layer == [block_len]
    assert w.kv_block_len_per_layer == [block_len]
    assert w.registered_layer_indices == [2]
    assert w.registered_group_indices == [0]


def test_register_kv_caches_blocks_first_halves_block_len(worker_env):
    """On the hybrid blocks-first layout the region length covers K+V, so
    each virtual half is half that length."""
    conn, _, build, _ = worker_env
    groups = [helpers.full_attention_group(0), helpers.mamba_group(2)]
    w = build("kv_consumer", groups=groups, has_mamba=True)
    cache = _kv_cache()

    w.register_kv_caches({"model.layers.0.self_attn": cache})

    assert w.kv_topo.is_kv_layout_blocks_first is True
    transposed = cache.transpose(0, 1)
    block_len = transposed.stride(0) * transposed.element_size()
    assert block_len == 512
    assert w.block_len_per_layer == [block_len]
    assert w.kv_block_len_per_layer == [block_len // 2]
    assert w.kv_caches_base_addr == [cache.untyped_storage().data_ptr()]


# ---------------------------------------------------------------------------
# Sender thread / worker
# ---------------------------------------------------------------------------


def test_sender_thread_dispatches_and_relays_done(worker_env, monkeypatch):
    """End to end on CPU: registering starts the ROUTER sender; a dispatch
    arms the transfer, the pool worker builds the write descriptors, the
    flagcx facade performs them, and TRANS_DONE is relayed back."""
    conn, recorder, build, _ = worker_env
    port = _free_tcp_port()
    monkeypatch.setenv("FLAGCX_BOOTSTRAP_PORT", str(port))

    w = build("kv_producer")
    cache = _kv_cache()
    w.register_kv_caches({"model.layers.0.self_attn": cache})

    assert w.rpc_port == 4242
    assert w._sender_t is not None
    assert w._sender_t.is_alive()
    # The producer registers MRs but does not accept RPC connections.
    assert ("flagcxP2pStartRpcServer",) not in recorder

    k_ptr, v_ptr = cache[0].data_ptr(), cache[1].data_ptr()
    block_len = cache[0].stride(0) * cache[0].element_size()
    assert w.kv_caches_base_addr == [k_ptr, v_ptr]

    meta = conn.FlagCXConnectorMetadata()
    meta.reqs_to_send["p1"] = ("t1", [[0]])
    w.start_load_kv(meta)

    remote_bases = [0x100000, 0x102000]
    agent = conn.FlagCXAgentMetadata(
        remote_hostname="127.0.0.1",
        remote_port=port,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d1": ("t1", [[0]])},
        kv_caches_base_addr=remote_bases,
        block_lens=[block_len, block_len],
        kv_block_lens=[block_len, block_len],
        registered_layer_names=list(w.registered_layer_names),
        registered_layer_indices=list(w.registered_layer_indices),
        registered_group_indices=list(w.registered_group_indices),
    )

    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 15000)
    dealer.setsockopt(zmq.LINGER, 0)
    try:
        dealer.connect(f"tcp://127.0.0.1:{port}")
        dealer.send_multipart([b"", w._encoder.encode(agent)])
        frames = dealer.recv_multipart()
    finally:
        dealer.close()
        ctx.term()

    assert frames == [b"", conn.TRANS_DONE]
    assert _writes(recorder) == [
        (
            "flagcxP2pBatchWriteSync",
            f"127.0.0.1:{port}",
            [k_ptr, v_ptr],
            [block_len, block_len],
        )
    ]

    stats = w.get_kv_connector_stats()
    assert stats.data["bytes_transferred"] == [2 * block_len]
    assert stats.data["num_descriptors"] == [2]
    assert w.get_finished() == ({"p1"}, None)


def test_sender_worker_reports_transfer_error(worker_env):
    """A setup failure inside the transfer is converted into TRANS_ERROR on
    the internal reply channel instead of escaping the pool thread."""
    conn, recorder, build, _ = worker_env
    w = build("kv_producer")
    agent, _ = _seed_send_state(
        conn, w, remote_names=("mismatched.layer", "mismatched.layer")
    )

    channel = conn.make_zmq_path("inproc", str(uuid.uuid4()))
    pull = conn.make_zmq_socket(w.zmq_ctx, channel, zmq.PULL)
    pull.setsockopt(zmq.RCVTIMEO, 5000)
    try:
        w._sender_worker(b"client", w._encoder.encode(agent), channel)
        frames = pull.recv_multipart()
    finally:
        pull.close()

    assert frames == [b"client", conn.TRANS_ERROR]
    assert _writes(recorder) == []
    # The failed worker did not mark the request finished.
    assert w.finished_sending_reqs.set == set()


# ---------------------------------------------------------------------------
# Transfer region expansion
# ---------------------------------------------------------------------------


def test_get_transfer_regions_keeps_tensors_whole(worker_env):
    """With a K/V-first layout no region is split; metadata is preserved."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")

    regions = w._get_transfer_regions(
        [0x1000, 0x1800],
        [512, 512],
        [512, 512],
        ["l0", "l0"],
        [3, 3],
        [0, 0],
    )

    assert [
        (r.layer_name, r.layer_index, r.base_addr, r.block_len, r.kv_block_len, r.group_index)
        for r in regions
    ] == [
        ("l0", 3, 0x1000, 512, 512, 0),
        ("l0", 3, 0x1800, 512, 512, 0),
    ]


def test_get_transfer_regions_blocks_first_splits_attention_only(worker_env):
    """Blocks-first splits full-attention regions into K and V at
    base + kv_block_len; Mamba and MLA regions stay whole."""
    conn, _, build, _ = worker_env
    groups = [
        helpers.full_attention_group(0),
        helpers.mamba_group(2),
        _mla_group(conn),
    ]
    w = build("kv_consumer", groups=groups, has_mamba=True)

    regions = w._get_transfer_regions(
        [4096, 8192, 12288],
        [512, 512, 512],
        [256, 256, 256],
        ["model.layers.0.self_attn", "model.layers.2.mamba", "model.layers.5.self_attn"],
        [0, 2, 5],
        [0, 1, 2],
    )

    assert [
        (r.layer_name, r.layer_index, r.base_addr, r.group_index) for r in regions
    ] == [
        ("model.layers.0.self_attn", 0, 4096, 0),
        ("model.layers.0.self_attn", 0, 4352, 0),
        ("model.layers.2.mamba", 2, 8192, 1),
        ("model.layers.5.self_attn", 5, 12288, 2),
    ]


# ---------------------------------------------------------------------------
# _build_transfer_params
# ---------------------------------------------------------------------------


def test_build_transfer_params_coalesces_contiguous_runs(worker_env):
    """Contiguous local/remote block runs become one descriptor per region,
    addressed at the first block of each run."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")
    local = [_region(conn, 0x1000), _region(conn, 0x1800)]
    remote = [_region(conn, 0x8000), _region(conn, 0x8800)]
    send_meta = _send_meta(conn, ids=[[0, 1, 2]])
    agent = _agent_meta(conn, {"d1": ("t1", [[5, 6, 7]])})

    src, dst, sizes = w._build_transfer_params(
        [("d1", send_meta)], agent, local, remote
    )

    assert src == [0x1000, 0x1800]
    assert dst == [0x8000 + 5 * 512, 0x8800 + 5 * 512]
    assert sizes == [3 * 512, 3 * 512]


def test_build_transfer_params_per_block_when_not_coalescable(worker_env):
    """When a transfer covers only part of a block (split K/V halves) each
    block gets its own descriptor."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")
    local = [
        _region(conn, 0x1000, block_len=512, kv_block_len=256),
        _region(conn, 0x1800, block_len=512, kv_block_len=256),
    ]
    remote = [
        _region(conn, 0x8000, block_len=512, kv_block_len=256),
        _region(conn, 0x8800, block_len=512, kv_block_len=256),
    ]
    send_meta = _send_meta(conn, ids=[[0, 1]])
    agent = _agent_meta(conn, {"d1": ("t1", [[2, 3]])})

    src, dst, sizes = w._build_transfer_params(
        [("d1", send_meta)], agent, local, remote
    )

    assert src == [0x1000, 0x1200, 0x1800, 0x1A00]
    assert dst == [0x8000 + 2 * 512, 0x8000 + 3 * 512, 0x8800 + 2 * 512, 0x8800 + 3 * 512]
    assert sizes == [256, 256, 256, 256]


def test_build_transfer_params_trims_local_tail(worker_env):
    """Extra local blocks are trimmed to the remote tail, side-aligned."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")
    local = [_region(conn, 0x1000), _region(conn, 0x1800)]
    remote = [_region(conn, 0x8000), _region(conn, 0x8800)]
    send_meta = _send_meta(conn, ids=[[0, 1, 2, 3]])
    agent = _agent_meta(conn, {"d1": ("t1", [[5]])})

    src, dst, sizes = w._build_transfer_params(
        [("d1", send_meta)], agent, local, remote
    )

    assert src == [0x1000 + 3 * 512, 0x1800 + 3 * 512]
    assert dst == [0x8000 + 5 * 512, 0x8800 + 5 * 512]
    assert sizes == [512, 512]


def test_build_transfer_params_skips_group_count_mismatch(worker_env):
    """A local/remote KV-group count mismatch drops the request entirely."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")
    local = [_region(conn, 0x1000)]
    send_meta = _send_meta(conn, ids=[[0]])
    agent = _agent_meta(conn, {"d1": ("t1", [[1], [2]])})

    assert w._build_transfer_params([("d1", send_meta)], agent, local, local) == (
        [],
        [],
        [],
    )


def test_build_transfer_params_skips_local_shorter_than_remote(worker_env):
    """A local block count below the remote one is an error for that request
    and emits no descriptors."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")
    local = [_region(conn, 0x1000)]
    send_meta = _send_meta(conn, ids=[[0]])
    agent = _agent_meta(conn, {"d1": ("t1", [[1, 2]])})

    assert w._build_transfer_params([("d1", send_meta)], agent, local, local) == (
        [],
        [],
        [],
    )


def test_build_transfer_params_skips_empty_remote_groups(worker_env):
    """A request whose remote groups hold no blocks is skipped."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")
    local = [_region(conn, 0x1000)]
    send_meta = _send_meta(conn, ids=[[0]])
    agent = _agent_meta(conn, {"d1": ("t1", [[]])})

    assert w._build_transfer_params([("d1", send_meta)], agent, local, local) == (
        [],
        [],
        [],
    )


def test_build_transfer_params_strips_mamba_null_blocks(worker_env):
    """Mamba null block placeholders are removed before pairing."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer", groups=[helpers.mamba_group(2)])
    local = [_region(conn, 0x1000)]
    remote = [_region(conn, 0x8000)]
    send_meta = _send_meta(conn, ids=[[0, 3]])
    agent = _agent_meta(conn, {"d1": ("t1", [[0, 4]])})

    src, dst, sizes = w._build_transfer_params(
        [("d1", send_meta)], agent, local, remote
    )

    assert src == [0x1000 + 3 * 512]
    assert dst == [0x8000 + 4 * 512]
    assert sizes == [512]


def test_build_transfer_params_skips_empty_and_unknown_groups(worker_env):
    """Empty local groups and regions tagged with an out-of-range group index
    produce no descriptors, while populated groups still transfer."""
    conn, _, build, _ = worker_env
    groups = [helpers.full_attention_group(0), helpers.full_attention_group(1)]
    w = build("kv_consumer", groups=groups)
    local = [
        _region(conn, 0x1000, group_index=0),
        _region(conn, 0x1800, group_index=1),
        _region(conn, 0x2000, group_index=5),
    ]
    remote = [
        _region(conn, 0x8000, group_index=0),
        _region(conn, 0x8800, group_index=1),
        _region(conn, 0x9000, group_index=5),
    ]
    send_meta = _send_meta(conn, ids=[[], [5]])
    agent = _agent_meta(conn, {"d1": ("t1", [[], [6]])})

    src, dst, sizes = w._build_transfer_params(
        [("d1", send_meta)], agent, local, remote
    )

    assert src == [0x1800 + 5 * 512]
    assert dst == [0x8800 + 6 * 512]
    assert sizes == [512]


def test_build_transfer_params_without_ready_reqs(worker_env):
    """No ready requests means no descriptors."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")

    assert w._build_transfer_params([], _agent_meta(conn, {}), [], []) == (
        [],
        [],
        [],
    )


# ---------------------------------------------------------------------------
# _send_kv_to_decode
# ---------------------------------------------------------------------------


def test_send_kv_to_decode_writes_and_finishes(worker_env):
    """A paired transfer writes both regions and moves the prefill request
    into finished_sending."""
    conn, recorder, build, _ = worker_env
    w = build("kv_producer")
    agent, _ = _seed_send_state(conn, w)

    w._send_kv_to_decode(agent)

    assert _writes(recorder) == [
        (
            "flagcxP2pBatchWriteSync",
            "10.0.0.2:4242",
            [0x1000, 0x1800],
            [2 * 512, 2 * 512],
        )
    ]
    assert w.reqs_need_send.reqs == {}
    assert w.finished_sending_reqs.set == {"p1"}
    stats = w.get_kv_connector_stats()
    assert stats.data["bytes_transferred"] == [4 * 512]
    assert stats.data["num_descriptors"] == [2]


def test_send_kv_to_decode_warns_for_unpaired_rank(worker_env):
    """A remote rank outside the handshake targets only logs a warning; the
    transfer proceeds."""
    conn, recorder, build, _ = worker_env
    w = build("kv_producer")
    agent, _ = _seed_send_state(conn, w, remote_tp_rank=1)

    w._send_kv_to_decode(agent)

    assert len(_writes(recorder)) == 1
    assert w.finished_sending_reqs.set == {"p1"}


def test_send_kv_to_decode_times_out_waiting_for_blocks(worker_env):
    """A transfer whose scheduler blocks never commit raises instead of
    writing."""
    conn, recorder, build, _ = worker_env
    w = build("kv_producer")
    agent, _ = _seed_send_state(conn, w, ready=False)
    w._abort_request_timeout = 0

    with pytest.raises(RuntimeError, match="Timed out waiting for prefill blocks"):
        w._send_kv_to_decode(agent)

    assert _writes(recorder) == []
    assert "t1" in w.reqs_need_send.reqs


def test_send_kv_to_decode_rejects_unalignable_regions(worker_env):
    """A remote region without a matching local occurrence aborts setup."""
    conn, recorder, build, _ = worker_env
    w = build("kv_producer")
    agent, _ = _seed_send_state(
        conn, w, remote_names=("mismatched.layer", "mismatched.layer")
    )

    with pytest.raises(RuntimeError, match="no matching consumer occurrence"):
        w._send_kv_to_decode(agent)

    assert _writes(recorder) == []


def test_send_kv_to_decode_rejects_asymmetric_lengths(worker_env):
    """Homogeneous TP requires equal region lengths; a mismatch aborts."""
    conn, recorder, build, _ = worker_env
    w = build("kv_producer")
    agent, _ = _seed_send_state(conn, w, remote_kv_block_lens=(1024, 1024))

    with pytest.raises(RuntimeError, match="KV region length mismatch"):
        w._send_kv_to_decode(agent)

    assert _writes(recorder) == []


def test_send_kv_to_decode_records_write_failure(worker_env):
    """A failing one-sided write is counted as a failed transfer and
    re-raised, leaving the pending send in place."""
    conn, recorder, build, _ = worker_env
    w = build("kv_producer")
    agent, _ = _seed_send_state(conn, w)
    w.flagcx.write_should_raise = RuntimeError("rdma-eio")

    with pytest.raises(RuntimeError, match="rdma-eio"):
        w._send_kv_to_decode(agent)

    assert w.xfer_stats.data["num_failed_transfers"] == [1]
    assert w.xfer_stats.data["transfer_duration"] == []
    assert "t1" in w.reqs_need_send.reqs


def test_send_kv_to_decode_requires_prefill_request_id(worker_env):
    """A completed transfer with no owning prefill request is an internal
    error."""
    conn, _, build, _ = worker_env
    w = build("kv_producer")
    agent, _ = _seed_send_state(conn, w, p_req_id="")

    with pytest.raises(RuntimeError, match="Missing Prefill request ID"):
        w._send_kv_to_decode(agent)

    assert "t1" not in w.reqs_need_send.reqs


# ---------------------------------------------------------------------------
# _receive_kv failure branches
# ---------------------------------------------------------------------------


def test_receive_kv_error_reply_records_failed_recv(worker_env):
    """A non-DONE reply is counted as a failed receive and never finishes
    the request."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.RCVTIMEO, 10000)
    rep.bind("tcp://127.0.0.1:0")
    port = int(rep.getsockopt(zmq.LAST_ENDPOINT).rsplit(b":", 1)[1])
    w._pull_pending["cr1"] = 2
    path = f"tcp://127.0.0.1:{port}"
    try:
        fut = asyncio.run_coroutine_threadsafe(
            w._receive_kv(path, {"cr1": ("t1", [[1]])}), w.receiver_loop
        )
        rep.recv()
        rep.send(conn.TRANS_ERROR)
        fut.result(timeout=10)
    finally:
        rep.close()
        ctx.term()

    assert w.xfer_stats.data["num_failed_recvs"] == [1]
    assert w.finished_recving_reqs.set == set()
    assert w._pull_pending == {"cr1": 2}


def test_receive_kv_exception_records_failed_recv(worker_env, monkeypatch):
    """A side-channel exception is reported as a failed receive and the
    socket is still closed."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")

    class _ExplodingSocket:
        def __init__(self):
            self.closed = False

        def setsockopt(self, *args):
            pass

        async def send(self, payload):
            raise RuntimeError("side-channel down")

        async def recv(self):  # pragma: no cover - must not be reached
            raise AssertionError("recv must not run")

        def close(self):
            self.closed = True

    sock = _ExplodingSocket()
    monkeypatch.setattr(conn, "make_zmq_socket", lambda *a, **k: sock)

    fut = asyncio.run_coroutine_threadsafe(
        w._receive_kv("tcp://127.0.0.1:1", {"cx": ("t1", [])}),
        w.receiver_loop,
    )
    fut.result(timeout=10)

    assert sock.closed is True
    assert w.xfer_stats.data["num_failed_recvs"] == [1]
    assert w.finished_recving_reqs.set == set()


def test_receive_kv_waits_for_every_target_rank(worker_env):
    """With several target prefill ranks the request finishes only after the
    last TRANS_DONE reply."""
    conn, _, build, _ = worker_env
    w = build("kv_consumer")
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.RCVTIMEO, 10000)
    rep.bind("tcp://127.0.0.1:0")
    port = int(rep.getsockopt(zmq.LAST_ENDPOINT).rsplit(b":", 1)[1])
    path = f"tcp://127.0.0.1:{port}"
    blocks = {"cr2": ("t2", [[3, 4]])}
    w._pull_pending["cr2"] = 2
    try:
        for expected_remaining in (1, None):
            fut = asyncio.run_coroutine_threadsafe(
                w._receive_kv(path, blocks), w.receiver_loop
            )
            rep.recv()
            rep.send(conn.TRANS_DONE)
            fut.result(timeout=10)
            if expected_remaining is None:
                assert "cr2" not in w._pull_pending
            else:
                assert w._pull_pending == {"cr2": expected_remaining}
    finally:
        rep.close()
        ctx.term()

    assert w.finished_recving_reqs.set == {"cr2"}

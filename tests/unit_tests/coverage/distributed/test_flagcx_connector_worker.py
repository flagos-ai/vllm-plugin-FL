# Copyright (c) 2026 BAAI. All rights reserved.
"""Worker-side lifecycle tests for FlagCXConnectorWorker.

Covers construction bookkeeping, send-request arming, timeout expiration
sweeps, finished-set accounting, connection memoization, the msgspec agent
metadata wire format and shutdown ordering.  The real GPU/FlagCX data-plane
paths (batch writes, the ZMQ sender thread, RDMA handshake) stay covered by
GPU functional tests.
"""

import threading
import time

import pytest

from tests.unit_tests.coverage.distributed._flagcx_stubs import (
    FlagcxStubEnv,
    VllmConfigDouble,
    make_kv_cache_config,
)

pytest.importorskip("msgspec")
pytest.importorskip("numpy")
pytest.importorskip("zmq")


@pytest.fixture
def worker_env(monkeypatch):
    """Install stubs and yield a factory building FlagCXConnectorWorker
    instances with the mocked FlagCX library boundary."""
    env = FlagcxStubEnv(monkeypatch).install()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as fc

    monkeypatch.setattr(fc, "get_current_attn_backends", lambda cfg: [_StubBackend()])
    env.fc = fc

    def make_worker(role="kv_both", extra_config=None, config=None):
        return fc.FlagCXConnectorWorker(
            VllmConfigDouble(role, extra_config=extra_config),
            "e0",
            config if config is not None else make_kv_cache_config(),
        )

    env.make_worker = make_worker
    return env


class _StubBackend:
    def get_name(self):
        return "stub_backend"


def _agent_meta(fc, **overrides):
    """Build a FlagCXAgentMetadata with sensible CPU defaults."""
    kwargs = dict(
        remote_hostname="10.0.0.1",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d1": ("t-agent", [[1]])},
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
        registered_layer_names=[],
        registered_layer_indices=[],
        registered_group_indices=[],
    )
    kwargs.update(overrides)
    return fc.FlagCXAgentMetadata(**kwargs)


def test_worker_construction_bookkeeping(worker_env):
    """The worker captures the engine handle, role, pool sizing and
    per-layer specs from the config."""
    worker = worker_env.make_worker(role="kv_both", extra_config={"num_workers": 3})
    try:
        assert worker.engine == "engine-handle"
        assert worker.kv_role == "kv_both"
        assert worker.num_workers == 3
        assert worker.registered_layer_names == []
        assert worker.kv_caches_base_addr == []
        assert list(worker._layer_specs) == ["model.layers.0.self_attn.attn"]
        assert worker._layer_group_indices == {"model.layers.0.self_attn.attn": 0}
        assert worker._physical_blocks_per_logical_kv_block == 1
        assert worker._abort_request_timeout == 480
        assert worker.backend_name == "stub_backend"
    finally:
        worker.shutdown()


def test_worker_default_num_workers(worker_env):
    """Without extra config the sender pool defaults to 10 workers."""
    worker = worker_env.make_worker(role="kv_both")
    try:
        assert worker.num_workers == 10
    finally:
        worker.shutdown()


def test_worker_abort_timeout_env(worker_env, monkeypatch):
    """FLAGCX_CONNECTOR_ABORT_REQUEST_TIMEOUT overrides the default."""
    monkeypatch.setenv("FLAGCX_CONNECTOR_ABORT_REQUEST_TIMEOUT", "7")
    worker = worker_env.make_worker(role="kv_both")
    try:
        assert worker._abort_request_timeout == 7
    finally:
        worker.shutdown()


def test_worker_library_path_resolution(worker_env, monkeypatch):
    """FLAGCX_LIB_PATH wins; otherwise FLAGCX_PATH/build/lib/libflagcx.so is
    passed to the native library constructor."""
    import os

    fc = worker_env.fc
    captured = []

    def fake_library(path):
        captured.append(path)
        return worker_env.flagcx_lib

    # The connector bound FLAGCXLibrary at import time; patch its own name.
    monkeypatch.setattr(fc, "FLAGCXLibrary", fake_library)
    monkeypatch.setenv("FLAGCX_PATH", "/flagcx")
    worker = worker_env.make_worker(role="kv_producer")
    try:
        # os.path.join is platform-dependent; assert the components.
        assert len(captured) == 1
        assert captured[0].replace("\\", "/") == "/flagcx/build/lib/libflagcx.so"
        assert os.path.isabs(captured[0]) or captured[0].startswith("/flagcx")
    finally:
        worker.shutdown()

    captured.clear()
    monkeypatch.setenv("FLAGCX_LIB_PATH", "/custom/libflagcx.so")
    worker2 = worker_env.make_worker(role="kv_producer")
    try:
        assert captured == ["/custom/libflagcx.so"]
    finally:
        worker2.shutdown()


def test_worker_roles_thread_setup(worker_env):
    """kv_producer starts the sender pool but no receiver loop;
    kv_consumer starts the receiver loop but no sender pool."""
    producer = worker_env.make_worker(role="kv_producer")
    try:
        assert producer._sender_executor is not None
        assert not hasattr(producer, "receiver_loop")
        finished_send, finished_recv = producer.get_finished()
        assert finished_recv is None
    finally:
        producer.shutdown()

    consumer = worker_env.make_worker(role="kv_consumer")
    try:
        assert consumer.receiver_loop.is_running()
        assert not hasattr(consumer, "_sender_executor")
        finished_send, finished_recv = consumer.get_finished()
        assert finished_send is None
    finally:
        consumer.shutdown()


def test_worker_start_load_kv_arms_send_requests(worker_env):
    """start_load_kv registers send metas, binds the prefill request id and
    arms ready requests with committed block ids and an expiry deadline."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        metadata = fc.FlagCXConnectorMetadata()
        metadata.add_new_req(
            request_id="p1",
            local_block_ids=[[1, 2]],
            kv_transfer_params={"transfer_id": "t1"},
            load_remote_cache=False,
        )
        worker.start_load_kv(metadata)

        send_meta = worker.reqs_need_send.reqs["t1"]
        assert send_meta.p_req_id == "p1"
        assert send_meta.local_block_ids == [[1, 2]]
        assert send_meta.ready.is_set()
        assert send_meta.expire_time > time.perf_counter()
        assert send_meta.need_send == 0
    finally:
        worker.shutdown()


def test_worker_start_load_kv_reuses_existing_send_meta(worker_env):
    """A send meta already registered by the decode side keeps its state
    and only binds the request id when blocks are committed."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        existing = fc.SendBlockMeta(
            p_req_id="",
            transfer_id="t2",
            local_block_ids=[],
            ready=threading.Event(),
        )
        existing.need_send = 1
        worker.reqs_need_send.reqs["t2"] = existing

        metadata = fc.FlagCXConnectorMetadata()
        metadata.add_new_req(
            request_id="p9",
            local_block_ids=[[7]],
            kv_transfer_params={"transfer_id": "t2"},
            load_remote_cache=False,
        )
        worker.start_load_kv(metadata)

        assert worker.reqs_need_send.reqs["t2"] is existing
        assert existing.p_req_id == "p9"
        assert existing.local_block_ids == [[7]]
        assert existing.ready.is_set()
        assert existing.need_send == 1  # preserved, not reset
    finally:
        worker.shutdown()


def test_worker_start_load_kv_no_blocks_leaves_unarmed(worker_env):
    """A send registration with empty block ids (blocks not committed yet)
    stays unarmed until request_finished commits them."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        metadata = fc.FlagCXConnectorMetadata()
        metadata.add_new_req(
            request_id="p2",
            local_block_ids=[],
            kv_transfer_params={"transfer_id": "t3"},
            load_remote_cache=False,
        )
        worker.start_load_kv(metadata)
        send_meta = worker.reqs_need_send.reqs["t3"]
        assert send_meta.p_req_id == "p2"
        assert not send_meta.ready.is_set()
        assert send_meta.expire_time == float("inf")
    finally:
        worker.shutdown()


def test_worker_get_finished_expires_stale_sends(worker_env):
    """Expired send metas are dropped, recorded as expired reqs, and their
    prefill request ids surface as finished sends."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        worker._abort_request_timeout = 0
        send_meta = fc.SendBlockMeta(
            p_req_id="p3",
            transfer_id="t4",
            local_block_ids=[[1]],
            ready=threading.Event(),
        )
        send_meta.expire_time = time.perf_counter() - 1  # already expired
        worker.reqs_need_send.reqs["t4"] = send_meta

        finished_send, finished_recv = worker.get_finished()
        assert finished_send == {"p3"}
        assert finished_recv is None
        assert worker.reqs_need_send.reqs == {}
        assert worker.xfer_stats.data["num_kv_expired_reqs"] == [1]
    finally:
        worker.shutdown()


def test_worker_get_finished_returns_and_drains_sets(worker_env):
    """Non-expired sends return finished sets that drain on read."""
    worker = worker_env.make_worker(role="kv_producer")
    try:
        worker.finished_sending_reqs.set.update({"a", "b"})
        finished_send, _ = worker.get_finished()
        assert finished_send == {"a", "b"}
        # Second call: nothing left, and no expiry fired.
        finished_send, _ = worker.get_finished()
        assert finished_send is None
        assert worker.xfer_stats.is_empty()
    finally:
        worker.shutdown()


def test_worker_get_finished_keeps_unexpired_sends(worker_env):
    """Sends armed with a future deadline are not swept by get_finished."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        send_meta = fc.SendBlockMeta(
            p_req_id="p4",
            transfer_id="t5",
            local_block_ids=[[1]],
            ready=threading.Event(),
        )
        send_meta.expire_time = time.perf_counter() + 3600
        worker.reqs_need_send.reqs["t5"] = send_meta

        finished_send, _ = worker.get_finished()
        assert finished_send is None
        assert "t5" in worker.reqs_need_send.reqs
        assert worker.xfer_stats.is_empty()
    finally:
        worker.shutdown()


def test_worker_get_kv_connector_stats_snapshot(worker_env):
    """get_kv_connector_stats clones-and-resets only when non-empty."""
    worker = worker_env.make_worker(role="kv_producer")
    try:
        assert worker.get_kv_connector_stats() is None
        worker.xfer_stats.record_transfer(0.25, 4096, 3)
        stats = worker.get_kv_connector_stats()
        assert stats.data["transfer_duration"] == [0.25]
        assert stats.data["bytes_transferred"] == [4096]
        assert stats.data["num_descriptors"] == [3]
        assert worker.get_kv_connector_stats() is None
    finally:
        worker.shutdown()


def test_worker_shutdown_destroys_engine_once(worker_env):
    """shutdown tears the ZMQ contexts down, stops the receiver loop,
    destroys the engine exactly once and is idempotent."""
    worker = worker_env.make_worker(role="kv_both")
    worker.shutdown()

    assert worker.engine is None
    worker_env.flagcx_lib.flagcxP2pEngineDestroy.assert_called_once_with(
        "engine-handle"
    )
    assert not worker.receiver_loop.is_running()
    # The sender thread is only spawned by register_kv_caches (GPU path);
    # before registration the constructor leaves the slot unset.
    assert worker._sender_t is None

    # Repeated shutdown stays a no-op on the native boundary; the destructor
    # delegates to the same teardown path (GC-driven), verified via the
    # already-idempotent shutdown above.
    worker.shutdown()
    assert worker_env.flagcx_lib.flagcxP2pEngineDestroy.call_count == 1


def test_worker_shutdown_survives_engine_destroy_error(worker_env):
    """A failing flagcxP2pEngineDestroy is logged, not raised, and the
    engine handle is still cleared."""
    worker_env.flagcx_lib.flagcxP2pEngineDestroy.side_effect = RuntimeError(
        "native teardown failed"
    )
    worker = worker_env.make_worker(role="kv_both")
    worker.shutdown()  # must not raise
    assert worker.engine is None


def test_worker_sender_thread_registration_failure_raises(worker_env):
    """register_kv_caches raises RuntimeError when no KV tensors are given
    (nothing registered with the native library)."""
    worker = worker_env.make_worker(role="kv_producer")
    try:
        with pytest.raises(RuntimeError, match="No KV cache tensors"):
            worker.register_kv_caches({})
    finally:
        worker.shutdown()


def test_worker_logical_to_kernel_blocks_identity(worker_env):
    """With a 1:1 logical/kernel block ratio the ids pass through."""
    worker = worker_env.make_worker(role="kv_both")
    try:
        assert worker._logical_to_kernel_block_ids([[1, 2], [3]]) == [[1, 2], [3]]
    finally:
        worker.shutdown()


def test_worker_sender_timeout_raises_after_deadline(worker_env):
    """_send_kv_to_decode times out with a clear error when the scheduler
    never commits the request's blocks."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        worker._abort_request_timeout = 0

        class _Topo:
            local_replicates_kv_cache = False
            is_kv_layout_blocks_first = False

            def handshake_target_ranks(self, remote_tp_size):
                return [0]

        worker.kv_topo = _Topo()
        meta = _agent_meta(fc)
        # need_send is set to 1; ready is never set -> deadline hit at once.
        with pytest.raises(RuntimeError, match="Timed out waiting for prefill blocks"):
            worker._send_kv_to_decode(meta)
        # The pending send meta was registered but never armed.
        assert "t-agent" in worker.reqs_need_send.reqs
    finally:
        worker.shutdown()


def test_worker_get_conn_caches_sessions(worker_env):
    """_get_conn memoizes the engine connection per session string under the
    connection lock."""
    worker = worker_env.make_worker(role="kv_producer")
    try:
        worker.flagcx.flagcxP2pGetConn = lambda engine, session: f"conn:{session}"

        first = worker._get_conn("h1:100")
        second = worker._get_conn("h1:100")
        other = worker._get_conn("h2:200")
        assert first == "conn:h1:100"
        assert second is first  # cached
        assert other == "conn:h2:200"
        assert worker.session_conns == {"h1:100": first, "h2:200": other}
    finally:
        worker.shutdown()


def test_worker_build_transfer_params_empty_remote_skips(worker_env):
    """_build_transfer_params skips requests whose remote block lists are
    all empty and produces no write descriptors."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        send_meta = fc.SendBlockMeta(
            p_req_id="p1",
            transfer_id="t1",
            local_block_ids=[[1, 2]],
            ready=threading.Event(),
        )
        agent_meta = _agent_meta(fc, req_blocks={"d1": ("t1", [[]])})
        src, dst, sizes = worker._build_transfer_params(
            [("d1", send_meta)], agent_meta, [], []
        )
        assert (src, dst, sizes) == ([], [], [])
    finally:
        worker.shutdown()


def test_worker_build_transfer_params_group_count_mismatch_skips(worker_env):
    """A local/remote KV-group count mismatch logs and skips the request."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        send_meta = fc.SendBlockMeta(
            p_req_id="p1",
            transfer_id="t1",
            local_block_ids=[[1]],
            ready=threading.Event(),
        )
        agent_meta = _agent_meta(fc, req_blocks={"d1": ("t1", [[1], [2]])})
        src, dst, sizes = worker._build_transfer_params(
            [("d1", send_meta)], agent_meta, [], []
        )
        assert (src, dst, sizes) == ([], [], [])
    finally:
        worker.shutdown()


def test_worker_build_transfer_params_local_shorter_skips(worker_env):
    """Fewer local blocks than remote blocks logs and skips the request."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        send_meta = fc.SendBlockMeta(
            p_req_id="p1",
            transfer_id="t1",
            local_block_ids=[[1]],
            ready=threading.Event(),
        )
        agent_meta = _agent_meta(fc, req_blocks={"d1": ("t1", [[1, 2, 3]])})
        src, dst, sizes = worker._build_transfer_params(
            [("d1", send_meta)], agent_meta, [], []
        )
        assert (src, dst, sizes) == ([], [], [])
    finally:
        worker.shutdown()


def test_worker_build_transfer_params_trims_partial_hit_tail(worker_env):
    """More local than remote blocks trims the local *tail* to the remote
    count before descriptors are built, so only the trailing local blocks
    reach the write lists."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        class _Topo:
            local_replicates_kv_cache = False
            is_kv_layout_blocks_first = False

            def handshake_target_ranks(self, remote_tp_size):
                return [0]

        worker.kv_topo = _Topo()
        worker.tp_rank = 0
        worker.tp_size = 1
        # Local group holds 3 blocks, the remote hit only 2: the partial-hit
        # tail trim keeps the last two local blocks ([2, 3]), not the head.
        send_meta = fc.SendBlockMeta(
            p_req_id="p1",
            transfer_id="t1",
            local_block_ids=[[1, 2, 3]],
            ready=threading.Event(),
        )
        agent_meta = _agent_meta(fc, req_blocks={"d1": ("t1", [[9, 10]])})
        local = fc.TransferRegion("L", 0, 1000, 64, 64, group_index=0)
        remote = fc.TransferRegion("L", 0, 5000, 64, 64, group_index=0)
        src, dst, sizes = worker._build_transfer_params(
            [("d1", send_meta)], agent_meta, [local], [remote]
        )
        # Trimmed tail [2, 3] coalesces into one contiguous write of 2 blocks
        # mapped onto the remote blocks [9, 10].
        assert src == [1000 + 2 * 64]
        assert dst == [5000 + 9 * 64]
        assert sizes == [64 * 2]
        # The caller's send meta is never mutated by the trim.
        assert send_meta.local_block_ids == [[1, 2, 3]]
    finally:
        worker.shutdown()


def test_worker_send_kv_alignment_failure_raises(worker_env):
    """_send_kv_to_decode fails fast when remote regions cannot be aligned
    with the local ones."""
    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_producer")
    try:
        worker._abort_request_timeout = 5

        class _Topo:
            local_replicates_kv_cache = False
            is_kv_layout_blocks_first = False

            def handshake_target_ranks(self, remote_tp_size):
                return [0]

        worker.kv_topo = _Topo()
        # Register an armed local send and matching remote request blocks.
        send_meta = fc.SendBlockMeta(
            p_req_id="p1",
            transfer_id="t-align",
            local_block_ids=[[1]],
            ready=threading.Event(),
        )
        send_meta.ready.set()
        send_meta.need_send = 1
        worker.reqs_need_send.reqs["t-align"] = send_meta
        # Give the worker one local region so local_regions is non-empty...
        worker.kv_caches_base_addr = [1000]
        worker.block_len_per_layer = [64]
        worker.kv_block_len_per_layer = [32]
        worker.registered_layer_names = ["model.layers.0.self_attn.attn"]
        worker.registered_layer_indices = [0]
        worker.registered_group_indices = [0]
        # ...but the remote side advertises a *different* layer, which has no
        # matching consumer occurrence -> alignment error.
        meta = _agent_meta(
            fc,
            req_blocks={"d1": ("t-align", [[1]])},
            registered_layer_names=["model.layers.9.self_attn.attn"],
            registered_layer_indices=[9],
            registered_group_indices=[0],
            block_lens=[64],
            kv_block_lens=[32],
            kv_caches_base_addr=[5000],
        )
        with pytest.raises(RuntimeError, match="no matching consumer occurrence"):
            worker._send_kv_to_decode(meta)
    finally:
        worker.shutdown()


def test_worker_group_kv_pull_fans_out_per_rank(worker_env):
    """_group_kv_pull registers pull-pending counts per request and groups
    the requests by the prefill rank paths."""
    import asyncio

    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_consumer")
    try:

        class _Topo:
            local_replicates_kv_cache = False
            is_kv_layout_blocks_first = False

            def handshake_target_ranks(self, remote_tp_size):
                return list(range(remote_tp_size))

        worker.kv_topo = _Topo()
        recv_meta = fc.RecvReqMeta(
            local_block_ids=[[1]],
            remote_host="10.0.0.1",
            remote_port=1234,
            transfer_id="t-pull",
            remote_tp_size=2,
        )
        launched = []

        async def fake_receive(path, req_blocks):
            launched.append((path, req_blocks))

        worker._receive_kv = fake_receive

        async def run():
            await worker._group_kv_pull({"d1": recv_meta})

        asyncio.run(run())
        # Two prefill ranks -> two paths, both carrying the same request.
        assert len(launched) == 2
        paths = {path for path, _ in launched}
        assert paths == {"tcp://10.0.0.1:1234", "tcp://10.0.0.1:1235"}
        assert worker._pull_pending == {"d1": 2}
    finally:
        worker.shutdown()


def test_worker_receive_kv_marks_finished_after_done(worker_env):
    """A TRANS_DONE reply marks the request finished; the pending count
    multi-rank pulls decrement correctly."""
    import asyncio

    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_consumer")
    try:
        worker._pull_pending["d1"] = 2

        class _FakeSock:
            async def send(self, data):
                worker._sent = data

            async def recv(self):
                return fc.TRANS_DONE

            def setsockopt(self, *a):
                pass

            def close(self):
                pass

        # msgspec's C encoder is immutable; swap the whole object.
        worker._encoder = type(
            "E", (), {"encode": staticmethod(lambda meta: b"meta")}
        )()
        fc.make_zmq_socket = lambda *a, **k: _FakeSock()

        async def run():
            await worker._receive_kv("tcp://x:1", {"d1": ("t1", [[1]])})

        asyncio.run(run())
        assert worker._pull_pending == {"d1": 1}
        assert "d1" not in worker.finished_recving_reqs.set

        # Second completion drains the pending count and finishes the req.
        async def run2():
            await worker._receive_kv("tcp://x:2", {"d1": ("t1", [[1]])})

        asyncio.run(run2())
        assert worker._pull_pending == {}
        assert worker.finished_recving_reqs.set == {"d1"}
    finally:
        worker.shutdown()


def test_worker_receive_kv_error_reply_records_failure(worker_env):
    """A non-DONE reply records a failed recv and does not finish the req."""
    import asyncio

    fc = worker_env.fc
    worker = worker_env.make_worker(role="kv_consumer")
    try:
        worker._pull_pending["d2"] = 1

        class _FakeSock:
            async def send(self, data):
                pass

            async def recv(self):
                return fc.TRANS_ERROR

            def setsockopt(self, *a):
                pass

            def close(self):
                pass

        worker._encoder = type(
            "E", (), {"encode": staticmethod(lambda meta: b"meta")}
        )()
        fc.make_zmq_socket = lambda *a, **k: _FakeSock()

        async def run():
            await worker._receive_kv("tcp://x:1", {"d2": ("t2", [[1]])})

        asyncio.run(run())
        assert worker.xfer_stats.data["num_failed_recvs"] == [1]
        assert "d2" not in worker.finished_recving_reqs.set
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# Agent metadata wire format (msgspec)
# ---------------------------------------------------------------------------


def test_agent_metadata_msgspec_roundtrip(worker_env):
    """FlagCXAgentMetadata survives a msgpack round-trip with tuples decoded
    as lists."""
    fc = worker_env.fc
    meta = _agent_meta(
        fc,
        req_blocks={"d1": ("t1", [[1, 2], [3]])},
        kv_caches_base_addr=[100, 200],
        block_lens=[64, 64],
        kv_block_lens=[32, 32],
        registered_layer_names=["a", "b"],
        registered_layer_indices=[0, 1],
        registered_group_indices=[0, 1],
    )
    encoder = fc.msgspec.msgpack.Encoder()
    decoder = fc.msgspec.msgpack.Decoder(fc.FlagCXAgentMetadata)
    restored = decoder.decode(encoder.encode(meta))
    assert restored.remote_hostname == "10.0.0.1"
    assert restored.remote_port == 1234
    assert restored.remote_tp_size == 1
    assert restored.remote_tp_rank == 0
    assert restored.req_blocks == {"d1": ("t1", [[1, 2], [3]])}
    assert restored.kv_caches_base_addr == [100, 200]
    assert restored.registered_group_indices == [0, 1]

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
"""Worker-side coverage tests for flagcx_connector.py.

FlagCXConnectorWorker owns the P2P engine lifecycle, per-session
connection memoization, the background sender executor / receiver event
loop chosen by KV role, the pending-send bookkeeping with expiry sweep,
and transfer-stat collection. The flagcx boundary is faked with the
recording wheel facade (R4: every expected value and recorded call
argument is derived from the inputs); a one-rank gloo TP group supplies
the parallel-state rank/size the worker reads.
"""

from __future__ import annotations

import time

import pytest

from . import _flagcx_env, _flagcx_helpers as helpers

_flagcx_env.ensure_importable_env()
_flagcx_env.ensure_vllm_api_compat()

import torch.distributed as dist  # noqa: E402

import vllm.distributed.parallel_state as ps  # noqa: E402
from vllm.distributed import (  # noqa: E402  (after env setup)
    destroy_model_parallel,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import (  # noqa: E402
    init_model_parallel_group,
)

_PORT = {"n": 0}


def _init_tp():
    """Start a fresh one-rank gloo TP group; return its teardown."""
    _PORT["n"] += 1
    init_distributed_environment(
        world_size=1,
        rank=0,
        backend="gloo",
        distributed_init_method=f"tcp://127.0.0.1:{29900 + _PORT['n']}",
    )
    tp = init_model_parallel_group([[0]], 0, "gloo", group_name="tp")
    prev = ps._TP
    ps._TP = tp

    def teardown():
        ps._TP = prev
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

    return teardown


@pytest.fixture()
def worker_env(monkeypatch):
    """Import the connector under the fake wheel and give it a TP group.

    Yields a builder for FlagCXConnectorWorker plus the call recorder;
    the teardown undoes the TP group, drops the module cache, and shuts
    every worker the tests built.
    """
    recorder = _flagcx_env.install_fake_flagcx_wheel(monkeypatch)
    _flagcx_env.drop_connector_module()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as conn

    teardown_tp = _init_tp()
    layer = helpers.FakeAttnLayer(helpers.FakeAttnBackend)
    workers: list = []

    def build(role="kv_consumer", num_workers=None, groups=None,
              tp_size=1, total_kv_heads=2, block_size=16, use_mla=False):
        """Construct a worker for ``role`` against faked configs."""
        if groups is None:
            groups = [helpers.full_attention_group(0, block_size=block_size)]
        extra = {"num_workers": num_workers} if num_workers else {}
        cfg = helpers.vllm_config(
            model=helpers.ModelInfo(
                use_mla=use_mla, total_num_kv_heads=total_kv_heads
            ),
            parallel=helpers.ParallelInfo(tp_size=tp_size),
            transfer=helpers.TransferInfo(
                kv_role=role, extra_config=extra
            ),
            cache=helpers.CacheInfo(block_size=block_size),
            compilation=helpers.CompilationInfo(
                {"model.layers.0.self_attn": layer}
            ),
        )
        w = conn.FlagCXConnectorWorker(
            cfg, "engw", helpers.FakeKVCacheConfig(groups)
        )
        workers.append(w)
        return w

    yield build, conn, recorder

    for w in workers:
        w.shutdown()
    teardown_tp()
    _flagcx_env.drop_connector_module()


def test_worker_init_role_threads_and_engine(worker_env):
    """__init__ creates one P2P engine through the wheel facade, binds the
    backend, and starts only the background machinery the role needs:
    kv_consumer gets a running receiver loop, kv_producer a sender
    executor and no receiver at all."""
    build, conn, recorder = worker_env

    w = build("kv_consumer")
    assert w.engine_id == "engw"
    assert w.kv_role == "kv_consumer"
    # One engine created via the facade; the wheel path passes None for
    # library_path and the port is unset until register_kv_caches runs.
    assert ("FLAGCXLibrary", None) in recorder
    assert ("flagcxP2pEngineCreate",) in recorder
    assert w.rpc_port == 0
    assert w.backend_name == "FAKE_ATTN"
    assert w.num_workers == 10
    assert w.session_conns == {}
    assert w.reqs_need_send.reqs == {}
    # Consumer side: receiver thread with a live event loop.
    assert w.receiver_loop.is_running()
    assert not hasattr(w, "_sender_executor")
    assert not hasattr(w, "_sender_t")

    p = build("kv_producer")
    assert p.kv_role == "kv_producer"
    assert not hasattr(p, "receiver_loop")
    assert p._sender_executor._max_workers == 10
    # Consumer and producer still share the one engine per worker.
    assert ("flagcxP2pEngineCreate",) in recorder


def test_worker_init_num_workers_override_and_layers(worker_env):
    """kv_connector_extra_config overrides the default worker-thread count;
    the layer bookkeeping maps every group layer to its spec and index."""
    build, conn, recorder = worker_env

    groups = [
        helpers.full_attention_group(0),
        helpers.sliding_window_group(1),
    ]
    w = build("kv_producer", num_workers=3, groups=groups)
    assert w.num_workers == 3
    assert w._sender_executor._max_workers == 3
    assert w._layer_group_indices == {
        "model.layers.0.self_attn": 0,
        "model.layers.1.self_attn": 1,
    }
    # Each layer resolves to its own group's spec object.
    assert (
        w._layer_specs["model.layers.0.self_attn"]
        is groups[0].kv_cache_spec
    )
    assert (
        w._layer_specs["model.layers.1.self_attn"]
        is groups[1].kv_cache_spec
    )
    # The registered-region lists start empty and are filled by
    # register_kv_caches.
    assert w.registered_layer_names == []
    assert w.kv_caches_base_addr == []


def test_worker_get_conn_memoizes_sessions(worker_env):
    """_get_conn opens one engine connection per "host:port" session and
    returns the same handle on every subsequent call for that session."""
    build, conn, recorder = worker_env

    w = build("kv_consumer")
    first = w._get_conn("10.0.0.5:4242")
    second = w._get_conn("10.0.0.5:4242")
    third = w._get_conn("10.0.0.6:4242")
    assert first is second
    assert third is not first
    assert list(w.session_conns) == ["10.0.0.5:4242", "10.0.0.6:4242"]
    # The facade saw exactly one GetConn per session, carrying the
    # session string itself.
    get_conns = [c for c in recorder if c[0] == "flagcxP2pGetConn"]
    assert get_conns == [
        ("flagcxP2pGetConn", "10.0.0.5:4242"),
        ("flagcxP2pGetConn", "10.0.0.6:4242"),
    ]


def test_start_load_kv_arms_producer_sends(worker_env):
    """On the producer, start_load_kv creates a pending SendBlockMeta per
    transfer: committed block ids arm it (ready set, finite deadline),
    empty ones stay unarmed with an infinite deadline."""
    build, conn, recorder = worker_env

    w = build("kv_producer")
    meta = conn.FlagCXConnectorMetadata()
    meta.reqs_to_send["pr1"] = ("tr1", [[1, 2], [3]])
    meta.reqs_to_send["pr2"] = ("tr2", [])
    w.start_load_kv(meta)

    sm = w.reqs_need_send.reqs["tr1"]
    assert sm.p_req_id == "pr1"
    assert sm.local_block_ids == [[1, 2], [3]]
    assert sm.ready.is_set()
    assert sm.expire_time <= time.perf_counter() + 480
    assert sm.need_send == 0

    unarmed = w.reqs_need_send.reqs["tr2"]
    assert unarmed.p_req_id == "pr2"
    assert unarmed.local_block_ids == []
    assert not unarmed.ready.is_set()
    assert unarmed.expire_time == float("inf")

    # Re-arming refreshes an existing meta's request id instead of
    # replacing the object.
    again = conn.FlagCXConnectorMetadata()
    again.reqs_to_send["pr1"] = ("tr1", [[7]])
    w.start_load_kv(again)
    assert w.reqs_need_send.reqs["tr1"] is sm
    assert sm.local_block_ids == [[7]]


def test_get_finished_expires_stale_sends(worker_env):
    """get_finished sweeps pending sends whose deadline passed: the
    transfer is dropped, an expiry is recorded in the stats, and the
    producer request id surfaces in finished_sending."""
    build, conn, recorder = worker_env

    w = build("kv_producer")
    meta = conn.FlagCXConnectorMetadata()
    meta.reqs_to_send["pr1"] = ("tr1", [[1]])
    meta.reqs_to_send["pr2"] = ("tr2", [[2]])
    w.start_load_kv(meta)

    fresh = w.reqs_need_send.reqs["tr1"]
    stale = w.reqs_need_send.reqs["tr2"]
    stale.expire_time = time.perf_counter() - 1

    finished_sending, finished_recving = w.get_finished()
    assert finished_sending == {"pr2"}
    assert finished_recving is None
    assert "tr2" not in w.reqs_need_send.reqs
    assert "tr1" in w.reqs_need_send.reqs
    assert w.xfer_stats.data["num_kv_expired_reqs"] == [1]
    assert fresh.expire_time > time.perf_counter()

    # A second sweep with nothing expired reports no sends and leaves
    # the pending entry alone.
    assert w.get_finished() == (None, None)
    assert "tr1" in w.reqs_need_send.reqs


def test_get_finished_consumer_returns_recving(worker_env):
    """The consumer side of get_finished drains the receiver-loop result
    (empty when no pull completed) and never reports sends."""
    build, conn, recorder = worker_env

    w = build("kv_consumer")
    meta = conn.FlagCXConnectorMetadata()
    meta.reqs_to_send["pr9"] = ("tr9", [[9]])
    # Consumers ignore the send table entirely.
    w.start_load_kv(meta)
    assert w.reqs_need_send.reqs == {}

    finished_sending, finished_recving = w.get_finished()
    assert finished_sending is None
    assert finished_recving is None


def test_get_kv_connector_stats_snapshot(worker_env):
    """get_kv_connector_stats returns None while nothing was recorded and
    a clone-and-reset snapshot once something was."""
    build, conn, recorder = worker_env

    w = build("kv_producer")
    assert w.get_kv_connector_stats() is None

    w.xfer_stats.record_transfer(0.5, 2**20, 3)
    snap = w.get_kv_connector_stats()
    assert snap.data["transfer_duration"] == [0.5]
    assert snap.data["bytes_transferred"] == [2**20]
    assert snap.data["num_descriptors"] == [3]
    # The snapshot drained the container.
    assert w.get_kv_connector_stats() is None


def test_producer_cache_is_replicated(worker_env):
    """_producer_cache_is_replicated mirrors TransferTopology: replicated
    when the model is MLA or the TP group replicates the KV heads. The
    worker reads the group from parallel state (a one-rank TP group
    here), so the sharded case is a plain full-attention model and the
    replicated case an MLA one."""
    build, conn, recorder = worker_env

    sharded = build("kv_producer", total_kv_heads=2)
    assert sharded.tp_size == 1
    assert sharded._producer_cache_is_replicated() is False

    mla = build(
        "kv_producer", total_kv_heads=2, use_mla=True
    )
    assert mla._producer_cache_is_replicated() is True


def test_shutdown_destroys_engine_and_threads(worker_env):
    """shutdown tears down the ZMQ contexts, stops the background threads,
    and destroys the P2P engine exactly once."""
    build, conn, recorder = worker_env

    w = build("kv_consumer")
    assert w.engine is not None
    loop = w.receiver_loop
    w.shutdown()
    assert w.engine is None
    assert not loop.is_running()
    destroys = [c for c in recorder if c[0] == "flagcxP2pEngineDestroy"]
    assert len(destroys) == 1
    # The fixture's shutdown for an already-torn-down worker must not
    # raise or double-destroy.
    w.shutdown()
    destroys = [c for c in recorder if c[0] == "flagcxP2pEngineDestroy"]
    assert len(destroys) == 1


def test_logical_to_kernel_block_ids_passthrough(worker_env):
    """With the default one-kernel-block-per-logical-block mapping, block
    id lists pass through unchanged (identity per group)."""
    build, conn, recorder = worker_env

    w = build("kv_consumer")
    assert w._physical_blocks_per_logical_kv_block == 1
    ids = [[3, 4], [7]]
    assert w._logical_to_kernel_block_ids(ids) == [[3, 4], [7]]
    assert w._logical_to_kernel_block_ids([]) == []

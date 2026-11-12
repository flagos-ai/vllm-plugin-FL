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
"""Consumer-side KV pull coverage test for flagcx_connector.py.

Exercises the full D-side pull path on CPU: start_load_kv hands each
recv request to the receiver loop, the pull is addressed to the
Prefill side channel at remote_port + rank, the wire metadata carries
the request's block ids and transfer id, and a TRANS_DONE reply moves
the request into finished_recving. A local REP socket stands in for
the Prefill side channel (the network boundary is a sanctioned patch
target).
"""

from __future__ import annotations

import asyncio
import sys
import time

import msgspec
import pytest

from . import _flagcx_env
from . import _flagcx_helpers as helpers

# zmq.asyncio needs a selector-based loop; on Windows the default is
# Proactor, which zmq cannot multiplex. Linux (the CI image) already
# uses epoll, so this only affects local runs.
if sys.platform == "win32":  # pragma: no cover - platform guard
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_flagcx_env.ensure_importable_env()
_flagcx_env.ensure_vllm_api_compat()

from vllm.distributed import (  # noqa: E402  (after env setup)
    destroy_model_parallel,
    init_distributed_environment,
)
import vllm.distributed.parallel_state as ps  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    init_model_parallel_group,
)
import torch.distributed as dist  # noqa: E402


_PORT = {"n": 0}


def _init_tp():
    """Start a fresh one-rank gloo TP group; return its teardown."""
    _PORT["n"] += 1
    init_distributed_environment(
        world_size=1,
        rank=0,
        backend="gloo",
        distributed_init_method=f"tcp://127.0.0.1:{29950 + _PORT['n']}",
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

    Yields a builder for a kv_consumer worker plus the call recorder;
    the teardown undoes the TP group, drops the module cache, and shuts
    every worker the tests built.
    """
    recorder = _flagcx_env.install_fake_flagcx_wheel(monkeypatch)
    _flagcx_env.drop_connector_module()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as conn

    teardown_tp = _init_tp()
    layer = helpers.FakeAttnLayer(helpers.FakeAttnBackend)
    workers: list = []

    def build(role="kv_consumer"):
        """Construct a kv_consumer worker against faked configs."""
        cfg = helpers.vllm_config(
            model=helpers.ModelInfo(use_mla=False, total_num_kv_heads=2),
            parallel=helpers.ParallelInfo(tp_size=1),
            transfer=helpers.TransferInfo(kv_role=role),
            compilation=helpers.CompilationInfo(
                {"model.layers.0.self_attn": layer}
            ),
        )
        w = conn.FlagCXConnectorWorker(
            cfg, "engw", helpers.FakeKVCacheConfig(
                [helpers.full_attention_group(0)]
            )
        )
        workers.append(w)
        return w

    yield build, conn, recorder

    for w in workers:
        w.shutdown()
    teardown_tp()
    _flagcx_env.drop_connector_module()


def test_start_load_kv_consumer_pull_roundtrip(worker_env):
    """On the consumer, start_load_kv fans each recv request out to the
    receiver loop: the pull is addressed to remote_port + rank, carries
    the request's block ids and transfer id on the wire, and a
    TRANS_DONE reply moves the request into finished_recving."""
    import zmq

    build, conn, recorder = worker_env

    # A local REP peer stands in for the Prefill side channel.
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.bind("tcp://127.0.0.1:0")
    port = int(rep.getsockopt(zmq.LAST_ENDPOINT).rsplit(b":", 1)[1])
    rep.setsockopt(zmq.RCVTIMEO, 10000)

    w = build("kv_consumer")
    meta = conn.FlagCXConnectorMetadata()
    meta.reqs_to_recv["cr1"] = conn.RecvReqMeta(
        local_block_ids=[[1, 2]],
        remote_host="127.0.0.1",
        remote_port=port,
        transfer_id="tcr1",
        remote_tp_size=1,
    )
    w.start_load_kv(meta)

    # The pull runs on the receiver loop; wait for it to be scheduled.
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline and "cr1" not in w._pull_pending:
        time.sleep(0.01)
    # The pull target is host:(port + rank) for each handshake rank.
    assert w._pull_pending["cr1"] == 1
    blob = rep.recv()
    agent_meta = msgspec.msgpack.decode(blob, type=conn.FlagCXAgentMetadata)
    assert agent_meta.req_blocks["cr1"] == ("tcr1", [[1, 2]])
    assert agent_meta.remote_tp_size == 1
    rep.send(conn.TRANS_DONE)
    rep.close()
    ctx.term()

    # After the reply the request finishes recving (bounded wait for
    # the receiver-loop coroutine to commit).
    deadline = time.perf_counter() + 10
    while time.perf_counter() < deadline:
        finished_sending, finished_recving = w.get_finished()
        if finished_recving:
            break
    assert finished_sending is None
    assert finished_recving == {"cr1"}
    assert "cr1" not in w._pull_pending
    # The dispatch never touched the send table.
    assert w.reqs_need_send.reqs == {}

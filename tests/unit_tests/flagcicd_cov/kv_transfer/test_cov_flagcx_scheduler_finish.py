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
"""Request-finish coverage tests for the FlagCX scheduler side.

request_finished decides whether a finishing request hands its KV off
to the peer engine (P-side decode handoff dict with endpoint, TP size
and transfer id; delay-free while blocks are still queued) or stays
local (D-side remote-prefill queueing).
"""

from __future__ import annotations

from . import _flagcx_env, _flagcx_helpers as helpers

_flagcx_env.ensure_importable_env()

import pytest

from vllm.v1.request import RequestStatus  # noqa: E402  (after env setup)


@pytest.fixture()
def conn(monkeypatch):
    """Import the connector against the fake flagcx wheel facade."""
    _flagcx_env.install_fake_flagcx_wheel(monkeypatch)
    _flagcx_env.ensure_vllm_api_compat()
    _flagcx_env.drop_connector_module()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as conn

    yield conn
    _flagcx_env.drop_connector_module()


def _scheduler(conn, role="kv_consumer", groups=None, has_mamba=False,
               hma=False, tp_size=2):
    """Build a scheduler with the given role and cache groups."""
    if groups is None:
        groups = [helpers.full_attention_group(0)]
    cfg = helpers.vllm_config(
        transfer=helpers.TransferInfo(kv_role=role),
        parallel=helpers.ParallelInfo(tp_size=tp_size),
        scheduler=helpers.SchedulerInfo(
            disable_hybrid_kv_cache_manager=not hma
        ),
    )
    return conn.FlagCXConnectorScheduler(
        cfg, "eng-sched", helpers.FakeKVCacheConfig(groups, has_mamba)
    )


def test_request_finished_no_transfer_id(conn):
    """Requests without params or a transfer id finish immediately."""
    sched = _scheduler(conn, role="kv_producer")
    no_params = helpers.FakeRequest(request_id="f0", kv_transfer_params=None)
    assert sched.request_finished(no_params, ([1],)) == (False, None)

    no_tid = helpers.FakeRequest(
        request_id="f1", kv_transfer_params={"do_remote_decode": True}
    )
    assert sched.request_finished(no_tid, ([1],)) == (False, None)


def test_request_finished_remote_prefill_queues_recv(conn):
    """A D-side request finishing with remote prefill pending is queued
    for receive instead of being handed off."""
    sched = _scheduler(conn, role="kv_consumer")
    req = helpers.FakeRequest(
        request_id="f2",
        kv_transfer_params={
            "transfer_id": "tf2",
            "do_remote_prefill": True,
        },
    )
    assert sched.request_finished(req, ([],)) == (False, None)
    assert list(sched._reqs_need_recv) == ["f2"]
    assert sched._reqs_need_recv["f2"][1] == []
    assert req.kv_transfer_params["do_remote_prefill"] is False


def test_request_finished_decode_handoff(conn):
    """A length-capped P-side request with blocks hands off a prefill
    handoff dict carrying the endpoint, TP size and transfer id, and is
    queued for send with its clipped blocks."""
    sched = _scheduler(conn, role="kv_producer")
    req = helpers.FakeRequest(
        request_id="f3",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params={"transfer_id": "tf3", "do_remote_decode": True},
    )
    delay_free, handoff = sched.request_finished(req, ([7, 8],))
    assert delay_free is True
    assert handoff == {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_host": sched.side_channel_host,
        "remote_port": sched.side_channel_port,
        "remote_tp_size": 2,
        "transfer_id": "tf3",
    }
    assert sched._reqs_need_send["f3"][1] == [[7, 8]]

    # Empty block groups still hand off the params but free immediately.
    req_empty = helpers.FakeRequest(
        request_id="f4",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params={"transfer_id": "tf4", "do_remote_decode": True},
    )
    delay4, handoff4 = sched.request_finished(req_empty, ([],))
    assert delay4 is False
    assert handoff4["transfer_id"] == "tf4"
    assert "f4" not in sched._reqs_need_send


def test_request_finished_non_length_capped_decode(conn):
    """Decode requests that finished for other reasons never hand off."""
    sched = _scheduler(conn, role="kv_producer")
    stopped = helpers.FakeRequest(
        request_id="f5",
        status=RequestStatus.FINISHED_STOPPED,
        kv_transfer_params={"transfer_id": "tf5", "do_remote_decode": True},
    )
    assert sched.request_finished(stopped, ([1],)) == (False, None)
    assert sched._reqs_need_send == {}

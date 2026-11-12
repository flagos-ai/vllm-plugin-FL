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
"""Scheduler-side coverage tests for flagcx_connector.py.

FlagCXConnectorScheduler tracks which requests need KV to be received
from (D) or sent to (P) the remote engine and turns the scheduler's
per-step bookkeeping into FlagCXConnectorMetadata. Every expected value
below is derived from the inputs.
"""

from __future__ import annotations

import pytest

from . import _flagcx_env, _flagcx_helpers as helpers

_flagcx_env.ensure_importable_env()



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


def test_scheduler_init_reads_role_and_port(conn, monkeypatch):
    """The scheduler records the engine id, its KV role, and derives the
    side-channel port from dp_rank * tp_size on top of the bootstrap base."""
    monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)
    sched = _scheduler(conn, role="kv_consumer", tp_size=4)
    assert sched.engine_id == "eng-sched"
    assert sched.kv_role == "kv_consumer"
    assert sched.block_size == 16
    assert sched.side_channel_port == 8998
    assert sched._reqs_need_recv == {}
    assert sched._reqs_need_send == {}


def test_scheduler_hma_detection_and_blocks_per_sw(conn):
    """HMA is required exactly when the manager is enabled and a group
    carries a non-full-attention spec; blocks_per_sw sizes each group's
    sliding window (cdiv(window, block) + 1), 0 for full attention."""
    groups = [helpers.full_attention_group(0),
              helpers.sliding_window_group(1, sliding_window=32)]
    hma_on = _scheduler(conn, role="kv_consumer", groups=groups, hma=True)
    assert hma_on._is_hma_required is True
    assert hma_on.blocks_per_sw == [0, 3]

    hma_off = _scheduler(conn, role="kv_consumer", groups=groups, hma=False)
    assert hma_off._is_hma_required is False
    assert hma_off.blocks_per_sw == [0, 3]

    full_only = _scheduler(
        conn, role="kv_consumer",
        groups=[helpers.full_attention_group(0)], hma=True,
    )
    assert full_only._is_hma_required is False


def test_get_sw_clipped_blocks(conn):
    """With HMA required, each group's blocks are clipped to the last
    blocks_per_sw entries; a zero entry keeps the whole group."""
    groups = [helpers.full_attention_group(0),
              helpers.sliding_window_group(1, sliding_window=32)]
    sched = _scheduler(conn, groups=groups, hma=True)
    clipped = sched.get_sw_clipped_blocks([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    # Group 0 (full attention): blocks_per_sw == 0 -> untouched.
    assert clipped[0] == [1, 2, 3, 4, 5]
    # Group 1 (32-token window over 16-token blocks -> 3 blocks).
    assert clipped[1] == [8, 9, 10]

    # Without HMA the ids pass through untouched (still copied).
    plain = _scheduler(conn, hma=False)
    assert plain.get_sw_clipped_blocks([[1, 2], [3]]) == [[1, 2], [3]]
    assert plain.get_sw_clipped_blocks([]) == []


def test_remote_prefill_token_count_mamba_subtracts_one(conn):
    """D-side mamba models remote-prefill N-1 tokens (the last prompt
    token is recomputed locally); non-mamba passes N through."""
    mamba = _scheduler(
        conn, groups=[helpers.mamba_group(0)], has_mamba=True
    )
    assert mamba._get_remote_prefill_token_count(5) == 4
    # A single-token prompt is not reduced (would be zero).
    assert mamba._get_remote_prefill_token_count(1) == 1

    plain = _scheduler(conn)
    assert plain._get_remote_prefill_token_count(5) == 5


def test_truncate_mamba_request_pops_last_token(conn):
    """P-side mamba truncation drops the final prompt token once,
    flips the guard, and forces max_tokens to 1."""
    sched = _scheduler(conn, groups=[helpers.mamba_group(0)], has_mamba=True)
    req = helpers.FakeRequest(
        request_id="p1",
        prompt_token_ids=[10, 11, 12, 13, 14],
        kv_transfer_params={"do_remote_decode": True},
    )
    sched._truncate_mamba_request_for_prefill(req)
    assert req.prompt_token_ids == [10, 11, 12, 13]
    assert req._all_token_ids == [10, 11, 12, 13]
    assert req.num_prompt_tokens == 4
    assert req.max_tokens == 1
    assert req.kv_transfer_params["_p_side_truncated"] is True

    # The guard keeps a second pass from truncating again.
    sched._truncate_mamba_request_for_prefill(req)
    assert req.prompt_token_ids == [10, 11, 12, 13]
    assert req.num_prompt_tokens == 4


def test_truncate_mamba_request_embeds_branch(conn):
    """Prompt embeds are truncated via a slice when token ids are absent."""
    import torch

    sched = _scheduler(conn, groups=[helpers.mamba_group(0)], has_mamba=True)
    req = helpers.FakeRequest(
        request_id="p2",
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(4, 3),
        kv_transfer_params={"do_remote_decode": True},
        num_prompt_tokens=4,
    )
    sched._truncate_mamba_request_for_prefill(req)
    assert req.prompt_embeds.shape == (3, 3)
    assert req.num_prompt_tokens == 3


def test_get_num_new_matched_tokens_remote_prefill(conn):
    """A remote-prefill request claims its remaining prompt tokens as an
    external hit; the mamba N-1 adjustment applies before subtraction."""
    plain = _scheduler(conn)
    req = helpers.FakeRequest(
        request_id="r1",
        prompt_token_ids=[1, 2, 3, 4, 5],
        kv_transfer_params={"do_remote_prefill": True},
    )
    assert plain.get_num_new_matched_tokens(req, 2) == (3, True)

    mamba = _scheduler(conn, groups=[helpers.mamba_group(0)], has_mamba=True)
    req_m = helpers.FakeRequest(
        request_id="r2",
        prompt_token_ids=[1, 2, 3, 4, 5],
        kv_transfer_params={"do_remote_prefill": True},
    )
    assert mamba.get_num_new_matched_tokens(req_m, 0) == (4, True)


def test_get_num_new_matched_tokens_fallthroughs(conn):
    """Requests without remote prefill, or fully-computed ones, return
    (0, False); a remote-decode mamba request gets truncated instead."""
    plain = _scheduler(conn)
    no_params = helpers.FakeRequest(request_id="r3", kv_transfer_params=None)
    assert plain.get_num_new_matched_tokens(no_params, 4) == (0, False)

    fully_computed = helpers.FakeRequest(
        request_id="r4",
        prompt_token_ids=[1, 2, 3],
        kv_transfer_params={"do_remote_prefill": True},
    )
    assert plain.get_num_new_matched_tokens(fully_computed, 3) == (0, False)

    decode_only = helpers.FakeRequest(
        request_id="r5", kv_transfer_params={"do_remote_decode": True}
    )
    assert plain.get_num_new_matched_tokens(decode_only, 0) == (0, False)

    # do_remote_decode + mamba triggers P-side truncation.
    mamba = _scheduler(conn, groups=[helpers.mamba_group(0)], has_mamba=True)
    req_m = helpers.FakeRequest(
        request_id="r6",
        prompt_token_ids=[1, 2, 3, 4],
        kv_transfer_params={"do_remote_decode": True},
    )
    assert mamba.get_num_new_matched_tokens(req_m, 0) == (0, False)
    assert req_m.num_prompt_tokens == 3


def test_update_state_after_alloc_recv_registration(conn):
    """A valid remote-prefill request is queued for receive with its
    unhashed per-group blocks, and the flag is consumed."""
    sched = _scheduler(conn, role="kv_consumer")
    req = helpers.FakeRequest(
        request_id="rA",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_host": "10.0.0.9",
            "remote_port": 4242,
            "transfer_id": "tA",
        },
    )
    sched.update_state_after_alloc(
        req, helpers.FakeBlocks([[1, 2, 3]]), num_external_tokens=10
    )
    assert list(sched._reqs_need_recv) == ["rA"]
    stored_req, stored_blocks = sched._reqs_need_recv["rA"]
    assert stored_req is req
    assert stored_blocks == [[1, 2, 3]]
    assert req.kv_transfer_params["do_remote_prefill"] is False


def test_update_state_after_alloc_recv_invalid_params(conn):
    """A remote-prefill request missing the endpoint fields is warned
    about and never queued, but the flag is still consumed."""
    sched = _scheduler(conn, role="kv_consumer")
    req = helpers.FakeRequest(
        request_id="rB",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_port": 4242,
            "transfer_id": "tB",
        },
    )
    sched.update_state_after_alloc(
        req, helpers.FakeBlocks([[1]]), num_external_tokens=5
    )
    assert sched._reqs_need_recv == {}
    assert req.kv_transfer_params["do_remote_prefill"] is False


def test_update_state_after_alloc_recv_zero_external_tokens(conn):
    """Zero external tokens queue the request with empty block ids."""
    sched = _scheduler(conn, role="kv_consumer")
    req = helpers.FakeRequest(
        request_id="rC",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_host": "h",
            "remote_port": 1,
            "transfer_id": "tC",
        },
    )
    sched.update_state_after_alloc(
        req, helpers.FakeBlocks([[7]]), num_external_tokens=0
    )
    assert sched._reqs_need_recv["rC"][1] == []


def test_update_state_after_alloc_send_registration(conn):
    """A remote-decode request with a transfer id is queued for send;
    without one it is warned about and dropped; no params is a no-op."""
    sched = _scheduler(conn, role="kv_producer")
    req = helpers.FakeRequest(
        request_id="rS",
        kv_transfer_params={"do_remote_decode": True, "transfer_id": "tS"},
    )
    sched.update_state_after_alloc(
        req, helpers.FakeBlocks([]), num_external_tokens=0
    )
    assert list(sched._reqs_need_send) == ["rS"]

    missing_id = helpers.FakeRequest(
        request_id="rS2", kv_transfer_params={"do_remote_decode": True}
    )
    sched.update_state_after_alloc(
        missing_id, helpers.FakeBlocks([]), num_external_tokens=0
    )
    assert "rS2" not in sched._reqs_need_send

    sched.update_state_after_alloc(
        helpers.FakeRequest(request_id="rN", kv_transfer_params=None),
        helpers.FakeBlocks([]),
        0,
    )
    assert set(sched._reqs_need_send) == {"rS"}


def test_build_connector_meta_role_filtering_and_drain(conn):
    """build_connector_meta drains the pending dicts into a metadata
    object, filtered by role, and leaves both dicts empty."""
    consumer = _scheduler(conn, role="kv_consumer")
    recv_req = helpers.FakeRequest(
        request_id="rcv",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_host": "10.0.0.1",
            "remote_port": 9000,
            "transfer_id": "tcv",
            "remote_tp_size": 2,
        },
    )
    consumer.update_state_after_alloc(
        recv_req, helpers.FakeBlocks([[4, 5]]), num_external_tokens=2
    )
    meta = consumer.build_connector_meta(None)
    assert set(meta.reqs_to_recv) == {"rcv"}
    assert meta.reqs_to_recv["rcv"].local_block_ids == [[4, 5]]
    assert meta.reqs_to_send == {}
    assert consumer._reqs_need_recv == {}

    producer = _scheduler(conn, role="kv_producer")
    send_req = helpers.FakeRequest(
        request_id="snd",
        kv_transfer_params={"do_remote_decode": True, "transfer_id": "tsd"},
    )
    producer.update_state_after_alloc(
        send_req, helpers.FakeBlocks([]), num_external_tokens=0
    )
    meta_p = producer.build_connector_meta(None)
    assert set(meta_p.reqs_to_send) == {"snd"}
    assert meta_p.reqs_to_send["snd"] == ("tsd", [])
    assert meta_p.reqs_to_recv == {}
    assert producer._reqs_need_send == {}

    # kv_producer never emits recv entries: force one into the dict and
    # confirm the role filter drops it.
    producer._reqs_need_recv["ghost"] = (recv_req, [[1]])
    assert "ghost" not in producer.build_connector_meta(None).reqs_to_recv

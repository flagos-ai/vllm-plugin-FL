# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side and public-facade coverage for the FlagCX connector.

The scheduler tracks which requests need KV received from (D) or sent to
(P) the remote engine and turns per-step bookkeeping into metadata.  The
public ``FlagCXConnector`` facade only dispatches to those halves.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import test_flagcx_connector_support as support


@pytest.fixture()
def conn(monkeypatch):
    module, recorder = support.load_connector(monkeypatch)
    support.reset_topology_state()
    yield module, recorder
    sys.modules.pop("vllm_fl.distributed.kv_transfer.flagcx_connector", None)


def _scheduler(conn, role="kv_consumer", groups=None, has_mamba=False, hma=False, tp_size=2):
    if groups is None:
        groups = [support.full_group(0)]
    cfg = support.vllm_config(
        transfer=support.TransferInfo(kv_role=role),
        parallel=support.ParallelInfo(tp_size=tp_size),
        scheduler=support.SchedulerInfo(disable_hybrid_kv_cache_manager=not hma),
    )
    return conn.FlagCXConnectorScheduler(
        cfg, "eng-sched", support.FakeKVCacheConfig(groups, has_mamba)
    )


def test_scheduler_init_reads_role_and_port(conn, monkeypatch):
    module, _ = conn
    monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)
    sched = _scheduler(module, role="kv_consumer", tp_size=4)
    assert sched.engine_id == "eng-sched"
    assert sched.kv_role == "kv_consumer"
    assert sched.block_size == 16
    assert sched.side_channel_host == "127.0.0.1"
    assert sched.side_channel_port == 8998
    assert sched._reqs_need_recv == {}
    assert sched._reqs_need_send == {}


def test_scheduler_hma_detection_and_blocks(conn):
    module, _ = conn
    groups = [
        support.full_group(0),
        support.sliding_group(1, sliding_window=32),
    ]
    on = _scheduler(module, groups=groups, hma=True)
    assert on._is_hma_required is True
    assert on.blocks_per_sw == [0, 3]

    off = _scheduler(module, groups=groups, hma=False)
    assert off._is_hma_required is False
    assert off.blocks_per_sw == [0, 3]

    full_only = _scheduler(module, groups=[support.full_group(0)], hma=True)
    assert full_only._is_hma_required is False


def test_get_sw_clipped_blocks(conn):
    module, _ = conn
    groups = [support.full_group(0), support.sliding_group(1, sliding_window=32)]
    sched = _scheduler(module, groups=groups, hma=True)
    clipped = sched.get_sw_clipped_blocks([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    assert clipped[0] == [1, 2, 3, 4, 5]
    assert clipped[1] == [8, 9, 10]

    plain = _scheduler(module, hma=False)
    assert plain.get_sw_clipped_blocks([[1, 2], [3]]) == [[1, 2], [3]]
    assert plain.get_sw_clipped_blocks([]) == []


def test_mamba_remote_prefill_token_count(conn):
    module, _ = conn
    mamba = _scheduler(module, groups=[support.mamba_group(0)], has_mamba=True)
    assert mamba._get_remote_prefill_token_count(5) == 4
    assert mamba._get_remote_prefill_token_count(1) == 1
    plain = _scheduler(module)
    assert plain._get_remote_prefill_token_count(5) == 5


def test_truncate_mamba_request_tokens_once(conn):
    module, _ = conn
    sched = _scheduler(module, groups=[support.mamba_group(0)], has_mamba=True)
    req = support.FakeRequest(
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

    sched._truncate_mamba_request_for_prefill(req)
    assert req.prompt_token_ids == [10, 11, 12, 13]


def test_truncate_mamba_request_embeds_branch(conn):
    module, _ = conn
    sched = _scheduler(module, groups=[support.mamba_group(0)], has_mamba=True)
    req = support.FakeRequest(
        request_id="p2",
        prompt_token_ids=None,
        prompt_embeds=[0.0] * 4,
        kv_transfer_params={"do_remote_decode": True},
        num_prompt_tokens=4,
    )
    req.prompt_embeds = [1.0, 2.0, 3.0, 4.0]
    sched._truncate_mamba_request_for_prefill(req)
    assert req.prompt_embeds == [1.0, 2.0, 3.0]
    assert req.num_prompt_tokens == 3


def test_truncate_mamba_request_guards(conn):
    module, _ = conn
    sched = _scheduler(module, groups=[support.mamba_group(0)], has_mamba=True)
    # No params, already truncated, single token, and no prompt data all
    # return without touching the request.
    for req in (
        support.FakeRequest(request_id="a", kv_transfer_params=None),
        support.FakeRequest(
            request_id="b",
            kv_transfer_params={"_p_side_truncated": True},
        ),
        support.FakeRequest(
            request_id="c",
            prompt_token_ids=[1],
            kv_transfer_params={"do_remote_decode": True},
        ),
        support.FakeRequest(
            request_id="d",
            prompt_token_ids=None,
            prompt_embeds=None,
            num_prompt_tokens=4,
            kv_transfer_params={"do_remote_decode": True},
        ),
    ):
        before = list(req.prompt_token_ids or [])
        sched._truncate_mamba_request_for_prefill(req)
        assert list(req.prompt_token_ids or []) == before


def test_get_num_new_matched_tokens_remote_prefill(conn):
    module, _ = conn
    plain = _scheduler(module)
    req = support.FakeRequest(
        request_id="r1",
        prompt_token_ids=[1, 2, 3, 4, 5],
        kv_transfer_params={"do_remote_prefill": True},
    )
    assert plain.get_num_new_matched_tokens(req, 2) == (3, True)

    mamba = _scheduler(module, groups=[support.mamba_group(0)], has_mamba=True)
    req_m = support.FakeRequest(
        request_id="r2",
        prompt_token_ids=[1, 2, 3, 4, 5],
        kv_transfer_params={"do_remote_prefill": True},
    )
    assert mamba.get_num_new_matched_tokens(req_m, 0) == (4, True)


def test_get_num_new_matched_tokens_fallthroughs(conn):
    module, _ = conn
    plain = _scheduler(module)
    assert (
        plain.get_num_new_matched_tokens(
            support.FakeRequest(request_id="r3", kv_transfer_params=None), 4
        )
        == (0, False)
    )
    fully = support.FakeRequest(
        request_id="r4",
        prompt_token_ids=[1, 2, 3],
        kv_transfer_params={"do_remote_prefill": True},
    )
    assert plain.get_num_new_matched_tokens(fully, 3) == (0, False)
    decode = support.FakeRequest(
        request_id="r5", kv_transfer_params={"do_remote_decode": True}
    )
    assert plain.get_num_new_matched_tokens(decode, 0) == (0, False)

    mamba = _scheduler(module, groups=[support.mamba_group(0)], has_mamba=True)
    trunc = support.FakeRequest(
        request_id="r6",
        prompt_token_ids=[1, 2, 3, 4],
        kv_transfer_params={"do_remote_decode": True},
    )
    assert mamba.get_num_new_matched_tokens(trunc, 0) == (0, False)
    assert trunc.num_prompt_tokens == 3


def test_update_state_after_alloc_recv(conn):
    module, _ = conn
    sched = _scheduler(module, role="kv_consumer")
    req = support.FakeRequest(
        request_id="rA",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_host": "10.0.0.9",
            "remote_port": 4242,
            "transfer_id": "tA",
        },
    )
    sched.update_state_after_alloc(
        req, support.FakeBlocks([[1, 2, 3]]), num_external_tokens=10
    )
    assert list(sched._reqs_need_recv) == ["rA"]
    assert sched._reqs_need_recv["rA"][0] is req
    assert sched._reqs_need_recv["rA"][1] == [[1, 2, 3]]
    assert req.kv_transfer_params["do_remote_prefill"] is False


def test_update_state_after_alloc_recv_zero_external_tokens(conn):
    module, _ = conn
    sched = _scheduler(module, role="kv_consumer")
    req = support.FakeRequest(
        request_id="rC",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_host": "h",
            "remote_port": 1,
            "transfer_id": "tC",
        },
    )
    sched.update_state_after_alloc(
        req, support.FakeBlocks([[7]]), num_external_tokens=0
    )
    assert sched._reqs_need_recv["rC"][1] == []


def test_update_state_after_alloc_recv_invalid(conn):
    module, _ = conn
    sched = _scheduler(module, role="kv_consumer")
    req = support.FakeRequest(
        request_id="rB",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_port": 4242,
            "transfer_id": "tB",
        },
    )
    sched.update_state_after_alloc(
        req, support.FakeBlocks([[1]]), num_external_tokens=5
    )
    assert sched._reqs_need_recv == {}
    assert req.kv_transfer_params["do_remote_prefill"] is False


def test_update_state_after_alloc_send_and_noop(conn):
    module, _ = conn
    sched = _scheduler(module, role="kv_producer")
    req = support.FakeRequest(
        request_id="rS",
        kv_transfer_params={"do_remote_decode": True, "transfer_id": "tS"},
    )
    sched.update_state_after_alloc(
        req, support.FakeBlocks([]), num_external_tokens=0
    )
    assert list(sched._reqs_need_send) == ["rS"]

    missing = support.FakeRequest(
        request_id="rS2", kv_transfer_params={"do_remote_decode": True}
    )
    sched.update_state_after_alloc(
        missing, support.FakeBlocks([]), num_external_tokens=0
    )
    assert "rS2" not in sched._reqs_need_send

    sched.update_state_after_alloc(
        support.FakeRequest(request_id="rN", kv_transfer_params=None),
        support.FakeBlocks([]),
        0,
    )
    assert set(sched._reqs_need_send) == {"rS"}


def test_build_connector_meta_role_filtering(conn):
    module, _ = conn
    consumer = _scheduler(module, role="kv_consumer")
    recv_req = support.FakeRequest(
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
        recv_req, support.FakeBlocks([[4, 5]]), num_external_tokens=2
    )
    meta = consumer.build_connector_meta(None)
    assert set(meta.reqs_to_recv) == {"rcv"}
    assert meta.reqs_to_recv["rcv"].local_block_ids == [[4, 5]]
    assert meta.reqs_to_send == {}
    assert consumer._reqs_need_recv == {}

    producer = _scheduler(module, role="kv_producer")
    send_req = support.FakeRequest(
        request_id="snd",
        kv_transfer_params={"do_remote_decode": True, "transfer_id": "tsd"},
    )
    producer.update_state_after_alloc(
        send_req, support.FakeBlocks([]), num_external_tokens=0
    )
    meta_p = producer.build_connector_meta(None)
    assert set(meta_p.reqs_to_send) == {"snd"}
    assert meta_p.reqs_to_send["snd"] == ("tsd", [])
    assert meta_p.reqs_to_recv == {}
    assert producer._reqs_need_send == {}

    producer._reqs_need_recv["ghost"] = (recv_req, [[1]])
    assert "ghost" not in producer.build_connector_meta(None).reqs_to_recv


def test_build_connector_meta_both_roles(conn):
    module, _ = conn
    both = _scheduler(module, role="kv_both")
    recv_req = support.FakeRequest(
        request_id="rcv",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_host": "h",
            "remote_port": 1,
            "transfer_id": "tcv",
        },
    )
    send_req = support.FakeRequest(
        request_id="snd",
        kv_transfer_params={"do_remote_decode": True, "transfer_id": "tsd"},
    )
    both.update_state_after_alloc(recv_req, support.FakeBlocks([[1]]), 1)
    both.update_state_after_alloc(send_req, support.FakeBlocks([]), 0)
    meta = both.build_connector_meta(None)
    assert set(meta.reqs_to_recv) == {"rcv"}
    assert set(meta.reqs_to_send) == {"snd"}


def test_request_finished_no_transfer_id(conn):
    module, _ = conn
    sched = _scheduler(module, role="kv_producer")
    assert sched.request_finished(
        support.FakeRequest(request_id="f0", kv_transfer_params=None), ([1],)
    ) == (False, None)
    no_tid = support.FakeRequest(
        request_id="f1", kv_transfer_params={"do_remote_decode": True}
    )
    assert sched.request_finished(no_tid, ([1],)) == (False, None)


def test_request_finished_remote_prefill_queues_recv(conn):
    module, _ = conn
    sched = _scheduler(module, role="kv_consumer")
    req = support.FakeRequest(
        request_id="f2",
        kv_transfer_params={"transfer_id": "tf2", "do_remote_prefill": True},
    )
    assert sched.request_finished(req, ([],)) == (False, None)
    assert list(sched._reqs_need_recv) == ["f2"]
    assert sched._reqs_need_recv["f2"][1] == []
    assert req.kv_transfer_params["do_remote_prefill"] is False


def test_request_finished_decode_handoff(conn):
    module, _ = conn
    sched = _scheduler(module, role="kv_producer")
    status = module.RequestStatus.FINISHED_LENGTH_CAPPED
    req = support.FakeRequest(
        request_id="f3",
        status=status,
        kv_transfer_params={"transfer_id": "tf3", "do_remote_decode": True},
    )
    delay, handoff = sched.request_finished(req, ([7, 8],))
    assert delay is True
    assert handoff == {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_host": sched.side_channel_host,
        "remote_port": sched.side_channel_port,
        "remote_tp_size": 2,
        "transfer_id": "tf3",
    }
    assert sched._reqs_need_send["f3"][1] == [[7, 8]]

    empty = support.FakeRequest(
        request_id="f4",
        status=status,
        kv_transfer_params={"transfer_id": "tf4", "do_remote_decode": True},
    )
    delay4, handoff4 = sched.request_finished(empty, ([],))
    assert delay4 is False
    assert handoff4["transfer_id"] == "tf4"
    assert "f4" not in sched._reqs_need_send


def test_request_finished_non_length_capped(conn):
    module, _ = conn
    sched = _scheduler(module, role="kv_producer")
    stopped = support.FakeRequest(
        request_id="f5",
        status=module.RequestStatus.FINISHED_STOPPED,
        kv_transfer_params={"transfer_id": "tf5", "do_remote_decode": True},
    )
    assert sched.request_finished(stopped, ([1],)) == (False, None)
    assert sched._reqs_need_send == {}


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


@pytest.fixture()
def facade(monkeypatch):
    module, recorder = support.load_connector(monkeypatch)
    support.reset_topology_state()

    def fake_base_init(self, vllm_config, role, kv_cache_config=None):
        self._connector_metadata = None

    monkeypatch.setattr(module.KVConnectorBase_V1, "__init__", fake_base_init)
    yield module, recorder
    sys.modules.pop("vllm_fl.distributed.kv_transfer.flagcx_connector", None)


def test_get_required_kvcache_layout(facade):
    module, _ = facade
    assert (
        module.FlagCXConnector.get_required_kvcache_layout(
            support.vllm_config(model=None)
        )
        is None
    )
    assert (
        module.FlagCXConnector.get_required_kvcache_layout(
            support.vllm_config(model=support.ModelInfo(use_mla=True))
        )
        is None
    )
    assert (
        module.FlagCXConnector.get_required_kvcache_layout(
            support.vllm_config(model=support.ModelInfo(use_mla=False))
        )
        == "HND"
    )


def test_facade_scheduler_role(facade):
    module, _ = facade
    cfg = support.vllm_config()
    connector = module.FlagCXConnector(
        cfg,
        module.KVConnectorRole.SCHEDULER,
        support.FakeKVCacheConfig([support.full_group(0)]),
    )
    assert connector.connector_scheduler is not None
    assert connector.connector_worker is None

    with pytest.raises(AssertionError):
        module.FlagCXConnector(cfg, module.KVConnectorRole.SCHEDULER, None)


def test_facade_worker_role(facade):
    module, _ = facade
    connector = module.FlagCXConnector(
        support.vllm_config(),
        module.KVConnectorRole.WORKER,
        support.FakeKVCacheConfig([support.full_group(0)]),
    )
    assert connector.connector_scheduler is None
    assert connector.connector_worker is not None
    assert connector.connector_worker.kv_role == "kv_both"
    connector.connector_worker.shutdown()


def test_facade_requires_engine_id(facade):
    module, _ = facade
    cfg = support.vllm_config(transfer=support.TransferInfo(engine_id=None))
    with pytest.raises(AssertionError):
        module.FlagCXConnector(
            cfg,
            module.KVConnectorRole.SCHEDULER,
            support.FakeKVCacheConfig([support.full_group(0)]),
        )


def test_facade_scheduler_delegation(facade):
    module, _ = facade
    connector = module.FlagCXConnector(
        support.vllm_config(),
        module.KVConnectorRole.SCHEDULER,
        support.FakeKVCacheConfig([support.full_group(0)]),
    )
    calls = []

    class _Sched:
        def get_num_new_matched_tokens(self, request, num):
            calls.append(("matched", request, num))
            return (7, True)

        def update_state_after_alloc(self, request, blocks, num):
            calls.append(("alloc", request, blocks, num))
            return "allocated"

        def build_connector_meta(self, scheduler_output):
            calls.append(("meta", scheduler_output))
            return "metadata"

        def request_finished(self, request, block_ids):
            calls.append(("finished", request, block_ids))
            return (True, {"k": 1})

    connector.connector_scheduler = _Sched()
    req = object()
    blocks = object()
    assert connector.get_num_new_matched_tokens(req, 3) == (7, True)
    assert connector.update_state_after_alloc(req, blocks, 9) == "allocated"
    assert connector.build_connector_meta("out") == "metadata"
    assert connector.request_finished(req, [1, 2]) == (True, {"k": 1})
    assert connector.request_finished_all_groups(req, ([1], [2])) == (
        True,
        {"k": 1},
    )
    assert calls[0] == ("matched", req, 3)
    assert calls[1] == ("alloc", req, blocks, 9)
    assert calls[2] == ("meta", "out")
    assert calls[3] == ("finished", req, ([1, 2],))
    assert calls[4] == ("finished", req, ([1], [2]))


def test_facade_worker_delegation(facade):
    module, _ = facade
    connector = module.FlagCXConnector(
        support.vllm_config(),
        module.KVConnectorRole.SCHEDULER,
        support.FakeKVCacheConfig([support.full_group(0)]),
    )
    calls = []

    class _Worker:
        kv_role = "kv_both"

        def register_kv_caches(self, caches):
            calls.append(("register", caches))

        def get_finished(self):
            return ({"s"}, {"r"})

        def start_load_kv(self, metadata):
            calls.append(("start", metadata))

        def wait_for_layer_load(self):
            calls.append(("wait",))

        def get_kv_connector_stats(self):
            return "stats"

    worker = _Worker()
    connector.connector_worker = worker
    connector.register_kv_caches({"l": 1})
    assert connector.get_finished({"x"}) == ({"s"}, {"r"})
    meta = module.FlagCXConnectorMetadata()
    connector._connector_metadata = meta
    connector.start_load_kv(forward_context=None)
    connector.wait_for_layer_load("layer")
    class _AttentionMetadata:
        pass
    connector.save_kv_layer("layer", None, _AttentionMetadata())
    connector.wait_for_save()
    assert connector.get_kv_connector_stats() == "stats"
    assert calls == [("register", {"l": 1}), ("start", meta), ("wait",)]


def test_facade_worker_none_paths(facade):
    module, _ = facade
    connector = module.FlagCXConnector(
        support.vllm_config(),
        module.KVConnectorRole.SCHEDULER,
        support.FakeKVCacheConfig([support.full_group(0)]),
    )
    assert connector.connector_worker is None
    assert connector.get_kv_connector_stats() is None
    connector.wait_for_layer_load("layer")

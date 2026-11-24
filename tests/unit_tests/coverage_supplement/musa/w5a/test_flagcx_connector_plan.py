# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Wire/metadata records and transfer-planning helpers of the FlagCX connector.

These tests exercise the pure-Python core: the msgspec wire struct, the
per-request metadata records, and the closed-form TP-ratio planners.  No
native library, socket, or accelerator is touched.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import msgspec
import pytest

import test_flagcx_connector_support as support


@pytest.fixture()
def conn(monkeypatch):
    module, _ = support.load_connector(monkeypatch)
    return module


def _region(conn, **kwargs):
    defaults = dict(
        layer_name="l0",
        layer_index=0,
        base_addr=0x1000,
        block_len=4096,
        kv_block_len=2048,
        group_index=0,
    )
    defaults.update(kwargs)
    return conn.TransferRegion(**defaults)


def test_wire_constants(conn):
    assert conn.TRANS_DONE == b"trans_done"
    assert conn.TRANS_ERROR == b"trans_error"


def test_transfer_region_is_frozen(conn):
    region = _region(conn)
    assert region.layer_name == "l0"
    assert region.group_index == 0
    with pytest.raises(Exception):
        region.layer_name = "other"


def test_agent_metadata_msgpack_roundtrip(conn):
    meta = conn.FlagCXAgentMetadata(
        remote_hostname="10.0.0.2",
        remote_port=4242,
        remote_tp_size=2,
        remote_tp_rank=1,
        req_blocks={"req-7": ("t-7", [[1, 2], [3]]), "req-8": ("t-8", [])},
        kv_caches_base_addr=[0x1000, 0x2000],
        block_lens=[4096, 4096],
        kv_block_lens=[2048, 2048],
        registered_layer_names=["model.layers.0.self_attn"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )
    blob = msgspec.msgpack.encode(meta)
    loaded = msgspec.msgpack.decode(blob, type=conn.FlagCXAgentMetadata)
    assert loaded.remote_hostname == "10.0.0.2"
    assert loaded.req_blocks["req-7"] == ("t-7", [[1, 2], [3]])
    assert loaded.req_blocks["req-8"] == ("t-8", [])
    assert loaded.kv_caches_base_addr == [0x1000, 0x2000]
    assert loaded.registered_group_indices == [0]


def test_send_block_meta_defaults(conn):
    meta = conn.SendBlockMeta(
        p_req_id="p1",
        transfer_id="t1",
        local_block_ids=[[1]],
        ready=support.types.SimpleNamespace(),
    )
    assert meta.expire_time == float("inf")
    assert meta.need_send == 0
    assert meta.sent == 0


def test_connector_metadata_add_new_req_recv(conn):
    meta = conn.FlagCXConnectorMetadata()
    meta.add_new_req(
        request_id="r1",
        local_block_ids=[[1, 2]],
        kv_transfer_params={
            "transfer_id": "t1",
            "remote_host": "10.0.0.1",
            "remote_port": 9000,
            "remote_tp_size": 4,
        },
    )
    recv = meta.reqs_to_recv["r1"]
    assert recv.local_block_ids == [[1, 2]]
    assert recv.remote_host == "10.0.0.1"
    assert recv.remote_port == 9000
    assert recv.transfer_id == "t1"
    assert recv.remote_tp_size == 4
    assert meta.reqs_to_send == {}


def test_connector_metadata_add_new_req_recv_default_tp_size(conn):
    meta = conn.FlagCXConnectorMetadata()
    meta.add_new_req(
        request_id="r1",
        local_block_ids=[],
        kv_transfer_params={
            "transfer_id": "t1",
            "remote_host": "h",
            "remote_port": 1,
        },
    )
    assert meta.reqs_to_recv["r1"].remote_tp_size == 0


def test_connector_metadata_add_new_req_send(conn):
    meta = conn.FlagCXConnectorMetadata()
    meta.add_new_req(
        request_id="r2",
        local_block_ids=[[5], [6]],
        kv_transfer_params={"transfer_id": "t2"},
        load_remote_cache=False,
    )
    assert meta.reqs_to_send["r2"] == ("t2", [[5], [6]])
    assert meta.reqs_to_recv == {}


def test_group_concurrent_contiguous_runs(conn):
    src, dst = conn.group_concurrent_contiguous([1, 2, 3, 7, 8], [10, 11, 12, 20, 21])
    assert src == [[1, 2, 3], [7, 8]]
    assert dst == [[10, 11, 12], [20, 21]]
    # A destination gap alone breaks the run.
    src, dst = conn.group_concurrent_contiguous([1, 2], [10, 12])
    assert src == [[1], [2]]
    assert dst == [[10], [12]]
    assert conn.group_concurrent_contiguous([], []) == ([], [])


def test_get_tp_ratio(conn):
    assert conn._get_tp_ratio(4, 2) == 2
    assert conn._get_tp_ratio(2, 2) == 1
    assert conn._get_tp_ratio(2, 4) == -2
    with pytest.raises(AssertionError):
        conn._get_tp_ratio(3, 2)
    with pytest.raises(AssertionError):
        conn._get_tp_ratio(2, 3)


def test_expand_transfer_regions_defaults_split(conn):
    regions = conn._expand_transfer_regions(
        base_addrs=[0x1000],
        block_lens=[4096],
        kv_block_lens=[2048],
        layer_names=["model.layers.0.self_attn"],
        layer_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    assert [r.base_addr for r in regions] == [0x1000, 0x1000 + 2048]
    assert [r.group_index for r in regions] == [0, 0]


def test_expand_transfer_regions_explicit_flags(conn):
    regions = conn._expand_transfer_regions(
        base_addrs=[0x1000, 0x9000],
        block_lens=[4096, 2048],
        kv_block_lens=[2048, 2048],
        layer_names=["l0", "l1"],
        layer_indices=[0, 1],
        is_kv_layout_blocks_first=False,
        group_indices=[0, 1],
        split_kv_regions=[True, False],
    )
    assert [(r.layer_name, r.base_addr, r.group_index) for r in regions] == [
        ("l0", 0x1000, 0),
        ("l0", 0x1000 + 2048, 0),
        ("l1", 0x9000, 1),
    ]


def test_expand_transfer_regions_length_guards(conn):
    ok = conn._expand_transfer_regions(
        base_addrs=[0x1000],
        block_lens=[4096],
        kv_block_lens=[2048],
        layer_names=["l0"],
        layer_indices=[0],
        is_kv_layout_blocks_first=False,
    )
    assert len(ok) == 1
    with pytest.raises(AssertionError):
        conn._expand_transfer_regions(
            base_addrs=[0x1000, 0x2000],
            block_lens=[4096],
            kv_block_lens=[2048],
            layer_names=["l0"],
            layer_indices=[0],
            is_kv_layout_blocks_first=False,
        )
    with pytest.raises(AssertionError):
        conn._expand_transfer_regions(
            base_addrs=[0x1000],
            block_lens=[4096],
            kv_block_lens=[2048],
            layer_names=["l0"],
            layer_indices=[0],
            is_kv_layout_blocks_first=False,
            group_indices=[0, 1],
        )


def test_compute_sender_transfer_plan_homogeneous(conn):
    assert conn._compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=2,
        remote_tp_rank=0,
        remote_tp_size=2,
        local_kv_block_len=2048,
        remote_kv_block_len=2048,
        producer_cache_replicated=False,
    ) == (True, 0, 0, 2048)


def test_compute_sender_transfer_plan_upscale(conn):
    assert conn._compute_sender_transfer_plan(
        local_tp_rank=1,
        local_tp_size=4,
        remote_tp_rank=0,
        remote_tp_size=2,
        local_kv_block_len=2048,
        remote_kv_block_len=4096,
        producer_cache_replicated=False,
    ) == (True, 0, 2048, 2048)
    assert conn._compute_sender_transfer_plan(
        local_tp_rank=3,
        local_tp_size=4,
        remote_tp_rank=0,
        remote_tp_size=2,
        local_kv_block_len=2048,
        remote_kv_block_len=4096,
        producer_cache_replicated=True,
    ) == (False, 0, 0, 2048)
    assert conn._compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=4,
        remote_tp_rank=0,
        remote_tp_size=2,
        local_kv_block_len=2048,
        remote_kv_block_len=4096,
        producer_cache_replicated=True,
    ) == (True, 0, 0, 2048)


def test_compute_sender_transfer_plan_downscale(conn):
    assert conn._compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=2,
        remote_tp_rank=3,
        remote_tp_size=4,
        local_kv_block_len=4096,
        remote_kv_block_len=2048,
        producer_cache_replicated=False,
    ) == (True, 2048, 0, 2048)
    assert conn._compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=2,
        remote_tp_rank=1,
        remote_tp_size=4,
        local_kv_block_len=4096,
        remote_kv_block_len=2048,
        producer_cache_replicated=True,
    ) == (True, 0, 0, 4096)


def test_can_coalesce_block_transfers(conn):
    assert (
        conn._can_coalesce_block_transfers(4096, 4096, 0, 0, 4096) is True
    )
    assert (
        conn._can_coalesce_block_transfers(4096, 8192, 0, 0, 4096) is False
    )
    assert (
        conn._can_coalesce_block_transfers(4096, 4096, 0, 2048, 2048) is False
    )
    assert (
        conn._can_coalesce_block_transfers(4096, 4096, 1024, 0, 4096) is False
    )


def test_validate_asymmetric_region_lengths_replicated_skips(conn):
    assert (
        conn._validate_asymmetric_region_lengths(
            [_region(conn, kv_block_len=1)],
            [_region(conn, kv_block_len=999)],
            local_tp_size=4,
            remote_tp_size=2,
            producer_cache_replicated=True,
        )
        is None
    )


def test_validate_asymmetric_region_lengths_count_mismatch(conn):
    err = conn._validate_asymmetric_region_lengths(
        [_region(conn)],
        [],
        local_tp_size=2,
        remote_tp_size=2,
        producer_cache_replicated=False,
    )
    assert err == "FlagCX asymmetric TP requires matching KV region counts."


def test_validate_asymmetric_region_lengths_per_ratio(conn):
    homogeneous = conn._validate_asymmetric_region_lengths(
        [_region(conn, kv_block_len=2048)],
        [_region(conn, kv_block_len=4096)],
        local_tp_size=2,
        remote_tp_size=2,
        producer_cache_replicated=False,
    )
    assert homogeneous.startswith("FlagCX KV region length mismatch")

    assert (
        conn._validate_asymmetric_region_lengths(
            [_region(conn, kv_block_len=2048)],
            [_region(conn, kv_block_len=4096)],
            local_tp_size=4,
            remote_tp_size=2,
            producer_cache_replicated=False,
        )
        is None
    )
    bad_up = conn._validate_asymmetric_region_lengths(
        [_region(conn, kv_block_len=2048)],
        [_region(conn, kv_block_len=2048)],
        local_tp_size=4,
        remote_tp_size=2,
        producer_cache_replicated=False,
    )
    assert "destination KV region length mismatch" in bad_up

    assert (
        conn._validate_asymmetric_region_lengths(
            [_region(conn, kv_block_len=4096)],
            [_region(conn, kv_block_len=2048)],
            local_tp_size=2,
            remote_tp_size=4,
            producer_cache_replicated=False,
        )
        is None
    )
    bad_down = conn._validate_asymmetric_region_lengths(
        [_region(conn, kv_block_len=2048)],
        [_region(conn, kv_block_len=2048)],
        local_tp_size=2,
        remote_tp_size=4,
        producer_cache_replicated=False,
    )
    assert "source KV region length mismatch" in bad_down


def test_align_transfer_regions_by_occurrence(conn):
    local = [
        _region(conn, layer_name="l0", base_addr=0x1000),
        _region(conn, layer_name="l0", base_addr=0x1800),
        _region(conn, layer_name="l1", base_addr=0x2000),
    ]
    remote = [
        _region(conn, layer_name="l0", base_addr=0x9000),
        _region(conn, layer_name="l0", base_addr=0x9800),
        _region(conn, layer_name="l1", base_addr=0xA000),
    ]
    aligned_local, aligned_remote, err = conn._align_transfer_regions(local, remote)
    assert err is None
    assert [r.base_addr for r in aligned_local] == [0x1000, 0x1800, 0x2000]
    assert [r.base_addr for r in aligned_remote] == [0x9000, 0x9800, 0xA000]


def test_align_transfer_regions_error_branches(conn):
    missing = conn._align_transfer_regions([_region(conn, layer_name="l9")], [])[2]
    assert "no matching consumer occurrence" in missing

    layer_mismatch = conn._align_transfer_regions(
        [_region(conn, layer_name="l0", layer_index=0)],
        [_region(conn, layer_name="l0", layer_index=5)],
    )[2]
    assert "layer index mismatch" in layer_mismatch

    group_mismatch = conn._align_transfer_regions(
        [_region(conn, layer_name="l0", group_index=0)],
        [_region(conn, layer_name="l0", group_index=3)],
    )[2]
    assert "group index mismatch" in group_mismatch


def test_get_side_channel_port_env_and_dp(conn, monkeypatch):
    monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)
    assert (
        conn._get_side_channel_port(
            support.vllm_config(
                parallel=support.ParallelInfo(tp_size=2, dp_rank=0)
            )
        )
        == 8998
    )
    assert (
        conn._get_side_channel_port(
            support.vllm_config(
                parallel=support.ParallelInfo(tp_size=4, dp_rank=3)
            )
        )
        == 8998 + 12
    )
    monkeypatch.setenv("FLAGCX_BOOTSTRAP_PORT", "7000")
    assert (
        conn._get_side_channel_port(
            support.vllm_config(
                parallel=support.ParallelInfo(tp_size=1, dp_rank=1)
            )
        )
        == 7001
    )

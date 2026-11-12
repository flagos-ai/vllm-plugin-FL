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
"""Transfer-plan helper coverage tests for flagcx_connector.py.

The module-level planning helpers decide how one producer rank copies
KV blocks to one consumer rank under any TP-ratio combination: run
splitting, region expansion/splitting, sender plans, coalescing,
asymmetric-length validation and producer/consumer region alignment.
Every expected value below is derived in closed form from the inputs.
"""

from __future__ import annotations

import pytest

from . import _flagcx_env

_flagcx_env.ensure_importable_env()


@pytest.fixture()
def conn(monkeypatch):
    """Import the connector against the fake flagcx wheel facade.

    Yields (module, call recorder); drops the cached module afterwards
    so the next import re-runs the ladder.
    """
    recorder = _flagcx_env.install_fake_flagcx_wheel(monkeypatch)
    _flagcx_env.ensure_vllm_api_compat()
    _flagcx_env.drop_connector_module()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as conn

    yield conn, recorder
    _flagcx_env.drop_connector_module()


def test_group_concurrent_contiguous_run_splitting(conn):
    """Runs advance while both src and dst ids are contiguous (diff==1)."""
    conn, _ = conn
    src_groups, dst_groups = conn.group_concurrent_contiguous(
        [1, 2, 3, 7, 8], [10, 11, 12, 20, 21]
    )
    assert src_groups == [[1, 2, 3], [7, 8]]
    assert dst_groups == [[10, 11, 12], [20, 21]]

    # A dst gap alone also breaks the run.
    src_groups, dst_groups = conn.group_concurrent_contiguous([1, 2], [10, 12])
    assert src_groups == [[1], [2]]
    assert dst_groups == [[10], [12]]

    # Empty inputs produce no groups.
    assert conn.group_concurrent_contiguous([], []) == ([], [])


def test_get_tp_ratio_positive_and_negative(conn):
    """tp_ratio is local/remote when local >= remote, else negated."""
    conn, _ = conn
    assert conn._get_tp_ratio(4, 2) == 2
    assert conn._get_tp_ratio(2, 2) == 1
    assert conn._get_tp_ratio(2, 4) == -2
    with pytest.raises(AssertionError):
        conn._get_tp_ratio(3, 2)
    with pytest.raises(AssertionError):
        conn._get_tp_ratio(2, 3)


def test_expand_transfer_regions_defaults(conn):
    """Without explicit lists the group index defaults to 0 and every
    region splits when the layout is blocks-first."""
    conn, _ = conn
    regions = conn._expand_transfer_regions(
        base_addrs=[0x1000],
        block_lens=[4096],
        kv_block_lens=[2048],
        layer_names=["model.layers.0.self_attn"],
        layer_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    assert [r.base_addr for r in regions] == [0x1000, 0x1000 + 2048]
    assert all(r.group_index == 0 for r in regions)
    assert [r.layer_name for r in regions] == ["model.layers.0.self_attn"] * 2


def test_expand_transfer_regions_split_flag(conn):
    """split_kv_regions decides per-region K/V splitting; group_indices
    and the layout default are honored separately."""
    conn, _ = conn
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
    # l0 splits into K then V; l1 stays one region.
    assert [(r.layer_name, r.base_addr, r.group_index) for r in regions] == [
        ("l0", 0x1000, 0),
        ("l0", 0x1000 + 2048, 0),
        ("l1", 0x9000, 1),
    ]


def test_expand_transfer_regions_length_mismatch(conn):
    """Mismatched metadata lengths are rejected up front, while a
    matching set of one-element lists expands cleanly."""
    conn, _ = conn
    ok = conn._expand_transfer_regions(
        base_addrs=[0x1000],
        block_lens=[4096],
        kv_block_lens=[2048],
        layer_names=["l0"],
        layer_indices=[0],
        is_kv_layout_blocks_first=False,
    )
    assert len(ok) == 1 and ok[0].layer_name == "l0"
    with pytest.raises(AssertionError):
        conn._expand_transfer_regions(
            base_addrs=[0x1000, 0x2000],
            block_lens=[4096],
            kv_block_lens=[2048],
            layer_names=["l0"],
            layer_indices=[0],
            is_kv_layout_blocks_first=False,
        )


def test_compute_sender_transfer_plan_homogeneous(conn):
    """tp_ratio == 1 transfers the whole local KV block at offset 0."""
    conn, _ = conn
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
    """Local TP larger than remote: each local rank writes its slice into
    the bigger remote block (dst offset = (rank % ratio) * local len)."""
    conn, _ = conn
    assert conn._compute_sender_transfer_plan(
        local_tp_rank=1,
        local_tp_size=4,
        remote_tp_rank=0,
        remote_tp_size=2,
        local_kv_block_len=2048,
        remote_kv_block_len=4096,
        producer_cache_replicated=False,
    ) == (True, 0, 2048, 2048)
    # With a replicated producer cache only rank 0 of each group sends.
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
    """Remote TP larger than local: gather a remote-sized slice from the
    bigger local block (src offset = (remote rank % ratio) * remote len)."""
    conn, _ = conn
    assert conn._compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=2,
        remote_tp_rank=3,
        remote_tp_size=4,
        local_kv_block_len=4096,
        remote_kv_block_len=2048,
        producer_cache_replicated=False,
    ) == (True, 2048, 0, 2048)
    # Replication makes every local rank able to serve the whole block.
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
    """Coalescing needs zero offsets and matching full-block lengths."""
    conn, _ = conn
    assert (
        conn._can_coalesce_block_transfers(
            local_region_block_len=4096,
            remote_region_block_len=4096,
            src_region_offset=0,
            dst_region_offset=0,
            transfer_len=4096,
        )
        is True
    )
    assert (
        conn._can_coalesce_block_transfers(
            local_region_block_len=4096,
            remote_region_block_len=8192,
            src_region_offset=0,
            dst_region_offset=0,
            transfer_len=4096,
        )
        is False
    )
    assert (
        conn._can_coalesce_block_transfers(
            local_region_block_len=4096,
            remote_region_block_len=4096,
            src_region_offset=0,
            dst_region_offset=2048,
            transfer_len=2048,
        )
        is False
    )


def _region(**kw):
    """Build a TransferRegion with sensible test defaults."""
    defaults = dict(
        layer_name="l0",
        layer_index=0,
        base_addr=0x1000,
        block_len=4096,
        kv_block_len=2048,
        group_index=0,
    )
    defaults.update(kw)
    from vllm_fl.distributed.kv_transfer.flagcx_connector import TransferRegion

    return TransferRegion(**defaults)


def test_validate_asymmetric_region_lengths_replicated(conn):
    """A replicated producer cache skips length validation entirely."""
    conn, _ = conn
    assert (
        conn._validate_asymmetric_region_lengths(
            [_region(kv_block_len=1)],
            [_region(kv_block_len=999)],
            local_tp_size=4,
            remote_tp_size=2,
            producer_cache_replicated=True,
        )
        is None
    )


def test_validate_asymmetric_region_lengths_mismatch_counts(conn):
    """Differing region counts are rejected."""
    conn, _ = conn
    err = conn._validate_asymmetric_region_lengths(
        [_region()],
        [],
        local_tp_size=2,
        remote_tp_size=2,
        producer_cache_replicated=False,
    )
    assert err == "FlagCX asymmetric TP requires matching KV region counts."


def test_validate_asymmetric_region_lengths_per_ratio(conn):
    """Homogeneous TP requires equal lengths; upscale requires remote ==
    local * ratio; downscale requires local == remote * ratio."""
    conn, _ = conn
    homogeneous_err = conn._validate_asymmetric_region_lengths(
        [_region(kv_block_len=2048)],
        [_region(kv_block_len=4096)],
        local_tp_size=2,
        remote_tp_size=2,
        producer_cache_replicated=False,
    )
    assert homogeneous_err is not None
    assert homogeneous_err.startswith("FlagCX KV region length mismatch")

    ok_up = conn._validate_asymmetric_region_lengths(
        [_region(kv_block_len=2048)],
        [_region(kv_block_len=4096)],
        local_tp_size=4,
        remote_tp_size=2,
        producer_cache_replicated=False,
    )
    assert ok_up is None
    bad_up = conn._validate_asymmetric_region_lengths(
        [_region(kv_block_len=2048)],
        [_region(kv_block_len=2048)],
        local_tp_size=4,
        remote_tp_size=2,
        producer_cache_replicated=False,
    )
    assert "destination KV region length mismatch" in bad_up

    ok_down = conn._validate_asymmetric_region_lengths(
        [_region(kv_block_len=4096)],
        [_region(kv_block_len=2048)],
        local_tp_size=2,
        remote_tp_size=4,
        producer_cache_replicated=False,
    )
    assert ok_down is None
    bad_down = conn._validate_asymmetric_region_lengths(
        [_region(kv_block_len=2048)],
        [_region(kv_block_len=2048)],
        local_tp_size=2,
        remote_tp_size=4,
        producer_cache_replicated=False,
    )
    assert "source KV region length mismatch" in bad_down


def test_align_transfer_regions_occurrence_keying(conn):
    """Regions pair by (layer_name, occurrence) so split K/V views of one
    layer align even when PP shards registered different layer subsets."""
    conn, _ = conn
    local = [
        _region(layer_name="l0", base_addr=0x1000),
        _region(layer_name="l0", base_addr=0x1800),
        _region(layer_name="l1", base_addr=0x2000),
    ]
    remote = [
        _region(layer_name="l0", base_addr=0x9000),
        _region(layer_name="l0", base_addr=0x9800),
        _region(layer_name="l1", base_addr=0xA000),
    ]
    aligned_local, aligned_remote, err = conn._align_transfer_regions(local, remote)
    assert err is None
    assert [r.base_addr for r in aligned_local] == [0x1000, 0x1800, 0x2000]
    assert [r.base_addr for r in aligned_remote] == [0x9000, 0x9800, 0xA000]


def test_align_transfer_regions_error_branches(conn):
    """A producer occurrence with no consumer, or a layer/group index
    mismatch, is reported as an error string."""
    conn, _ = conn
    missing = conn._align_transfer_regions([_region(layer_name="l9")], [])[2]
    assert "no matching consumer occurrence" in missing

    layer_mismatch = conn._align_transfer_regions(
        [_region(layer_name="l0", layer_index=0)],
        [_region(layer_name="l0", layer_index=5)],
    )[2]
    assert "layer index mismatch" in layer_mismatch

    group_mismatch = conn._align_transfer_regions(
        [_region(layer_name="l0", group_index=0)],
        [_region(layer_name="l0", group_index=3)],
    )[2]
    assert "group index mismatch" in group_mismatch

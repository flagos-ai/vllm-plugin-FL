# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU tests for ``vllm_fl.distributed.kv_transfer.flagcx_connector``.

Covers the CPU-testable surface of the FlagCX KV-transfer connector:

* module-level transfer planning helpers (TP ratio, region expansion and
  alignment, coalescing, contiguity grouping);
* ``FlagCXConnectorMetadata`` request bookkeeping (recv vs send sides);
* ``FlagCXConnectorScheduler`` (matched-token planning, sliding-window block
  clipping, mamba N-1 truncation, state-after-alloc, request_finished
  handoff payloads);
* ``FlagCXConnectorWorker`` construction bookkeeping, send-request arming,
  timeout expiration sweeps, finished-set accounting and shutdown ordering;
* the ``FlagCXConnector`` facade role dispatch and factory classmethods.

The real GPU/FlagCX data-plane paths (flagcxP2pBatchWriteSync, the ZMQ sender
thread, RDMA handshake) require the native library and GPUs and stay covered
by GPU functional tests.
"""

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
def connector_env(monkeypatch):
    """Install stubs, import the connector module and rebind its backend
    probing helpers (they were captured at import time)."""
    env = FlagcxStubEnv(monkeypatch).install()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as fc

    monkeypatch.setattr(fc, "get_current_attn_backends", lambda cfg: [_StubBackend()])
    env.fc = fc
    return env


class _StubBackend:
    def get_name(self):
        return "stub_backend"


# ---------------------------------------------------------------------------
# Module-level helpers: TP ratio planning
# ---------------------------------------------------------------------------


def test_get_tp_ratio_equal_sizes(connector_env):
    """Equal local/remote TP sizes yield a ratio of 1 (homogeneous)."""
    fc = connector_env.fc
    assert fc._get_tp_ratio(4, 4) == 1
    assert fc._get_tp_ratio(1, 1) == 1


def test_get_tp_ratio_local_larger(connector_env):
    """Local > remote gives a positive ratio of the division."""
    fc = connector_env.fc
    assert fc._get_tp_ratio(4, 2) == 2
    assert fc._get_tp_ratio(8, 2) == 4


def test_get_tp_ratio_remote_larger(connector_env):
    """Remote > local gives a negative ratio (gather from multiple peers)."""
    fc = connector_env.fc
    assert fc._get_tp_ratio(2, 8) == -4
    assert fc._get_tp_ratio(1, 3) == -3


@pytest.mark.parametrize(
    ("local", "remote", "message"),
    [
        (3, 2, "Local TP size 3 not divisible by remote TP size 2."),
        (2, 3, "Remote TP size 3 not divisible by local TP size 2."),
    ],
)
def test_get_tp_ratio_rejects_non_divisible(connector_env, local, remote, message):
    """Non-divisible TP combinations fail with an explicit assertion."""
    fc = connector_env.fc
    with pytest.raises(AssertionError, match=message):
        fc._get_tp_ratio(local, remote)


def test_sender_plan_homogeneous_full_copy(connector_env):
    """tp_ratio == 1: full local block copied at both offsets 0."""
    plan = connector_env.fc._compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=4,
        remote_tp_rank=1,
        remote_tp_size=4,
        local_kv_block_len=128,
        remote_kv_block_len=128,
        producer_cache_replicated=False,
    )
    assert plan == (True, 0, 0, 128)


@pytest.mark.parametrize(
    ("local_rank", "expected"),
    [(0, (True, 0, 0, 128)), (1, (False, 0, 0, 128)), (2, (True, 0, 0, 128))],
)
def test_sender_plan_local_larger_replicated(connector_env, local_rank, expected):
    """Local TP > remote TP with a replicated producer cache: only ranks
    whose index is a multiple of the ratio transfer (others skip)."""
    plan = connector_env.fc._compute_sender_transfer_plan(
        local_tp_rank=local_rank,
        local_tp_size=4,
        remote_tp_rank=0,
        remote_tp_size=2,
        local_kv_block_len=128,
        remote_kv_block_len=256,
        producer_cache_replicated=True,
    )
    assert plan == expected


@pytest.mark.parametrize(
    ("local_rank", "expected_dst"),
    [(0, 0), (1, 128), (2, 256), (3, 384)],
)
def test_sender_plan_local_larger_sharded_offsets(
    connector_env, local_rank, expected_dst
):
    """Local TP > remote TP, sharded cache: every rank sends its slice at a
    destination offset (rank % ratio) * local_kv_block_len."""
    plan = connector_env.fc._compute_sender_transfer_plan(
        local_tp_rank=local_rank,
        local_tp_size=8,
        remote_tp_rank=0,
        remote_tp_size=2,
        local_kv_block_len=128,
        remote_kv_block_len=512,
        producer_cache_replicated=False,
    )
    assert plan == (True, 0, expected_dst, 128)


def test_sender_plan_remote_larger_replicated_sends_all(connector_env):
    """Remote TP > local TP with a replicated cache: full copy, no offset."""
    plan = connector_env.fc._compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=2,
        remote_tp_rank=3,
        remote_tp_size=8,
        local_kv_block_len=128,
        remote_kv_block_len=32,
        producer_cache_replicated=True,
    )
    assert plan == (True, 0, 0, 128)


@pytest.mark.parametrize(
    ("remote_rank", "expected_src"),
    [(0, 0), (1, 32), (5, 32), (7, 96), (4, 0)],
)
def test_sender_plan_remote_larger_gathers_slice(
    connector_env, remote_rank, expected_src
):
    """Remote TP > local TP, sharded: each consumer rank pulls the slice
    (remote_rank % ratio) * remote_kv_block_len from the source."""
    plan = connector_env.fc._compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=2,
        remote_tp_rank=remote_rank,
        remote_tp_size=8,
        local_kv_block_len=128,
        remote_kv_block_len=32,
        producer_cache_replicated=False,
    )
    assert plan == (True, expected_src, 0, 32)


# ---------------------------------------------------------------------------
# Module-level helpers: region expansion and alignment
# ---------------------------------------------------------------------------


def test_group_concurrent_contiguous_runs(connector_env):
    """Contiguous runs in both src and dst collapse into one group each."""
    src_groups, dst_groups = connector_env.fc.group_concurrent_contiguous(
        [1, 2, 3, 7, 8, 20], [10, 11, 12, 30, 31, 99]
    )
    assert src_groups == [[1, 2, 3], [7, 8], [20]]
    assert dst_groups == [[10, 11, 12], [30, 31], [99]]


def test_group_concurrent_contiguous_breaks_on_dst_gap(connector_env):
    """A gap in the *destination* ids breaks the run even when src is
    contiguous (and vice versa)."""
    src_groups, dst_groups = connector_env.fc.group_concurrent_contiguous(
        [1, 2, 3], [10, 12, 13]
    )
    assert src_groups == [[1], [2, 3]]
    assert dst_groups == [[10], [12, 13]]


def test_group_concurrent_contiguous_empty(connector_env):
    """Empty input yields empty group lists."""
    assert connector_env.fc.group_concurrent_contiguous([], []) == ([], [])


def test_expand_transfer_regions_plain(connector_env):
    """Without blocks-first splitting each region stays one entry."""
    fc = connector_env.fc
    regions = fc._expand_transfer_regions(
        base_addrs=[1000],
        block_lens=[64],
        kv_block_lens=[32],
        layer_names=["model.layers.0.self_attn.attn"],
        layer_indices=[0],
        is_kv_layout_blocks_first=False,
    )
    assert len(regions) == 1
    region = regions[0]
    assert region.base_addr == 1000
    assert region.block_len == 64
    assert region.kv_block_len == 32
    assert region.layer_name == "model.layers.0.self_attn.attn"
    assert region.layer_index == 0
    assert region.group_index == 0


def test_expand_transfer_regions_splits_k_and_v(connector_env):
    """Blocks-first layouts with split=True emit K at base and V at
    base + kv_block_len."""
    fc = connector_env.fc
    regions = fc._expand_transfer_regions(
        base_addrs=[1000, 2000],
        block_lens=[64, 64],
        kv_block_lens=[32, 32],
        layer_names=["layers_a", "layers_b"],
        layer_indices=[1, 2],
        is_kv_layout_blocks_first=True,
        group_indices=[0, 1],
        split_kv_regions=[True, False],
    )
    assert [(r.base_addr, r.group_index) for r in regions] == [
        (1000, 0),  # K
        (1032, 0),  # V
        (2000, 1),
    ]
    assert regions[0].layer_name == regions[1].layer_name == "layers_a"


def test_expand_transfer_regions_rejects_mismatched_lengths(connector_env):
    """Mismatched parallel-list lengths fail the alignment assertion."""
    fc = connector_env.fc
    with pytest.raises(AssertionError, match="matching metadata lengths"):
        fc._expand_transfer_regions(
            base_addrs=[1000, 2000],
            block_lens=[64],
            kv_block_lens=[32],
            layer_names=["layers_a"],
            layer_indices=[0],
            is_kv_layout_blocks_first=False,
        )


def test_can_coalesce_only_when_full_aligned_copy(connector_env):
    """Coalescing requires zero offsets and transfer_len equal to both
    region block lengths."""
    fc = connector_env.fc
    assert fc._can_coalesce_block_transfers(64, 64, 0, 0, 64) is True
    assert fc._can_coalesce_block_transfers(64, 128, 0, 0, 64) is False
    assert fc._can_coalesce_block_transfers(64, 64, 8, 0, 64) is False
    assert fc._can_coalesce_block_transfers(64, 64, 0, 8, 64) is False
    assert fc._can_coalesce_block_transfers(64, 64, 0, 0, 32) is False


def _make_region(layer, layer_index, base, block_len, kv_block_len, group=0):
    from vllm_fl.distributed.kv_transfer import flagcx_connector as _fc

    return _fc.TransferRegion(
        layer_name=layer,
        layer_index=layer_index,
        base_addr=base,
        block_len=block_len,
        kv_block_len=kv_block_len,
        group_index=group,
    )


def test_align_transfer_regions_pairs_by_layer_occurrence(connector_env):
    """PP shards with repeated layer names pair by (layer, occurrence)."""
    fc = connector_env.fc
    local = [
        _make_region("a", 0, 10, 64, 32),
        _make_region("b", 1, 20, 64, 32),
        _make_region("a", 0, 30, 64, 32),
    ]
    remote = [
        _make_region("a", 0, 110, 64, 32),
        _make_region("a", 0, 130, 64, 32),
        _make_region("b", 1, 120, 64, 32),
    ]
    aligned_local, aligned_remote, err = fc._align_transfer_regions(local, remote)
    assert err is None
    assert [r.base_addr for r in aligned_local] == [10, 20, 30]
    assert [r.base_addr for r in aligned_remote] == [110, 120, 130]


def test_align_transfer_regions_reports_missing_consumer(connector_env):
    """A producer layer with no consumer occurrence returns a clear error."""
    fc = connector_env.fc
    local = [_make_region("a", 0, 10, 64, 32), _make_region("x", 9, 50, 64, 32)]
    remote = [_make_region("a", 0, 110, 64, 32)]
    _, _, err = fc._align_transfer_regions(local, remote)
    assert err is not None
    assert "no matching consumer occurrence" in err
    assert "x" in err


def test_align_transfer_regions_rejects_index_mismatch(connector_env):
    """Same layer name but different layer index fails alignment."""
    fc = connector_env.fc
    local = [_make_region("a", 0, 10, 64, 32)]
    remote = [_make_region("a", 5, 110, 64, 32)]
    _, _, err = fc._align_transfer_regions(local, remote)
    assert err is not None
    assert "layer index mismatch" in err


def test_align_transfer_regions_rejects_group_mismatch(connector_env):
    """Same layer but different KV-cache group fails alignment."""
    fc = connector_env.fc
    local = [_make_region("a", 0, 10, 64, 32, group=0)]
    remote = [_make_region("a", 0, 110, 64, 32, group=1)]
    _, _, err = fc._align_transfer_regions(local, remote)
    assert err is not None
    assert "group index mismatch" in err


def test_validate_asymmetric_region_lengths(connector_env):
    """Region-length validation across the three TP-ratio regimes."""
    fc = connector_env.fc

    def region(kv_block_len):
        return _make_region("a", 0, 0, 64, kv_block_len)

    # Count mismatch always errors.
    assert (
        fc._validate_asymmetric_region_lengths([region(32)], [], 2, 2, False)
        == "FlagCX asymmetric TP requires matching KV region counts."
    )

    # Replicated producer skips length checks entirely.
    assert (
        fc._validate_asymmetric_region_lengths([region(32)], [region(64)], 2, 1, True)
        is None
    )

    # Homogeneous mismatch.
    err = fc._validate_asymmetric_region_lengths(
        [region(32)], [region(64)], 2, 2, False
    )
    assert "homogeneous TP" in err and "region 0" in err

    # Destination larger (tp_ratio 2): remote must be 2x local.
    ok = fc._validate_asymmetric_region_lengths([region(32)], [region(64)], 2, 1, False)
    assert ok is None
    err = fc._validate_asymmetric_region_lengths(
        [region(32)], [region(48)], 2, 1, False
    )
    assert "destination KV region length mismatch" in err

    # Source larger (negative ratio): local must be 2x remote.
    ok = fc._validate_asymmetric_region_lengths([region(64)], [region(32)], 1, 2, False)
    assert ok is None
    err = fc._validate_asymmetric_region_lengths(
        [region(48)], [region(32)], 1, 2, False
    )
    assert "source KV region length mismatch" in err


# ---------------------------------------------------------------------------
# FlagCXConnectorMetadata
# ---------------------------------------------------------------------------


def test_metadata_add_new_recv_req(connector_env):
    """do-remote-prefill requests land in reqs_to_recv with their remote
    endpoint metadata."""
    fc = connector_env.fc
    meta = fc.FlagCXConnectorMetadata()
    meta.add_new_req(
        request_id="r1",
        local_block_ids=[[1, 2]],
        kv_transfer_params={
            "transfer_id": "t1",
            "remote_host": "10.0.0.1",
            "remote_port": 9000,
            "remote_tp_size": 2,
        },
    )
    assert "r1" not in meta.reqs_to_send
    recv = meta.reqs_to_recv["r1"]
    assert recv.local_block_ids == [[1, 2]]
    assert recv.remote_host == "10.0.0.1"
    assert recv.remote_port == 9000
    assert recv.transfer_id == "t1"
    assert recv.remote_tp_size == 2


def test_metadata_add_new_recv_defaults_tp_size(connector_env):
    """A missing remote_tp_size defaults to 0 on the recv meta."""
    fc = connector_env.fc
    meta = fc.FlagCXConnectorMetadata()
    meta.add_new_req(
        request_id="r2",
        local_block_ids=[[]],
        kv_transfer_params={
            "transfer_id": "t2",
            "remote_host": "h",
            "remote_port": 1,
        },
    )
    assert meta.reqs_to_recv["r2"].remote_tp_size == 0


def test_metadata_add_new_send_req(connector_env):
    """load_remote_cache=False routes the request to the send side."""
    fc = connector_env.fc
    meta = fc.FlagCXConnectorMetadata()
    meta.add_new_req(
        request_id="s1",
        local_block_ids=[[3, 4], [5]],
        kv_transfer_params={
            "transfer_id": "t3",
            "remote_host": "h",
            "remote_port": 1,
        },
        load_remote_cache=False,
    )
    assert "s1" not in meta.reqs_to_recv
    transfer_id, block_ids = meta.reqs_to_send["s1"]
    assert transfer_id == "t3"
    assert block_ids == [[3, 4], [5]]


def test_metadata_missing_transfer_id_raises(connector_env):
    """kv_transfer_params without transfer_id raises KeyError."""
    fc = connector_env.fc
    meta = fc.FlagCXConnectorMetadata()
    with pytest.raises(KeyError, match="transfer_id"):
        meta.add_new_req(
            request_id="x",
            local_block_ids=[[]],
            kv_transfer_params={"remote_host": "h", "remote_port": 1},
        )


# ---------------------------------------------------------------------------
# FlagCXConnector facade and scheduler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("use_mla", "model_config", "expected"),
    [
        (None, None, None),  # no model config (unit-test mode)
        (True, object(), None),  # MLA keeps engine default
        (False, object(), "HND"),  # dense forces HND
    ],
)
def test_get_required_kvcache_layout(connector_env, use_mla, model_config, expected):
    """Layout selection: None without a model config, None for MLA, HND
    for dense models."""
    fc = connector_env.fc
    if model_config is None:
        config = type("C", (), {"model_config": None})()
    else:
        config = type(
            "C", (), {"model_config": type("M", (), {"use_mla": use_mla})()}
        )()
    assert fc.FlagCXConnector.get_required_kvcache_layout(config) == expected


def test_connector_facade_scheduler_role(connector_env):
    """SCHEDULER role creates only the scheduler side."""
    fc = connector_env.fc
    conn = fc.FlagCXConnector(
        VllmConfigDouble("kv_consumer"), "scheduler", make_kv_cache_config()
    )
    assert conn.connector_scheduler is not None
    assert conn.connector_worker is None
    # Worker-role-only methods must fail their precondition on a scheduler.
    with pytest.raises(AssertionError):
        conn.register_kv_caches({})


def test_connector_facade_worker_role(connector_env):
    """WORKER role creates only the worker side."""
    fc = connector_env.fc
    conn = fc.FlagCXConnector(
        VllmConfigDouble("kv_consumer"), "worker", make_kv_cache_config()
    )
    assert conn.connector_worker is not None
    assert conn.connector_scheduler is None
    with pytest.raises(AssertionError):
        conn.get_num_new_matched_tokens(None, 0)
    conn.connector_worker.shutdown()


def test_connector_scheduler_role_requires_kv_cache_config(connector_env):
    """SCHEDULER role asserts kv_cache_config is not None."""
    fc = connector_env.fc
    with pytest.raises(AssertionError, match="SCHEDULER role"):
        fc.FlagCXConnector(VllmConfigDouble("kv_consumer"), "scheduler", None)


def test_build_kv_connector_stats_default_is_empty(connector_env):
    """build_kv_connector_stats({}) resets to the full empty column set."""
    stats = connector_env.fc.FlagCXConnector.build_kv_connector_stats()
    assert stats.data["transfer_duration"] == []
    assert stats.num_successful_transfers == 0
    assert stats.is_empty()


def test_build_kv_connector_stats_none_data(connector_env):
    """build_kv_connector_stats(None) behaves like the empty default."""
    stats = connector_env.fc.FlagCXConnector.build_kv_connector_stats(None)
    assert stats.is_empty()


def test_get_kv_connector_stats_passthrough(connector_env):
    """The facade returns worker stats; None before anything is recorded."""
    conn = connector_env.fc.FlagCXConnector(
        VllmConfigDouble("kv_consumer"), "worker", make_kv_cache_config()
    )
    assert conn.get_kv_connector_stats() is None
    worker = conn.connector_worker
    worker.xfer_stats.record_transfer(0.1, 100, 1)
    stats = conn.get_kv_connector_stats()
    assert stats.data["transfer_duration"] == [0.1]
    # Snapshot consumed: the next call reports nothing new.
    assert conn.get_kv_connector_stats() is None
    worker.shutdown()


def test_scheduler_builds_blocks_per_sw(connector_env):
    """Sliding-window groups compute ceil(window/block)+1; others get 0.

    The clipping tables are only built when the hybrid KV cache manager is
    enabled (disable_hybrid_kv_cache_manager=False) and a non-full-attention
    group exists."""
    fc = connector_env.fc
    from tests.unit_tests.coverage.distributed._flagcx_stubs import (
        _FullSpec,
        _Group,
        _SlidingSpec,
    )

    config = make_kv_cache_config(
        groups=[
            _Group(_SlidingSpec(128, 16), ["sw_layer"]),
            _Group(_FullSpec(), ["full_layer"]),
        ]
    )
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer", disable_hybrid_kv_cache_manager=False),
        "e0",
        config,
    )
    assert sched._is_hma_required is True
    assert sched.blocks_per_sw == [9, 0]

    # Hybrid manager disabled: HMA not required, table still built with 0s
    # for full-attention-only groups.
    plain = fc.FlagCXConnectorScheduler(VllmConfigDouble("kv_consumer"), "e0", config)
    assert plain._is_hma_required is False
    assert plain.blocks_per_sw == [9, 0]


def test_scheduler_get_sw_clipped_blocks(connector_env):
    """Clipping keeps only the trailing blocks_per_sw per group when HMA is
    required; without HMA the ids pass through unchanged (copied)."""
    fc = connector_env.fc
    from tests.unit_tests.coverage.distributed._flagcx_stubs import (
        _FullSpec,
        _Group,
        _SlidingSpec,
    )

    config = make_kv_cache_config(
        groups=[
            _Group(_SlidingSpec(32, 16), ["sw"]),
            _Group(_FullSpec(), ["full"]),
        ]
    )
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer", disable_hybrid_kv_cache_manager=False),
        "e0",
        config,
    )
    clipped = sched.get_sw_clipped_blocks([list(range(8)), [1, 2, 3]])
    # cdiv(32, 16) + 1 = 3 trailing blocks kept for the sliding group; the
    # full-attention group's table entry is 0 -> unchanged.
    assert clipped == [[5, 6, 7], [1, 2, 3]]

    # Empty input returns an empty list (no iteration).
    assert sched.get_sw_clipped_blocks([]) == []

    # Non-HMA config: identity (but copied) list.
    plain = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    blocks = [[1, 2], [3]]
    out = plain.get_sw_clipped_blocks(blocks)
    assert out == blocks and out is not blocks


class _FakeRequest:
    """Minimal Request double driving the scheduler paths."""

    def __init__(self, request_id, params=None, num_prompt_tokens=10):
        self.request_id = request_id
        self.kv_transfer_params = params
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_token_ids = list(range(num_prompt_tokens))
        self.prompt_embeds = None
        self.max_tokens = 64
        self._all_token_ids = list(range(num_prompt_tokens + 5))
        self.status = 3  # RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.parametrize(
    ("has_mamba", "num_prompt", "expected"),
    [
        (False, 10, 10),
        (True, 10, 9),  # mamba: decoder recomputes the last token
        (True, 1, 1),  # single-token prompts are not truncated
    ],
)
def test_scheduler_remote_prefill_token_count(
    connector_env, has_mamba, num_prompt, expected
):
    """D-side matched tokens: N for dense, N-1 for mamba (last token
    recomputed locally)."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"),
        "e0",
        make_kv_cache_config(has_mamba=has_mamba),
    )
    assert sched._get_remote_prefill_token_count(num_prompt) == expected


def test_scheduler_truncates_mamba_request_for_prefill_once(connector_env):
    """P-side mamba truncation drops the last prompt token and is guarded by
    a params flag so a preempted+rescheduled request is not truncated twice."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config(has_mamba=True)
    )
    req = _FakeRequest("m1", params={"do_remote_decode": True}, num_prompt_tokens=4)
    sched._truncate_mamba_request_for_prefill(req)
    assert req.num_prompt_tokens == 3
    assert req.prompt_token_ids == [0, 1, 2]
    assert req._all_token_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8][:8]
    assert req.max_tokens == 1
    assert req.kv_transfer_params["_p_side_truncated"] is True

    # Second call is a no-op thanks to the guard.
    sched._truncate_mamba_request_for_prefill(req)
    assert req.num_prompt_tokens == 3


def test_scheduler_truncates_via_prompt_embeds_when_no_token_ids(connector_env):
    """When only prompt_embeds is set the truncation slices the embeds."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config(has_mamba=True)
    )
    req = _FakeRequest("m2", params={"do_remote_decode": True}, num_prompt_tokens=4)
    req.prompt_token_ids = None
    req.prompt_embeds = [1, 2, 3, 4]
    sched._truncate_mamba_request_for_prefill(req)
    assert req.prompt_embeds == [1, 2, 3]
    assert req.num_prompt_tokens == 3


def test_scheduler_truncation_skips_single_token(connector_env):
    """A single-token prompt is never truncated."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config(has_mamba=True)
    )
    req = _FakeRequest("m3", params={"do_remote_decode": True}, num_prompt_tokens=1)
    sched._truncate_mamba_request_for_prefill(req)
    assert req.num_prompt_tokens == 1
    assert req.prompt_token_ids == [0]


def test_scheduler_num_new_matched_remote_prefill(connector_env):
    """do_remote_prefill returns remaining prompt tokens minus already
    computed ones, flagged as cache-hit."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    req = _FakeRequest("r1", params={"do_remote_prefill": True}, num_prompt_tokens=10)
    matched, cached = sched.get_num_new_matched_tokens(req, 3)
    assert matched == 7
    assert cached is True


def test_scheduler_num_new_matched_no_remaining_returns_zero(connector_env):
    """When everything is already computed the request is not matched."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    req = _FakeRequest("r1", params={"do_remote_prefill": True}, num_prompt_tokens=4)
    matched, cached = sched.get_num_new_matched_tokens(req, 4)
    assert (matched, cached) == (0, False)


def test_scheduler_num_new_matched_without_params_returns_zero(connector_env):
    """Requests without kv_transfer_params never match remote prefill."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    req = _FakeRequest("r1")
    assert sched.get_num_new_matched_tokens(req, 0) == (0, False)


def test_scheduler_num_new_matched_triggers_mamba_truncation(connector_env):
    """A do_remote_decode mamba request gets truncated once when polled."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config(has_mamba=True)
    )
    req = _FakeRequest("r2", params={"do_remote_decode": True}, num_prompt_tokens=5)
    assert sched.get_num_new_matched_tokens(req, 0) == (0, False)
    assert req.num_prompt_tokens == 4


def test_scheduler_update_state_after_alloc_recv(connector_env):
    """do_remote_prefill with complete params registers the request for
    receiving with clipped unhashed block ids and clears the flag."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    params = {
        "do_remote_prefill": True,
        "remote_host": "h",
        "remote_port": 1,
        "transfer_id": "t9",
    }
    req = _FakeRequest("rr", params=params)
    blocks = type(
        "B",
        (),
        {"get_unhashed_block_ids_all_groups": lambda self: [[7, 8]]},
    )()
    sched.update_state_after_alloc(req, blocks, num_external_tokens=5)
    stored_req, stored_blocks = sched._reqs_need_recv["rr"]
    assert stored_req is req
    assert stored_blocks == [[7, 8]]
    assert params["do_remote_prefill"] is False


def test_scheduler_update_state_after_alloc_recv_no_external_tokens(connector_env):
    """Zero external tokens register with empty block ids."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    params = {
        "do_remote_prefill": True,
        "remote_host": "h",
        "remote_port": 1,
        "transfer_id": "t9",
    }
    req = _FakeRequest("rr", params=params)
    blocks = type(
        "B",
        (),
        {"get_unhashed_block_ids_all_groups": lambda self: [[7, 8]]},
    )()
    sched.update_state_after_alloc(req, blocks, num_external_tokens=0)
    assert sched._reqs_need_recv["rr"][1] == []


def test_scheduler_update_state_after_alloc_invalid_params(connector_env):
    """Missing endpoint keys leave the request unregistered but still clear
    the do_remote_prefill flag."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    params = {"do_remote_prefill": True, "remote_host": "h"}
    req = _FakeRequest("rr", params=params)
    sched.update_state_after_alloc(req, object(), num_external_tokens=0)
    assert "rr" not in sched._reqs_need_recv
    assert params["do_remote_prefill"] is False


def test_scheduler_update_state_after_alloc_send(connector_env):
    """do_remote_decode registers the request for sending; a missing
    transfer_id only warns and skips."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config()
    )
    req = _FakeRequest("rs", params={"do_remote_decode": True, "transfer_id": "t1"})
    sched.update_state_after_alloc(req, None, 0)
    assert "rs" in sched._reqs_need_send

    req_bad = _FakeRequest("rs2", params={"do_remote_decode": True})
    sched.update_state_after_alloc(req_bad, None, 0)
    assert "rs2" not in sched._reqs_need_send


def test_scheduler_update_state_after_alloc_no_params_noop(connector_env):
    """Requests without params are ignored entirely."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    req = _FakeRequest("rn", params=None)
    sched.update_state_after_alloc(req, None, 0)
    assert not sched._reqs_need_recv and not sched._reqs_need_send


def test_scheduler_build_connector_meta_partitions_by_role(connector_env):
    """kv_consumer builds only recv entries, kv_producer only send entries;
    kv_both builds both; the queues drain afterwards."""
    fc = connector_env.fc

    def make_sched(role):
        sched = fc.FlagCXConnectorScheduler(
            VllmConfigDouble(role), "e0", make_kv_cache_config()
        )
        sched.side_channel_host = "10.0.0.1"
        return sched

    recv_req = _FakeRequest(
        "recv-1",
        params={
            "do_remote_prefill": True,
            "remote_host": "h",
            "remote_port": 1,
            "transfer_id": "tr",
        },
    )
    send_req = _FakeRequest(
        "send-1",
        params={"do_remote_decode": True, "transfer_id": "ts"},
    )
    consumer_meta = make_sched("kv_consumer")
    consumer_meta._reqs_need_recv["recv-1"] = (recv_req, [[1]])
    consumer_meta._reqs_need_send["send-1"] = (send_req, [[2]])
    meta = consumer_meta.build_connector_meta(None)
    assert list(meta.reqs_to_recv) == ["recv-1"]
    assert meta.reqs_to_send == {}
    assert not consumer_meta._reqs_need_recv

    producer_meta = make_sched("kv_producer")
    producer_meta._reqs_need_recv["recv-1"] = (recv_req, [[1]])
    producer_meta._reqs_need_send["send-1"] = (send_req, [[2]])
    meta = producer_meta.build_connector_meta(None)
    assert meta.reqs_to_recv == {}
    assert list(meta.reqs_to_send) == ["send-1"]
    assert not producer_meta._reqs_need_send

    both = make_sched("kv_both")
    both._reqs_need_recv["recv-1"] = (recv_req, [[1]])
    both._reqs_need_send["send-1"] = (send_req, [[2]])
    meta = both.build_connector_meta(None)
    assert list(meta.reqs_to_recv) == ["recv-1"]
    assert list(meta.reqs_to_send) == ["send-1"]


def test_scheduler_request_finished_no_transfer_id(connector_env):
    """Requests without params or transfer_id are never delayed."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config()
    )
    assert sched.request_finished(_FakeRequest("a"), ([],)) == (False, None)
    assert sched.request_finished(_FakeRequest("b", params={"other": 1}), ([],)) == (
        False,
        None,
    )


def test_scheduler_request_finished_remote_prefill_deferred(connector_env):
    """A finishing decode-side request with do_remote_prefill re-registers
    for the next round instead of freeing blocks."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    params = {"do_remote_prefill": True, "transfer_id": "t1"}
    req = _FakeRequest("f1", params=params)
    assert sched.request_finished(req, ([1],)) == (False, None)
    assert "f1" in sched._reqs_need_recv
    assert params["do_remote_prefill"] is False


def test_scheduler_request_finished_producer_hands_off(connector_env):
    """A length-capped do_remote_decode request frees its blocks and returns
    the consumer-side bootstrap params."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config()
    )
    sched.side_channel_host = "10.0.0.7"
    params = {"do_remote_decode": True, "transfer_id": "t5"}
    req = _FakeRequest("f2", params=params)
    delay_free, handoff = sched.request_finished(req, ([1, 2], [3]))
    assert delay_free is True
    assert handoff == {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_host": "10.0.0.7",
        "remote_port": sched.side_channel_port,
        "remote_tp_size": 1,
        "transfer_id": "t5",
    }
    assert "f2" in sched._reqs_need_send


def test_scheduler_request_finished_not_length_capped(connector_env):
    """Non-length-capped finishes do not trigger the KV handoff."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config()
    )
    req = _FakeRequest("f3", params={"do_remote_decode": True, "transfer_id": "t6"})
    req.status = 1  # something other than FINISHED_LENGTH_CAPPED
    assert sched.request_finished(req, ([1],)) == (False, None)


def test_scheduler_request_finished_empty_blocks_no_delay(connector_env):
    """All-empty block groups mean no delayed free, but the handoff params
    are still returned."""
    fc = connector_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config()
    )
    req = _FakeRequest("f4", params={"do_remote_decode": True, "transfer_id": "t7"})
    delay_free, handoff = sched.request_finished(req, ([], []))
    assert delay_free is False
    assert handoff is not None and handoff["transfer_id"] == "t7"
    assert "f4" not in sched._reqs_need_send


def test_scheduler_side_channel_port_env(connector_env, monkeypatch):
    """The side-channel port is the bootstrap base plus dp_rank * tp_size."""
    fc = connector_env.fc
    monkeypatch.setenv("FLAGCX_BOOTSTRAP_PORT", "9000")
    config = make_kv_cache_config()
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer", dp_rank=1, tp_size=4), "e0", config
    )
    assert sched.side_channel_port == 9004
    other = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer", dp_rank=0, tp_size=1), "e0", config
    )
    assert other.side_channel_port == 9000


def test_scheduler_side_channel_port_default(connector_env, monkeypatch):
    """Without the env var the bootstrap port defaults to 8998."""
    fc = connector_env.fc
    monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
    )
    assert sched.side_channel_port == 8998

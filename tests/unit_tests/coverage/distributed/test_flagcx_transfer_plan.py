# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU tests for FlagCX transfer planning and the device communicator.

Deepens ``test_flagcx_connector.py`` and
``test_flagcx_connector_worker.py`` (same directory) with the still-uncovered
surface of both FlagCX modules:

* ``vllm_fl/distributed/kv_transfer/flagcx_connector.py`` — independent-oracle
  grids for the module-level planning helpers (``group_concurrent_contiguous``,
  ``_get_tp_ratio``, ``_expand_transfer_regions``, ``_compute_sender_transfer_plan``,
  ``_can_coalesce_block_transfers``, ``_align_transfer_regions``), the facade
  pass-throughs, ``register_kv_caches`` with real CPU tensors, the ZMQ sender
  thread/worker loops against fake sockets, ``_send_kv_to_decode`` success and
  bookkeeping paths, ``_build_transfer_params`` coalesced vs per-block
  descriptor oracles, ``_receive_kv`` exception paths and the worker
  ``start_load_kv``/``shutdown`` lifecycle lines;
* ``vllm_fl/distributed/device_communicators/flagcx.py`` — loaded from source
  under a unique module name (the core-suite pattern from
  ``tests/unit_tests/distributed/test_flagcx.py``): the wheel/AMD/Kunlunxin API
  loaders, ``is_trace_tensor``, ``PyFlagcxCommunicator.__init__`` device and
  unique-id branches, ``destroy`` on TXDA, and every collective
  (all_reduce/all_gather/all_gatherv/reduce_scatter/reduce_scatterv/send/recv/
  broadcast/group_start/group_end) against a mocked native library.

The FlagCX C library is never present: every native call goes through the
``plugin.interservice.flagcx_wrapper`` boundary stubbed per
``_flagcx_stubs.py`` conventions, and all global state (sys.modules entries,
``sys.path``, environment variables, module-level torch attributes) is
restored via monkeypatch.
"""

import asyncio
import contextlib
import ctypes
import importlib.util
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit_tests.coverage.distributed._flagcx_stubs import (
    FlagcxStubEnv,
    MambaSpec,
    MLAAttentionSpec,
    VllmConfigDouble,
    _FullSpec,
    _Group,
    make_kv_cache_config,
)

pytest.importorskip("msgspec")
pytest.importorskip("numpy")
pytest.importorskip("zmq")
pytest.importorskip("torch")

import torch  # noqa: E402

_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_CONNECTOR_SOURCE = _SOURCE_ROOT / "vllm_fl/distributed/kv_transfer/flagcx_connector.py"
_FLAGCX_SOURCE = _SOURCE_ROOT / "vllm_fl/distributed/device_communicators/flagcx.py"
_COMMUNICATOR_SUBJECT = "_test_flagcx_transfer_plan_communicator"

# API source for wheel/kunlunxin loader fixtures: provides all six names the
# module under test binds at import time.
_API_SOURCE = """\
class flagcxUniqueId:
    def __init__(self):
        self.internal = bytearray(8)

class FLAGCXLibrary:
    pass

buffer_type = flagcxComm_t = flagcxDataTypeEnum = flagcxRedOpTypeEnum = object()
"""


class _StubBackend:
    """Attention backend double for the worker's backend probing."""

    def get_name(self):
        return "stub_backend"


# ---------------------------------------------------------------------------
# Fixtures: connector environment
# ---------------------------------------------------------------------------


@pytest.fixture
def conn_env(monkeypatch):
    """Install stubs and import the connector module, like the existing
    connector fixtures, but expose the worker factory and module together."""
    env = FlagcxStubEnv(monkeypatch).install()
    import vllm_fl.distributed.kv_transfer.flagcx_connector as fc

    monkeypatch.setattr(fc, "get_current_attn_backends", lambda cfg: [_StubBackend()])
    env.fc = fc

    def make_worker(role="kv_both", config=None):
        return fc.FlagCXConnectorWorker(
            VllmConfigDouble(role),
            "e0",
            config if config is not None else make_kv_cache_config(),
        )

    env.make_worker = make_worker
    return env


class _PlainTopo:
    """kv_topo double with the smallest honest surface."""

    local_replicates_kv_cache = False
    is_kv_layout_blocks_first = False

    def handshake_target_ranks(self, remote_tp_size):
        return [0]


class _SplitTopo:
    """kv_topo double exposing a blocks-first layout and a replicated cache."""

    local_replicates_kv_cache = True
    is_kv_layout_blocks_first = True

    def handshake_target_ranks(self, remote_tp_size):
        return [0]


def _armed_send_meta(fc, transfer_id, block_ids, p_req_id="p1"):
    """A ready-to-send block meta with its event already set."""
    meta = fc.SendBlockMeta(
        p_req_id=p_req_id,
        transfer_id=transfer_id,
        local_block_ids=block_ids,
        ready=threading.Event(),
    )
    meta.ready.set()
    meta.need_send = 1
    return meta


def _agent_meta(fc, **overrides):
    """Build a FlagCXAgentMetadata with one matching remote layer region."""
    kwargs = dict(
        remote_hostname="10.0.0.1",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d1": ("t1", [[5, 6, 7]])},
        kv_caches_base_addr=[5000],
        block_lens=[64],
        kv_block_lens=[64],
        registered_layer_names=["model.layers.0.self_attn.attn"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )
    kwargs.update(overrides)
    return fc.FlagCXAgentMetadata(**kwargs)


def _registered_local_layer(worker):
    """Give the worker one registered full-attention layer region."""
    worker.kv_caches_base_addr = [1000]
    worker.block_len_per_layer = [64]
    worker.kv_block_len_per_layer = [64]
    worker.registered_layer_names = ["model.layers.0.self_attn.attn"]
    worker.registered_layer_indices = [0]
    worker.registered_group_indices = [0]


# ---------------------------------------------------------------------------
# Module-level helpers: independent-oracle grids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "dst", "exp_src", "exp_dst"),
    [
        ([1, 2, 3], [10, 11, 12], [[1, 2, 3]], [[10, 11, 12]]),
        ([1, 2, 4], [10, 11, 12], [[1, 2], [4]], [[10, 11], [12]]),
        ([1, 2, 3], [10, 11, 13], [[1, 2], [3]], [[10, 11], [13]]),
        ([5], [9], [[5]], [[9]]),
    ],
)
def test_group_concurrent_contiguous_oracle(conn_env, src, dst, exp_src, exp_dst):
    """Independent oracle: split runs wherever a hand-computed boundary
    exists (either side's consecutive difference is not 1)."""
    # Oracle: walk pairs and cut before each index where either side jumps.
    cuts = [
        i
        for i in range(1, len(src))
        if src[i] - src[i - 1] != 1 or dst[i] - dst[i - 1] != 1
    ]
    src_groups, dst_groups = conn_env.fc.group_concurrent_contiguous(src, dst)
    assert src_groups == exp_src
    assert dst_groups == exp_dst
    assert (src_groups, dst_groups) == _oracle_split(src, dst, cuts)


def _oracle_split(src, dst, cuts):
    """Hand oracle for group_concurrent_contiguous given cut positions."""
    bounds = [0, *cuts, len(src)]
    return (
        [src[a:b] for a, b in zip(bounds, bounds[1:])],
        [dst[a:b] for a, b in zip(bounds, bounds[1:])],
    )


def test_group_concurrent_contiguous_multi_run_oracle(conn_env):
    """Mixed gaps produce alternating grouped/singleton runs, verified by a
    from-scratch run-length loop rather than the implementation's diff."""
    src = [0, 1, 2, 9, 10, 40, 41, 90]
    dst = [100, 101, 102, 200, 201, 300, 301, 400]
    src_groups, dst_groups = conn_env.fc.group_concurrent_contiguous(src, dst)

    # Oracle: accumulate runs while both sides stay contiguous.
    oracle_src, oracle_dst = [], []
    run_src: list = [src[0]]
    run_dst: list = [dst[0]]
    for i in range(1, len(src)):
        if src[i] - src[i - 1] == 1 and dst[i] - dst[i - 1] == 1:
            run_src.append(src[i])
            run_dst.append(dst[i])
        else:
            oracle_src.append(run_src)
            oracle_dst.append(run_dst)
            run_src, run_dst = [src[i]], [dst[i]]
    oracle_src.append(run_src)
    oracle_dst.append(run_dst)

    assert (src_groups, dst_groups) == (oracle_src, oracle_dst)
    assert [len(g) for g in src_groups] == [3, 2, 2, 1]


@pytest.mark.parametrize(
    ("local", "remote", "expected"),
    [
        (1, 1, 1),
        (8, 8, 1),
        (4, 2, 2),
        (8, 2, 4),
        (6, 3, 2),
        (2, 4, -2),
        (1, 8, -8),
        (3, 9, -3),
    ],
)
def test_get_tp_ratio_oracle_grid(conn_env, local, remote, expected):
    """Independent oracle: sign follows the larger side, magnitude is the
    integer division of larger by smaller."""
    if local >= remote:
        assert local % remote == 0
        ratio = local // remote
    else:
        assert remote % local == 0
        ratio = -(remote // local)
    assert ratio == expected
    assert conn_env.fc._get_tp_ratio(local, remote) == expected


def test_get_tp_ratio_all_pairs_up_to_eight(conn_env):
    """Exhaustive grid over TP sizes 1..8: every divisible pair matches the
    oracle and non-divisible pairs raise."""
    fc = conn_env.fc
    for local in range(1, 9):
        for remote in range(1, 9):
            if local >= remote:
                if local % remote:
                    with pytest.raises(AssertionError):
                        fc._get_tp_ratio(local, remote)
                else:
                    assert fc._get_tp_ratio(local, remote) == local // remote
            elif remote % local:
                with pytest.raises(AssertionError):
                    fc._get_tp_ratio(local, remote)
            else:
                assert fc._get_tp_ratio(local, remote) == -(remote // local)


def test_expand_transfer_regions_oracle_grid(conn_env):
    """Independent oracle: each split entry becomes K at base and V at
    base + kv_block_len; unsplit entries stay single."""
    fc = conn_env.fc
    base_addrs = [1000, 2000, 3000]
    block_lens = [64, 64, 64]
    kv_block_lens = [32, 16, 32]
    layer_names = ["a", "b", "c"]
    layer_indices = [0, 1, 2]
    split = [True, False, True]
    group_indices = [0, 1, 0]

    regions = fc._expand_transfer_regions(
        base_addrs,
        block_lens,
        kv_block_lens,
        layer_names,
        layer_indices,
        is_kv_layout_blocks_first=True,
        group_indices=group_indices,
        split_kv_regions=split,
    )

    expected = []  # (layer, index, base, block_len, kv_block_len, group)
    for i in range(3):
        expected.append(
            (
                layer_names[i],
                layer_indices[i],
                base_addrs[i],
                64,
                kv_block_lens[i],
                group_indices[i],
            )
        )
        if split[i]:
            expected.append(
                (
                    layer_names[i],
                    layer_indices[i],
                    base_addrs[i] + kv_block_lens[i],
                    64,
                    kv_block_lens[i],
                    group_indices[i],
                )
            )
    assert [
        (
            r.layer_name,
            r.layer_index,
            r.base_addr,
            r.block_len,
            r.kv_block_len,
            r.group_index,
        )
        for r in regions
    ] == expected


def test_expand_transfer_regions_defaults_fill_lists(conn_env):
    """Omitted group_indices default to 0; omitted split list defaults to the
    layout flag for every entry."""
    fc = conn_env.fc
    regions = fc._expand_transfer_regions(
        [10, 20],
        [8, 8],
        [4, 4],
        ["x", "y"],
        [0, 1],
        is_kv_layout_blocks_first=True,
    )
    assert [r.group_index for r in regions] == [0] * 4
    assert [r.base_addr for r in regions] == [10, 14, 20, 24]

    unsplit = fc._expand_transfer_regions(
        [10], [8], [4], ["x"], [0], is_kv_layout_blocks_first=False
    )
    assert len(unsplit) == 1 and unsplit[0].base_addr == 10


@pytest.mark.parametrize(
    (
        "local_rank",
        "local_tp",
        "remote_rank",
        "remote_tp",
        "replicated",
        "local_len",
        "remote_len",
        "expected",
    ),
    [
        # ratio == 1: full copy.
        (0, 2, 1, 2, False, 128, 128, (True, 0, 0, 128)),
        # ratio 2 replicated: only multiples of the ratio send.
        (0, 2, 0, 1, True, 128, 256, (True, 0, 0, 128)),
        (1, 2, 0, 1, True, 128, 256, (False, 0, 0, 128)),
        # ratio 2 sharded: destination slice at (rank % ratio) * local_len.
        (1, 2, 0, 1, False, 128, 256, (True, 0, 128, 128)),
        # ratio -2 replicated: full copy from each remote.
        (0, 1, 3, 2, True, 128, 64, (True, 0, 0, 128)),
        # ratio -2 sharded: source slice from the remote rank.
        (0, 1, 1, 2, False, 128, 64, (True, 64, 0, 64)),
    ],
)
def test_compute_sender_transfer_plan_oracle(
    conn_env,
    local_rank,
    local_tp,
    remote_rank,
    remote_tp,
    replicated,
    local_len,
    remote_len,
    expected,
):
    """The plan matches the documented invariant per ratio regime, computed
    independently from the tp sizes in the test."""
    fc = conn_env.fc
    ratio = local_tp // remote_tp if local_tp >= remote_tp else -(remote_tp // local_tp)
    # Independent expectation per regime.
    if ratio == 1:
        assert expected == (True, 0, 0, local_len)
    elif ratio > 0:
        if replicated:
            assert expected == (local_rank % ratio == 0, 0, 0, local_len)
        else:
            assert expected[1:] == (0, (local_rank % ratio) * local_len, local_len)
    else:
        ratio_abs = -ratio
        if replicated:
            assert expected == (True, 0, 0, local_len)
        else:
            assert expected[1:] == (
                (remote_rank % ratio_abs) * remote_len,
                0,
                remote_len,
            )

    plan = fc._compute_sender_transfer_plan(
        local_tp_rank=local_rank,
        local_tp_size=local_tp,
        remote_tp_rank=remote_rank,
        remote_tp_size=remote_tp,
        local_kv_block_len=local_len,
        remote_kv_block_len=remote_len,
        producer_cache_replicated=replicated,
    )
    assert plan == expected


def test_compute_sender_transfer_plan_exhaustive_grid(conn_env):
    """Every (local_tp, remote_tp, ranks) combination under 4 matches a
    from-scratch oracle."""
    fc = conn_env.fc
    for local_tp in (1, 2, 4):
        for remote_tp in (1, 2, 4):
            ratio = (
                local_tp // remote_tp
                if local_tp >= remote_tp
                else -(remote_tp // local_tp)
            )
            for local_rank in range(local_tp):
                for remote_rank in range(remote_tp):
                    for replicated in (False, True):
                        remote_len = 64 * abs(ratio)
                        plan = fc._compute_sender_transfer_plan(
                            local_rank,
                            local_tp,
                            remote_rank,
                            remote_tp,
                            64,
                            remote_len,
                            replicated,
                        )
                        should, src_off, dst_off, length = plan
                        if ratio == 1:
                            assert (should, src_off, dst_off, length) == (
                                True,
                                0,
                                0,
                                64,
                            )
                        elif ratio > 0:
                            if replicated:
                                assert should == (local_rank % ratio == 0)
                                assert (src_off, dst_off, length) == (0, 0, 64)
                            else:
                                assert (should, src_off, length) == (True, 0, 64)
                                assert dst_off == (local_rank % ratio) * 64
                        else:
                            ratio_abs = -ratio
                            if replicated:
                                assert plan == (True, 0, 0, 64)
                            else:
                                assert (should, dst_off, length) == (
                                    True,
                                    0,
                                    remote_len,
                                )
                                assert src_off == (remote_rank % ratio_abs) * remote_len


@pytest.mark.parametrize(
    ("block_lens", "offsets", "length", "expected"),
    [
        ((64, 64), (0, 0), 64, True),
        ((64, 128), (0, 0), 64, False),
        ((64, 64), (8, 0), 64, False),
        ((64, 64), (0, 8), 64, False),
        ((64, 64), (0, 0), 32, False),
        ((128, 128), (0, 0), 128, True),
    ],
)
def test_can_coalesce_oracle(conn_env, block_lens, offsets, length, expected):
    """Coalescing is exactly: zero offsets and full-length on both sides."""
    (local_len, remote_len), (src_off, dst_off) = block_lens, offsets
    manual = (
        src_off == 0 and dst_off == 0 and length == local_len and length == remote_len
    )
    assert manual == expected
    assert (
        conn_env.fc._can_coalesce_block_transfers(
            local_len, remote_len, src_off, dst_off, length
        )
        is expected
    )


def test_align_transfer_regions_reordered_remote(conn_env):
    """Alignment reorders the remote list to follow the local order of
    (layer_name, occurrence) pairs."""
    fc = conn_env.fc

    def region(layer, occurrence_base, layer_index=0, group=0):
        return fc.TransferRegion(
            layer_name=layer,
            layer_index=layer_index,
            base_addr=occurrence_base,
            block_len=64,
            kv_block_len=32,
            group_index=group,
        )

    local = [region("a", 10), region("b", 20), region("a", 30)]
    remote = [region("b", 120), region("a", 110), region("a", 130)]
    aligned_local, aligned_remote, err = fc._align_transfer_regions(local, remote)
    assert err is None
    assert [r.base_addr for r in aligned_local] == [10, 20, 30]
    assert [r.base_addr for r in aligned_remote] == [110, 120, 130]


# ---------------------------------------------------------------------------
# Facade pass-throughs
# ---------------------------------------------------------------------------


def test_facade_scheduler_passthroughs_return_inner_results(conn_env):
    """Each scheduler facade method forwards its arguments and returns the
    scheduler's result unchanged."""
    fc = conn_env.fc
    conn = fc.FlagCXConnector(
        VllmConfigDouble("kv_consumer"), "scheduler", make_kv_cache_config()
    )
    sched = conn.connector_scheduler
    sent = []
    sched.get_num_new_matched_tokens = lambda req, n: (
        sent.append(("gnmt", req, n)) or (5, True)
    )
    sched.update_state_after_alloc = lambda req, blocks, n: sent.append(
        ("usaa", req, blocks, n)
    )
    sched.build_connector_meta = lambda out: sent.append(("bcm", out)) or "meta"
    sched.request_finished = lambda req, blocks: (
        sent.append(("rf", req, blocks)) or (True, {"k": 1})
    )

    assert conn.get_num_new_matched_tokens("req", 3) == (5, True)
    conn.update_state_after_alloc("req", "blocks", 7)
    assert conn.build_connector_meta("sched-out") == "meta"
    assert conn.request_finished("req", [1]) == (True, {"k": 1})
    assert conn.request_finished_all_groups("req", ([1], [2])) == (True, {"k": 1})
    assert [entry[0] for entry in sent] == ["gnmt", "usaa", "bcm", "rf", "rf"]


def test_facade_worker_passthroughs_and_role_guards(conn_env):
    """Worker facade methods forward to the worker; scheduler-role connectors
    have no worker side (stats None, wait_for_layer_load no-op)."""
    fc = conn_env.fc
    conn = fc.FlagCXConnector(
        VllmConfigDouble("kv_consumer"), "worker", make_kv_cache_config()
    )
    worker = conn.connector_worker
    sent = []
    worker.register_kv_caches = lambda kv: sent.append(("rkc", kv))
    worker.get_finished = lambda: sent.append("gf") or ({"a"}, {"b"})
    worker.start_load_kv = lambda meta: sent.append(("slk", meta))
    worker.wait_for_layer_load = lambda: sent.append("wfll")
    worker.get_kv_connector_stats = lambda: sent.append("gkcs") or "stats"

    conn.register_kv_caches({"l": object()})
    assert conn.get_finished(set()) == ({"a"}, {"b"})
    conn._connector_metadata = fc.FlagCXConnectorMetadata()
    conn.start_load_kv(object())
    conn.wait_for_layer_load("layer.0")
    assert conn.get_kv_connector_stats() == "stats"
    assert [entry[0] if isinstance(entry, tuple) else entry for entry in sent] == [
        "rkc",
        "gf",
        "slk",
        "wfll",
        "gkcs",
    ]

    # save_kv_layer / wait_for_save are documented no-ops.
    conn.save_kv_layer("l", None, None)
    conn.wait_for_save()

    sched_conn = fc.FlagCXConnector(
        VllmConfigDouble("kv_consumer"), "scheduler", make_kv_cache_config()
    )
    assert sched_conn.get_kv_connector_stats() is None
    sched_conn.wait_for_layer_load("layer.0")  # no worker: silent no-op

    # start_load_kv asserts the metadata was built first.
    conn._connector_metadata = None
    with pytest.raises(AssertionError):
        conn.start_load_kv(object())


def test_facade_build_prom_metrics(conn_env):
    """build_prom_metrics constructs the FlagCX Prometheus collector."""
    prom = conn_env.fc.FlagCXConnector.build_prom_metrics(
        object(),
        {"counter": object, "histogram": object},
        ["model_name"],
        {0: ["m=e1"]},
    )
    assert type(prom).__name__ == "FlagCXPromMetrics"


# ---------------------------------------------------------------------------
# Scheduler mamba truncation edge
# ---------------------------------------------------------------------------


def test_scheduler_truncate_mamba_without_prompt_payload_is_noop(conn_env):
    """A request with neither prompt_token_ids nor prompt_embeds is left
    untouched (no truncation, no guard flag)."""
    fc = conn_env.fc
    sched = fc.FlagCXConnectorScheduler(
        VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config(has_mamba=True)
    )

    class _BareReq:
        kv_transfer_params = {"do_remote_decode": True}
        num_prompt_tokens = 4
        prompt_token_ids = None
        prompt_embeds = None
        _all_token_ids = [1, 2, 3, 4, 5]
        max_tokens = 64

    req = _BareReq()
    sched._truncate_mamba_request_for_prefill(req)
    assert req.num_prompt_tokens == 4
    assert req.max_tokens == 64
    assert "_p_side_truncated" not in req.kv_transfer_params


# ---------------------------------------------------------------------------
# Worker: block-size sync and kernel block ids
# ---------------------------------------------------------------------------


def test_worker_sync_block_size_adopts_kernel_size(conn_env, monkeypatch):
    """A smaller kernel block size wins: block_size shrinks and the physical
    fan-out is recorded; matching sizes leave both unchanged."""
    fc = conn_env.fc
    worker = conn_env.make_worker()
    try:
        monkeypatch.setattr(fc, "select_common_block_size", lambda bs, backends: 8)
        worker.block_size = 16
        worker._sync_block_size_with_kernel()
        assert worker.block_size == 8
        assert worker._physical_blocks_per_logical_kv_block == 2

        monkeypatch.setattr(fc, "select_common_block_size", lambda bs, backends: 8)
        worker._sync_block_size_with_kernel()  # already 8: stable
        assert worker._physical_blocks_per_logical_kv_block == 2
    finally:
        worker.shutdown()


def test_worker_logical_to_kernel_block_ids_expansion(conn_env, monkeypatch):
    """With a 2:1 logical/kernel ratio, attention-group ids expand to the
    interleaved kernel ids while mamba groups stay logical."""
    fc = conn_env.fc

    class _BlockTable:
        @staticmethod
        def map_to_kernel_blocks(group, ratio, arange):
            return group[:, None] * ratio + arange

    monkeypatch.setattr(fc, "BlockTable", _BlockTable)
    config = make_kv_cache_config(
        groups=[
            _Group(_FullSpec(), ["model.layers.0.self_attn.attn"]),
            _Group(MambaSpec(), ["mamba_layer"]),
        ]
    )
    worker = conn_env.make_worker(config=config)
    try:
        worker._physical_blocks_per_logical_kv_block = 2
        expanded = worker._logical_to_kernel_block_ids([[1, 2], [3, 4]])
        # Oracle: kernel ids b*2 and b*2+1 for each logical block b; the
        # result stays grouped per KV-cache group.
        assert expanded[0] == [[2, 3], [4, 5]]
        assert expanded[1] == [3, 4]  # mamba group untouched
        # 1:1 ratio passes the ids through unchanged.
        worker._physical_blocks_per_logical_kv_block = 1
        assert worker._logical_to_kernel_block_ids([[9], [7]]) == [[9], [7]]
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# Worker: register_kv_caches with real CPU tensors
# ---------------------------------------------------------------------------


def test_worker_register_kv_caches_full_attention_blocks_first(conn_env):
    """Registration records region geometry from real tensor strides, dedups
    MRs by storage, registers them with the library and starts the RPC server
    on the consumer role."""
    worker = conn_env.make_worker(role="kv_consumer")
    try:
        worker.kv_topo = _SplitTopo()

        class _TopoWithRegions(_SplitTopo):
            def get_transfer_cache_regions(self, cache_or_caches, layer_spec):
                return [cache_or_caches]

        worker.kv_topo = _TopoWithRegions()
        tensor = torch.zeros(2, 3, 4)  # stride(0)=12, element_size=4 -> 48 bytes
        worker.register_kv_caches({"model.layers.0.self_attn.attn": tensor})

        assert worker.registered_layer_names == ["model.layers.0.self_attn.attn"]
        assert worker.registered_layer_indices == [0]
        assert worker.block_len_per_layer == [48]
        assert worker.kv_block_len_per_layer == [24]  # blocks-first halves it
        assert worker.kv_caches_base_addr == [tensor.data_ptr()]
        # One MR: (engine handle, storage ptr, storage nbytes).
        conn_env.flagcx_lib.flagcxP2pRegister.assert_called_once_with(
            "engine-handle",
            tensor.untyped_storage().data_ptr(),
            tensor.untyped_storage().nbytes(),
        )
        assert worker.rpc_port == 12345
        conn_env.flagcx_lib.flagcxP2pStartRpcServer.assert_called_once_with(
            "engine-handle"
        )
        assert not hasattr(worker, "_sender_t")
    finally:
        worker.shutdown()


def test_worker_register_kv_caches_producer_starts_sender_thread(conn_env):
    """The producer role skips the RPC server and spawns the (stubbed) sender
    thread, waiting for its ready event."""
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()

        class _TopoWithRegions(_PlainTopo):
            def get_transfer_cache_regions(self, cache_or_caches, layer_spec):
                return [cache_or_caches]

        worker.kv_topo = _TopoWithRegions()
        started = []

        def fake_sender_thread(ready_event, base_port, tp_rank):
            started.append((base_port, tp_rank))
            ready_event.set()

        worker._sender_thread = fake_sender_thread
        worker.register_kv_caches(
            {"model.layers.0.self_attn.attn": torch.zeros(2, 2, 2)}
        )
        assert started == [(worker.side_channel_port, worker.tp_rank)]
        assert worker._sender_t.name == "flagcx_sender"
        assert worker._sender_t.daemon is True
        assert not conn_env.flagcx_lib.flagcxP2pStartRpcServer.called
    finally:
        worker.shutdown()


def test_worker_register_kv_caches_mla_uses_page_size(conn_env):
    """MLA layers take kv_block_len from the spec's page_size_bytes instead
    of halving the block length."""

    class _MLASpec(MLAAttentionSpec):
        page_size_bytes = 4096

    config = make_kv_cache_config(
        groups=[_Group(_MLASpec(), ["model.layers.0.self_attn.attn"])]
    )
    worker = conn_env.make_worker(role="kv_consumer", config=config)
    try:
        worker.kv_topo = _PlainTopo()

        class _TopoWithRegions(_PlainTopo):
            def get_transfer_cache_regions(self, cache_or_caches, layer_spec):
                return [cache_or_caches]

        worker.kv_topo = _TopoWithRegions()
        worker._sender_thread = lambda ready_event, base_port, tp_rank: (
            ready_event.set()
        )
        worker.register_kv_caches(
            {"model.layers.0.self_attn.attn": torch.zeros(2, 3, 4)}
        )
        assert worker.kv_block_len_per_layer == [4096]
        assert worker.block_len_per_layer == [48]
    finally:
        worker.shutdown()


def test_worker_register_kv_caches_skips_unknown_layer(conn_env):
    """Layers without a KV cache spec are skipped; the unknown layer never
    reaches the parallel region lists."""
    worker = conn_env.make_worker(role="kv_consumer")
    try:
        worker.kv_topo = _PlainTopo()
        worker._sender_thread = lambda ready_event, base_port, tp_rank: (
            ready_event.set()
        )
        with pytest.raises(RuntimeError, match="No KV cache tensors"):
            worker.register_kv_caches({"model.layers.9.other": torch.zeros(2, 2)})
        assert worker.registered_layer_names == []
    finally:
        worker.shutdown()


def test_worker_register_kv_caches_dedups_shared_storage(conn_env):
    """Two region views over one storage register the MR exactly once."""
    worker = conn_env.make_worker(role="kv_consumer")
    try:

        class _TopoWithRegions(_PlainTopo):
            def get_transfer_cache_regions(self, cache_or_caches, layer_spec):
                return [cache_or_caches]

        worker.kv_topo = _TopoWithRegions()
        big = torch.zeros(8, 3, 4)
        worker.register_kv_caches(
            {
                "model.layers.0.self_attn.attn": big[:2],
            }
        )
        calls = conn_env.flagcx_lib.flagcxP2pRegister.call_args_list
        assert len(calls) == 1
        assert calls[0].args[1] == big.untyped_storage().data_ptr()
        assert calls[0].args[2] == big.untyped_storage().nbytes()
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# Worker: region and plan plumbing
# ---------------------------------------------------------------------------


def test_worker_get_transfer_regions_splits_blocks_first(conn_env):
    """The worker's region builder splits K/V for blocks-first layouts with
    non-MLA, non-mamba layers and passes the group indices through."""
    worker = conn_env.make_worker(role="kv_consumer")
    try:
        worker.kv_topo = _SplitTopo()
        worker._layer_specs = {"model.layers.0.self_attn.attn": _FullSpec()}
        regions = worker._get_transfer_regions(
            [1000], [64], [32], ["model.layers.0.self_attn.attn"], [0], [0]
        )
        # Oracle: K at base, V at base + kv_block_len.
        assert [(r.base_addr, r.kv_block_len, r.group_index) for r in regions] == [
            (1000, 32, 0),
            (1032, 32, 0),
        ]
        assert worker._producer_cache_is_replicated() is True
    finally:
        worker.shutdown()


def test_worker_get_sender_transfer_plan_binds_worker_state(conn_env, monkeypatch):
    """The plan wrapper forwards the worker's rank/size and replication flag."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        captured = {}

        def fake_plan(**kwargs):
            captured.update(kwargs)
            return (True, 0, 0, 64)

        monkeypatch.setattr(fc, "_compute_sender_transfer_plan", fake_plan)
        result = worker._get_sender_transfer_plan(
            local_kv_block_len=64,
            remote_kv_block_len=128,
            remote_tp_rank=3,
            remote_tp_size=4,
        )
        assert result == (True, 0, 0, 64)
        assert captured["local_tp_rank"] == worker.tp_rank
        assert captured["local_tp_size"] == worker.tp_size
        assert captured["remote_tp_rank"] == 3
        assert captured["remote_tp_size"] == 4
        assert captured["local_kv_block_len"] == 64
        assert captured["remote_kv_block_len"] == 128
        assert captured["producer_cache_replicated"] is False
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# Worker: _build_transfer_params oracles
# ---------------------------------------------------------------------------


def test_worker_build_transfer_params_coalesced_contiguous(conn_env):
    """Contiguous aligned block runs coalesce into one descriptor per run:
    src = base + first_block * block_len, size = transfer_len * run length."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        worker.tp_rank = 0
        worker.tp_size = 1
        send_meta = _armed_send_meta(fc, "t1", [[1, 2, 3]])
        agent = _agent_meta(fc, req_blocks={"d1": ("t1", [[5, 6, 7]])})
        local = fc.TransferRegion("L", 0, 1000, 64, 64)
        remote = fc.TransferRegion("L", 0, 5000, 64, 64)
        src, dst, sizes = worker._build_transfer_params(
            [("d1", send_meta)], agent, [local], [remote]
        )
        # Oracle: one coalesced write 1000+1*64 -> 5000+5*64, len 64*3.
        assert src == [1000 + 1 * 64]
        assert dst == [5000 + 5 * 64]
        assert sizes == [64 * 3]
    finally:
        worker.shutdown()


def test_worker_build_transfer_params_non_coalesced_per_block(conn_env):
    """A sharded plan (nonzero destination offset) emits one descriptor per
    block with the offset applied to every destination address."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        worker.tp_rank = 1
        worker.tp_size = 2
        send_meta = _armed_send_meta(fc, "t1", [[1, 2]])
        agent = _agent_meta(fc, req_blocks={"d1": ("t1", [[3, 4]])})
        local = fc.TransferRegion("L", 0, 1000, 64, 32)
        remote = fc.TransferRegion("L", 0, 5000, 128, 64)
        src, dst, sizes = worker._build_transfer_params(
            [("d1", send_meta)], agent, [local], [remote]
        )
        # ratio 2, rank 1: dst_off = 32, transfer_len = 32; per block.
        assert src == [1000 + 1 * 64, 1000 + 2 * 64]
        assert dst == [5000 + 3 * 128 + 32, 5000 + 4 * 128 + 32]
        assert sizes == [32, 32]
    finally:
        worker.shutdown()


def test_worker_build_transfer_params_mamba_null_blocks_stripped(conn_env):
    """Mamba groups drop NULL_BLOCK_ID placeholders on both sides before
    building descriptors."""
    fc = conn_env.fc
    config = make_kv_cache_config(groups=[_Group(MambaSpec(), ["mamba_layer"])])
    worker = conn_env.make_worker(role="kv_producer", config=config)
    try:
        worker.kv_topo = _PlainTopo()
        send_meta = _armed_send_meta(fc, "t1", [[-1, 4]])
        agent = _agent_meta(fc, req_blocks={"d1": ("t1", [[-1, 6]])})
        local = fc.TransferRegion("L", 0, 1000, 64, 64, group_index=0)
        remote = fc.TransferRegion("L", 0, 5000, 64, 64, group_index=0)
        src, dst, sizes = worker._build_transfer_params(
            [("d1", send_meta)], agent, [local], [remote]
        )
        # Only block 4 -> 6 survives the NULL strip.
        assert src == [1000 + 4 * 64]
        assert dst == [5000 + 6 * 64]
        assert sizes == [64]
    finally:
        worker.shutdown()


def test_worker_build_transfer_params_empty_groups_after_strip_skips(conn_env):
    """When every group empties after the mamba strip the request produces no
    descriptors (the `not any(local_by_group)` guard)."""
    fc = conn_env.fc
    config = make_kv_cache_config(groups=[_Group(MambaSpec(), ["mamba_layer"])])
    worker = conn_env.make_worker(role="kv_producer", config=config)
    try:
        worker.kv_topo = _PlainTopo()
        send_meta = _armed_send_meta(fc, "t1", [[-1]])
        agent = _agent_meta(fc, req_blocks={"d1": ("t1", [[-1]])})
        src, dst, sizes = worker._build_transfer_params(
            [("d1", send_meta)], agent, [], []
        )
        assert (src, dst, sizes) == ([], [], [])
    finally:
        worker.shutdown()


def test_worker_build_transfer_params_group_out_of_range_skipped(conn_env):
    """A region whose group index exceeds the request's group count is
    skipped without descriptors."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        send_meta = _armed_send_meta(fc, "t1", [[1]])
        agent = _agent_meta(fc, req_blocks={"d1": ("t1", [[2]])})
        local = fc.TransferRegion("L", 0, 1000, 64, 64, group_index=5)
        remote = fc.TransferRegion("L", 0, 5000, 64, 64, group_index=5)
        src, dst, sizes = worker._build_transfer_params(
            [("d1", send_meta)], agent, [local], [remote]
        )
        assert (src, dst, sizes) == ([], [], [])
    finally:
        worker.shutdown()


def test_worker_build_transfer_params_multiple_requests(conn_env):
    """Two requests each map their own blocks onto the shared regions."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        first = _armed_send_meta(fc, "t1", [[1]])
        second = _armed_send_meta(fc, "t2", [[2]], p_req_id="p2")
        agent = _agent_meta(
            fc,
            req_blocks={"d1": ("t1", [[7]]), "d2": ("t2", [[8]])},
        )
        local = fc.TransferRegion("L", 0, 1000, 64, 64)
        remote = fc.TransferRegion("L", 0, 5000, 64, 64)
        src, dst, sizes = worker._build_transfer_params(
            [("d1", first), ("d2", second)], agent, [local], [remote]
        )
        assert src == [1000 + 64, 1000 + 2 * 64]
        assert dst == [5000 + 7 * 64, 5000 + 8 * 64]
        assert sizes == [64, 64]
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# Worker: _send_kv_to_decode success and bookkeeping
# ---------------------------------------------------------------------------


def test_worker_send_kv_to_decode_success_path(conn_env):
    """The full happy path: ready blocks, one coalesced BatchWriteSync,
    stats recorded, finished ids published and the send meta drained."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        worker._sender_thread = lambda *args: None
        _registered_local_layer(worker)
        worker.reqs_need_send.reqs["t1"] = _armed_send_meta(fc, "t1", [[1, 2, 3]])
        meta = _agent_meta(fc)
        worker._send_kv_to_decode(meta)

        conn_env.flagcx_lib.flagcxP2pBatchWriteSync.assert_called_once()
        args = conn_env.flagcx_lib.flagcxP2pBatchWriteSync.call_args.args
        assert args[1] == [1000 + 64]  # src VA
        assert args[2] == [5000 + 5 * 64]  # dst VA
        assert args[3] == [64 * 3]  # sizes
        assert worker.finished_sending_reqs.set == {"p1"}
        assert worker.reqs_need_send.reqs == {}
        assert worker.xfer_stats.data["num_descriptors"] == [1]
        assert worker.xfer_stats.data["bytes_transferred"] == [64 * 3]
    finally:
        worker.shutdown()


def test_worker_send_kv_to_decode_sizes_empty_skips_write(conn_env):
    """With no descriptors to write the BatchWriteSync boundary stays
    untouched, yet the request still finishes."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        worker._sender_thread = lambda *args: None
        _registered_local_layer(worker)
        worker.reqs_need_send.reqs["t1"] = _armed_send_meta(fc, "t1", [[]])
        meta = _agent_meta(fc, req_blocks={"d1": ("t1", [[]])})
        worker._send_kv_to_decode(meta)
        assert not conn_env.flagcx_lib.flagcxP2pBatchWriteSync.called
        assert worker.finished_sending_reqs.set == {"p1"}
    finally:
        worker.shutdown()


def test_worker_send_kv_to_decode_missing_prefill_id_raises(conn_env):
    """Completing a transfer whose send meta never bound a prefill request id
    raises instead of publishing an empty id."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        worker._sender_thread = lambda *args: None
        _registered_local_layer(worker)
        worker.reqs_need_send.reqs["t1"] = _armed_send_meta(
            fc, "t1", [[1]], p_req_id=""
        )
        meta = _agent_meta(fc)
        with pytest.raises(RuntimeError, match="Missing Prefill request ID"):
            worker._send_kv_to_decode(meta)
    finally:
        worker.shutdown()


def test_worker_send_kv_to_decode_non_target_rank_warns(conn_env, caplog):
    """A decode rank outside this producer's handshake targets logs a
    warning but proceeds with the transfer."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:

        class _OtherTargetTopo(_PlainTopo):
            def handshake_target_ranks(self, remote_tp_size):
                return [1]

        worker.kv_topo = _OtherTargetTopo()
        worker._sender_thread = lambda *args: None
        _registered_local_layer(worker)
        worker.reqs_need_send.reqs["t1"] = _armed_send_meta(fc, "t1", [[1]])
        meta = _agent_meta(fc, req_blocks={"d1": ("t1", [[5]])})
        with caplog.at_level(
            logging.WARNING, logger="vllm_fl.distributed.kv_transfer.flagcx_connector"
        ):
            worker._send_kv_to_decode(meta)
        assert any(
            "not a valid transfer pair" in message for message in caplog.messages
        )
        assert conn_env.flagcx_lib.flagcxP2pBatchWriteSync.called
    finally:
        worker.shutdown()


def test_worker_send_kv_to_decode_length_validation_raises(conn_env):
    """Region-length validation errors surface as RuntimeError after
    alignment succeeds."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        worker._sender_thread = lambda *args: None
        _registered_local_layer(worker)
        worker.kv_block_len_per_layer = [32]
        worker.reqs_need_send.reqs["t1"] = _armed_send_meta(fc, "t1", [[1]])
        meta = _agent_meta(fc, req_blocks={"d1": ("t1", [[5]])})
        with pytest.raises(RuntimeError, match="length mismatch"):
            worker._send_kv_to_decode(meta)
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# Worker: sender thread / sender worker loops (fake ZMQ boundary)
# ---------------------------------------------------------------------------


class _FakeFrontend:
    """ROUTER-socket double: one queued multipart request, records relays."""

    def __init__(self, request=(b"id", b"", b"meta-bytes")):
        self.request = request
        self.relayed = []

    def recv_multipart(self):
        return self.request

    def send_multipart(self, parts):
        self.relayed.append(parts)

    def close(self):
        pass


class _FakeBackend:
    """PULL-socket double: one queued status reply."""

    def __init__(self, reply=(b"id", b"status")):
        self.reply = reply

    def recv_multipart(self):
        return self.reply

    def close(self):
        pass


class _FakePoller:
    """Poller double serving queued poll results then ContextTerminated."""

    def __init__(self):
        self.registered = []
        self.polls = []

    def register(self, sock, flag):
        self.registered.append((sock, flag))

    def poll(self):
        if self.polls:
            return self.polls.pop(0)
        raise _zmq().ContextTerminated()


def _zmq():
    import zmq

    return zmq


def test_worker_sender_thread_loop(conn_env, monkeypatch):
    """One poll cycle: the frontend request is submitted to the executor
    with its identity and the backend status is relayed back; the loop exits
    cleanly on ContextTerminated and closes both sockets."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        frontend = _FakeFrontend()
        backend = _FakeBackend()
        poller = _FakePoller()
        poller.polls = [{frontend: 1, backend: 1}]

        def fake_path(*args):
            if len(args) == 2:
                return f"inproc://{args[1]}"
            return f"{args[0]}://{args[1]}:{args[2]}"

        monkeypatch.setattr(fc, "make_zmq_path", fake_path)
        router_kind = _zmq().ROUTER

        def fake_socket(ctx, path, kind, **kwargs):
            return frontend if kind == router_kind else backend

        monkeypatch.setattr(fc, "make_zmq_socket", fake_socket)
        monkeypatch.setattr(_zmq(), "Poller", lambda: poller)
        submitted = []
        worker._sender_executor = SimpleNamespace(
            submit=lambda *args: submitted.append(args),
            shutdown=Mock(),
        )

        ready = threading.Event()
        worker._sender_thread(ready, 9000, 3)
        assert ready.is_set()
        assert submitted[0][1] == b"id"
        assert submitted[0][2] == b"meta-bytes"
        assert submitted[0][3].startswith("inproc://")
        assert frontend.relayed == [(b"id", b"", b"status")]
        assert len(poller.registered) == 2
    finally:
        worker.shutdown()


def test_worker_sender_thread_error_logged_and_sockets_closed(conn_env, monkeypatch):
    """An unexpected poll error is logged and both sockets still close."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        frontend = _FakeFrontend()
        backend = _FakeBackend()
        closed = []
        frontend.close = lambda: closed.append("frontend")
        backend.close = lambda: closed.append("backend")

        class _BrokenPoller(_FakePoller):
            def poll(self):
                raise RuntimeError("poller exploded")

        monkeypatch.setattr(
            fc,
            "make_zmq_path",
            lambda *args: "inproc://x" if len(args) == 2 else "tcp://h:1",
        )
        router_kind = _zmq().ROUTER

        def fake_socket(ctx, path, kind, **kwargs):
            return frontend if kind == router_kind else backend

        monkeypatch.setattr(fc, "make_zmq_socket", fake_socket)
        monkeypatch.setattr(_zmq(), "Poller", lambda: _BrokenPoller())
        ready = threading.Event()
        worker._sender_thread(ready, 9000, 0)  # must not raise
        assert ready.is_set()
        assert closed == ["frontend", "backend"]
    finally:
        worker.shutdown()


def test_worker_sender_worker_done_and_error_statuses(conn_env, monkeypatch):
    """The sender worker replies TRANS_DONE after a successful send and
    TRANS_ERROR when either decode or the transfer fails; the status pusher
    is closed in every case."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()
        sent = []

        class _FakePusher:
            def send_multipart(self, parts):
                sent.append(parts)

            def close(self):
                sent.append("closed")

        monkeypatch.setattr(
            fc, "make_zmq_socket", lambda *args, **kwargs: _FakePusher()
        )
        meta = _agent_meta(fc, req_blocks={})

        class _FakeDecoder:
            def decode(self, raw):
                return meta

        worker._decoder = _FakeDecoder()
        calls = []
        worker._send_kv_to_decode = lambda m: calls.append(m)
        worker._sender_worker(b"id", b"raw", "inproc://chan")
        assert sent[0] == (b"id", fc.TRANS_DONE)
        assert calls == [meta]

        def boom(m):
            raise RuntimeError("xfer failed")

        worker._send_kv_to_decode = boom
        worker._sender_worker(b"id", b"raw", "inproc://chan")
        assert sent[2] == (b"id", fc.TRANS_ERROR)

        worker._decoder = SimpleNamespace(
            decode=lambda raw: (_ for _ in ()).throw(ValueError("bad msgpack"))
        )
        worker._sender_worker(b"id", b"raw", "inproc://chan")
        assert sent[4] == (b"id", fc.TRANS_ERROR)
        assert sent[-1] == "closed"
    finally:
        worker.shutdown()


def test_worker_sender_worker_push_failure_logged(conn_env, monkeypatch, caplog):
    """A failing status push (server shutting down) is logged, not raised."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_producer")
    try:
        worker.kv_topo = _PlainTopo()

        class _BrokenPusher:
            def send_multipart(self, parts):
                raise _zmq().ZMQError("shutting down")

            def close(self):
                pass

        monkeypatch.setattr(
            fc, "make_zmq_socket", lambda *args, **kwargs: _BrokenPusher()
        )
        worker._decoder = SimpleNamespace(
            decode=lambda raw: (_ for _ in ()).throw(ValueError("nope"))
        )
        with caplog.at_level(
            logging.WARNING, logger="vllm_fl.distributed.kv_transfer.flagcx_connector"
        ):
            worker._sender_worker(b"id", b"raw", "inproc://chan")
        assert any("shutting down" in message for message in caplog.messages)
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# Worker: receive path and lifecycle lines
# ---------------------------------------------------------------------------


def test_worker_receive_kv_context_terminated_is_quiet(conn_env, monkeypatch):
    """A ContextTerminated during recv is debug-level quiet: no failed-recv
    stat is recorded (the finally block still finishes the requests)."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_consumer")
    try:
        worker.kv_topo = _PlainTopo()

        class _TermSock:
            async def send(self, data):
                pass

            async def recv(self):
                raise _zmq().ContextTerminated()

            def setsockopt(self, *args):
                pass

            def close(self):
                pass

        monkeypatch.setattr(fc, "make_zmq_socket", lambda *args, **kwargs: _TermSock())
        worker._encoder = SimpleNamespace(encode=lambda m: b"meta")
        asyncio.run(worker._receive_kv("tcp://x:1", {"d1": ("t1", [[1]])}))
        assert worker.xfer_stats.data["num_failed_recvs"] == []
        # The finally block still publishes the (single-shot) request ids.
        assert worker.finished_recving_reqs.set == {"d1"}
    finally:
        worker.shutdown()


def test_worker_receive_kv_generic_exception_records_failure(conn_env, monkeypatch):
    """Any other recv error records a failed recv and returns without
    finishing the request."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_consumer")
    try:
        worker.kv_topo = _PlainTopo()

        class _ExcSock:
            async def send(self, data):
                pass

            async def recv(self):
                raise RuntimeError("zmq blew up")

            def setsockopt(self, *args):
                pass

            def close(self):
                pass

        monkeypatch.setattr(fc, "make_zmq_socket", lambda *args, **kwargs: _ExcSock())
        worker._encoder = SimpleNamespace(encode=lambda m: b"meta")
        asyncio.run(worker._receive_kv("tcp://x:2", {"d2": ("t2", [[1]])}))
        assert worker.xfer_stats.data["num_failed_recvs"] == [1]
        assert not worker.finished_recving_reqs.set
    finally:
        worker.shutdown()


def test_worker_start_load_kv_consumer_schedules_pull(conn_env, monkeypatch):
    """The consumer role schedules the group pull coroutine on its receiver
    loop when there are requests to receive."""
    fc = conn_env.fc
    worker = conn_env.make_worker(role="kv_consumer")
    try:
        worker.kv_topo = _PlainTopo()
        launched = []

        def fake_rcs(coro, loop):
            launched.append(coro)
            coro.close()  # the fake loop never runs it; silence the warning

            class _Future:
                def result(self):
                    return set()

            return _Future()

        monkeypatch.setattr(fc.asyncio, "run_coroutine_threadsafe", fake_rcs)
        metadata = fc.FlagCXConnectorMetadata()
        metadata.add_new_req(
            request_id="d1",
            local_block_ids=[[1]],
            kv_transfer_params={
                "transfer_id": "t9",
                "remote_host": "h",
                "remote_port": 1,
            },
        )
        worker.start_load_kv(metadata)
        assert len(launched) == 1
    finally:
        worker.shutdown()


def test_worker_shutdown_joins_sender_thread(conn_env):
    """Shutdown joins the registered sender thread with the documented
    two-second timeout."""
    worker = conn_env.make_worker(role="kv_producer")
    joined = []

    class _FakeThread:
        def join(self, timeout=None):
            joined.append(timeout)

    worker._sender_t = _FakeThread()
    worker.shutdown()
    assert joined == [2]


# ---------------------------------------------------------------------------
# Fixture: load the real device_communicators/flagcx.py under stubs
# ---------------------------------------------------------------------------


class _FakeUniqueId(ctypes.Structure):
    """ctypes-backed unique id double (legacy API needs ``ctypes.pointer``)."""

    _fields_ = [("internal", ctypes.c_ubyte * 8)]


class _StubGroup:
    """StatelessProcessGroup double for rank 0 of a 2-rank world."""

    rank = 0
    world_size = 2

    def broadcast_obj(self, value, src):
        return value


def _install_wrapper_stub(monkeypatch, library_factory=None):
    """Install the six-name plugin.interservice.flagcx_wrapper boundary."""
    wrapper = SimpleNamespace(
        FLAGCXLibrary=library_factory or (lambda path: Mock(name="flagcx-lib")),
        # Args-sensitive so root/non-root broadcast buffers are distinguishable.
        buffer_type=lambda *args: ("buffer", args),
        flagcxComm_t=int,
        flagcxDataTypeEnum=SimpleNamespace(from_torch=lambda dtype: "dtype"),
        flagcxUniqueId=_FakeUniqueId,
        flagcxRedOpTypeEnum=SimpleNamespace(from_torch=lambda op: "redop"),
    )
    monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", wrapper)
    return wrapper


@pytest.fixture
def load_communicator(monkeypatch):
    """Yield a loader executing the real flagcx.py source under stubs.

    Follows the core-suite pattern (tests/unit_tests/distributed/test_flagcx.py):
    the module runs under a unique subject name so each load re-executes the
    import-time vendor dispatch, and the ``_wheel_api``/``_amd_api`` child
    entries are cleaned up afterwards.
    """
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.utils",
        SimpleNamespace(StatelessProcessGroup=_StubGroup),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.logger",
        SimpleNamespace(init_logger=lambda name: logging.getLogger(name)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.torch_utils",
        SimpleNamespace(current_stream=Mock()),
    )
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    _install_wrapper_stub(monkeypatch)
    wheel_api = f"{_COMMUNICATOR_SUBJECT}._wheel_api"
    amd_api = f"{_COMMUNICATOR_SUBJECT}._amd_api"
    monkeypatch.delitem(sys.modules, wheel_api, raising=False)
    monkeypatch.delitem(sys.modules, amd_api, raising=False)

    def load():
        spec = importlib.util.spec_from_file_location(
            _COMMUNICATOR_SUBJECT, _FLAGCX_SOURCE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield load
    sys.modules.pop(wheel_api, None)
    sys.modules.pop(amd_api, None)


def _wheel_layout(tmp_path, api_source=_API_SOURCE):
    """Create a wheel-style FlagCX root: api.py + lib/libflagcx.so."""
    root = tmp_path / "flagcx-wheel"
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "libflagcx.so").touch()
    (root / "api.py").write_text(api_source, encoding="utf-8")
    return root


class _FakeDevice:
    """Duck-typed torch.device for exotic vendor types (npu/ptpu/txda).

    ``torch.device`` cannot be subclassed and ``rename_privateuse1_backend``
    is one-shot per process (one exotic name), so vendor-type branches are
    exercised through this double instead: it is installed as ``torch.device``
    on a per-module torch proxy, making both the string conversion and the
    ``isinstance(device, torch.device)`` assertion in the constructor pass.
    Instances are context managers (the ptpu branch re-wraps the device
    through ``torch.device(self.device)`` and enters the result).
    """

    def __new__(cls, value):
        if isinstance(value, _FakeDevice):
            return value  # torch.device(self.device) re-wrap: identity.
        return super().__new__(cls)

    def __init__(self, value):
        if isinstance(value, _FakeDevice):
            return
        if isinstance(value, torch.device):
            type_, index = value.type, value.index
        else:
            name = str(value)
            type_, _, idx = name.partition(":")
            index = int(idx) if idx else None
        self._type = type_
        self.index = index if index is not None else 0
        self.entered = []

    @property
    def type(self):
        return self._type

    def __eq__(self, other):
        return (
            isinstance(other, _FakeDevice)
            and self._type == other._type
            and self.index == other.index
        )

    def __hash__(self):
        return hash((self._type, self.index))

    def __repr__(self):
        return f"{self._type}:{self.index}"

    def __enter__(self):
        self.entered.append(True)
        return self

    def __exit__(self, *args):
        return False


class _TorchProxy:
    """Proxy over the real torch letting tests swap per-device namespaces.

    ``device`` and ``zeros`` are overridden per test; every other attribute
    (ByteTensor, empty_like, ...) delegates to the real torch.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, item):
        return getattr(self._real, item)


# ---------------------------------------------------------------------------
# flagcx.py: API loaders
# ---------------------------------------------------------------------------


def test_load_flagcx_wheel_bindings_returns_six_names(load_communicator, tmp_path):
    """The wheel loader returns the six API objects from the api.py file."""
    module = load_communicator()
    api_path = tmp_path / "wheel-api.py"
    api_path.write_text(_API_SOURCE, encoding="utf-8")
    results = module._load_flagcx_wheel_bindings(api_path)
    assert len(results) == 6
    # The class comes from the freshly executed api module, not the subject.
    assert type(results[4]).__module__ != _COMMUNICATOR_SUBJECT


def test_load_flagcx_wheel_bindings_incomplete_api(load_communicator, tmp_path):
    """An api.py missing one of the six names raises ImportError and leaves
    no entry behind in sys.modules."""
    module = load_communicator()
    api_path = tmp_path / "incomplete.py"
    api_path.write_text("class FLAGCXLibrary: pass\n", encoding="utf-8")
    with pytest.raises(ImportError, match="Incomplete FlagCX API"):
        module._load_flagcx_wheel_bindings(api_path)
    assert f"{_COMMUNICATOR_SUBJECT}._wheel_api" not in sys.modules


def test_load_flagcx_wheel_bindings_failure_restores_previous(
    load_communicator, tmp_path
):
    """A raising api.py propagates the original error; a pre-existing
    sys.modules entry is restored, an absent one stays absent."""
    module = load_communicator()
    api_name = f"{_COMMUNICATOR_SUBJECT}._wheel_api"

    raising = tmp_path / "raising.py"
    raising.write_text("import sys\nraise ValueError('boom')\n", encoding="utf-8")
    sys.modules[api_name] = "previous"
    try:
        with pytest.raises(ValueError, match="boom"):
            module._load_flagcx_wheel_bindings(raising)
        assert sys.modules[api_name] == "previous"
    finally:
        del sys.modules[api_name]

    with pytest.raises(ValueError, match="boom"):
        module._load_flagcx_wheel_bindings(raising)
    assert api_name not in sys.modules


def test_load_amd_flagcx_api_executes_and_cleans_up(load_communicator, tmp_path):
    """The AMD loader executes the api module under its private name and
    removes it from sys.modules when execution fails."""
    module = load_communicator()
    ok = tmp_path / "amd-ok.py"
    ok.write_text("X = 41\n", encoding="utf-8")
    api = module._load_amd_flagcx_api(ok)
    assert api.X == 41
    assert api.__name__ == f"{_COMMUNICATOR_SUBJECT}._amd_api"
    sys.modules.pop(f"{_COMMUNICATOR_SUBJECT}._amd_api", None)

    bad = tmp_path / "amd-bad.py"
    bad.write_text("raise ValueError('amd boom')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="amd boom"):
        module._load_amd_flagcx_api(bad)
    assert f"{_COMMUNICATOR_SUBJECT}._amd_api" not in sys.modules


def test_load_kunlunxin_flagcx_api_prefers_wrapper(load_communicator, tmp_path):
    """A directory containing plugin/interservice/flagcx_wrapper.py is used
    directly; an import failure falls back to the flagcx package."""
    module = load_communicator()
    root = tmp_path / "kx"
    (root / "plugin" / "interservice").mkdir(parents=True)
    (root / "plugin" / "__init__.py").write_text("", encoding="utf-8")
    (root / "plugin" / "interservice" / "__init__.py").write_text("", encoding="utf-8")
    (root / "plugin" / "interservice" / "flagcx_wrapper.py").write_text(
        'MARKER = "kunlunxin-wrapper"\n', encoding="utf-8"
    )
    saved = {
        name: sys.modules.pop(name)
        for name in (
            "plugin",
            "plugin.interservice",
            "plugin.interservice.flagcx_wrapper",
        )
        if name in sys.modules
    }
    sys.path.insert(0, str(root))
    try:
        api = module._load_kunlunxin_flagcx_api(str(root))
        assert getattr(api, "MARKER", None) == "kunlunxin-wrapper"
    finally:
        sys.path.remove(str(root))
        sys.modules.update(saved)


def test_load_kunlunxin_flagcx_api_falls_back_to_package(
    load_communicator, monkeypatch
):
    """Without a wrapper module the loader falls back to ``flagcx.api``."""
    import types as types_mod

    module = load_communicator()
    api_mod = types_mod.ModuleType("flagcx.api")
    api_mod.MARKER = "flagcx-package"
    pkg = types_mod.ModuleType("flagcx")
    pkg.api = api_mod
    monkeypatch.setitem(sys.modules, "flagcx", pkg)
    monkeypatch.setitem(sys.modules, "flagcx.api", api_mod)
    api = module._load_kunlunxin_flagcx_api("")
    assert api.MARKER == "flagcx-package"


# ---------------------------------------------------------------------------
# flagcx.py: vendor detection and import-time dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_vendor", "platform_vendor", "expected"),
    [
        ("kunlunxin", None, True),
        ("nvidia", None, False),  # env set but different vendor: short-circuits
        (None, "kunlunxin", True),
        (None, "nvidia", False),
    ],
)
def test_is_kunlunxin_detection(
    load_communicator, monkeypatch, env_vendor, platform_vendor, expected
):
    """The env var wins when set; otherwise the platform vendor decides."""
    module = load_communicator()
    if env_vendor is not None:
        monkeypatch.setenv("GEMS_VENDOR", env_vendor)
    else:
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace(vendor_name=platform_vendor)),
    )
    assert module._is_kunlunxin() is expected


def test_is_kunlunxin_platform_import_failure_is_false(load_communicator, monkeypatch):
    """A failing vllm.platforms import means 'not kunlunxin'."""
    module = load_communicator()
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)
    assert module._is_kunlunxin() is False


def test_kunlunxin_package_dispatch_sets_packaged_library(
    load_communicator, monkeypatch
):
    """GEMS_VENDOR=kunlunxin with an importable flagcx package binds all six
    names and marks the packaged-library flag."""
    import types as types_mod

    api_mod = types_mod.ModuleType("flagcx.api")

    class _KxLib:
        pass

    api_mod.FLAGCXLibrary = _KxLib
    api_mod.buffer_type = object()
    api_mod.flagcxComm_t = object()
    api_mod.flagcxDataTypeEnum = object()
    api_mod.flagcxUniqueId = object()
    api_mod.flagcxRedOpTypeEnum = object()
    pkg = types_mod.ModuleType("flagcx")
    pkg.api = api_mod
    monkeypatch.setitem(sys.modules, "flagcx", pkg)
    monkeypatch.setitem(sys.modules, "flagcx.api", api_mod)
    monkeypatch.setenv("GEMS_VENDOR", "kunlunxin")
    module = load_communicator()
    assert module._flagcx_available is True
    assert module._flagcx_kunlunxin is True
    assert module._flagcx_packaged_library is True
    assert module.FLAGCXLibrary is _KxLib


def test_amd_source_dispatch_loads_pinned_api(load_communicator, monkeypatch, tmp_path):
    """FLAGCX_PATH + torch.version.hip + an AMD platform vendor loads the
    pinned source API from <root>/src/flagcx/api.py."""
    root = tmp_path / "amd-root"
    (root / "src" / "flagcx").mkdir(parents=True)
    (root / "src" / "flagcx" / "api.py").write_text(_API_SOURCE, encoding="utf-8")
    monkeypatch.setenv("FLAGCX_PATH", str(root))
    monkeypatch.setattr(torch.version, "hip", "5.7.0")
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace(vendor_name="amd")),
    )
    module = load_communicator()
    assert module._flagcx_amd_source is True
    assert module._flagcx_available is True
    assert module._flagcx_packaged_library is False
    sys.modules.pop(f"{_COMMUNICATOR_SUBJECT}._amd_api", None)


def test_plain_path_non_import_error_reraises(load_communicator, monkeypatch, tmp_path):
    """An OSError on the plain wrapper path propagates instead of disabling.

    The raising wrapper lives in a real package tree prepended to sys.path,
    so its import is found deterministically on every environment."""
    tmp = tmp_path / "broken"
    (tmp / "plugin" / "interservice").mkdir(parents=True)
    (tmp / "plugin" / "__init__.py").write_text("", encoding="utf-8")
    (tmp / "plugin" / "interservice" / "__init__.py").write_text("", encoding="utf-8")
    (tmp / "plugin" / "interservice" / "flagcx_wrapper.py").write_text(
        "raise OSError('no such device')\n", encoding="utf-8"
    )
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path / "elsewhere"))
    for name in ("plugin", "plugin.interservice", "plugin.interservice.flagcx_wrapper"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp))
    with pytest.raises(OSError, match="no such device"):
        load_communicator()


def test_is_trace_tensor_variants(load_communicator):
    """None and non-tensors are False; meta and FakeTensors are True; a real
    CPU tensor is False."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    module = load_communicator()
    assert module.is_trace_tensor(None) is False
    assert module.is_trace_tensor("not-a-tensor") is False
    assert module.is_trace_tensor(torch.zeros(3)) is False
    assert module.is_trace_tensor(torch.zeros(3, device="meta")) is True
    with FakeTensorMode():
        fake = torch.zeros(3)
    assert module.is_trace_tensor(fake) is True


# ---------------------------------------------------------------------------
# flagcx.py: PyFlagcxCommunicator.__init__
# ---------------------------------------------------------------------------


def test_init_world_size_one_disables(load_communicator):
    """A single-rank group returns early with the communicator disabled."""
    module = load_communicator()

    class _SingleRank(_StubGroup):
        rank = 0
        world_size = 1

    comm = module.PyFlagcxCommunicator(_SingleRank(), "cpu")
    assert comm.available is False
    assert comm.disabled is True
    assert comm.comm is None


def test_init_unavailable_library_raises_recorded(load_communicator, monkeypatch):
    """A None FLAGCXLibrary raises ImportError inside the guarded block and
    leaves the communicator disabled with init_error set."""
    module = load_communicator()
    monkeypatch.setattr(module, "FLAGCXLibrary", None)
    comm = module.PyFlagcxCommunicator(_StubGroup(), "cpu")
    assert comm.disabled is True
    assert comm.available is False
    assert isinstance(comm.init_error, ImportError)
    assert "unavailable" in str(comm.init_error)


def test_init_library_ctor_failure_disables(load_communicator, monkeypatch):
    """An OSError from the native library constructor disables the
    communicator (non-GPU environments)."""
    module = load_communicator()

    def failing(path):
        raise OSError("stub native library")

    monkeypatch.setattr(module, "FLAGCXLibrary", failing)
    comm = module.PyFlagcxCommunicator(
        _StubGroup(), "cpu", library_path="stub/libflagcx.so"
    )
    assert comm.disabled is True
    assert isinstance(comm.init_error, OSError)


def test_init_wheel_library_path_preferred(load_communicator, monkeypatch, tmp_path):
    """With a wheel-loaded module the constructor passes the recorded .so
    path; an explicit library_path wins over both."""
    root = _wheel_layout(tmp_path)
    monkeypatch.setenv("FLAGCX_PATH", str(root))
    module = load_communicator()
    captured = []

    def fake_library(path):
        captured.append(path)
        raise OSError("no native library in CPU tests")

    monkeypatch.setattr(module, "FLAGCXLibrary", fake_library)
    module.PyFlagcxCommunicator(_StubGroup(), "cpu")
    assert captured == [str(root / "lib" / "libflagcx.so")]

    captured.clear()
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path / "elsewhere"))
    module.PyFlagcxCommunicator(
        _StubGroup(), "cpu", library_path="explicit/libflagcx.so"
    )
    assert captured == ["explicit/libflagcx.so"]


def test_init_success_cpu_device_and_warmup(load_communicator, monkeypatch):
    """The full happy path on a CPU device: unique id broadcast via
    broadcast_obj, new-API comm init, warmup all_reduce and synchronize."""
    module = load_communicator()
    lib = Mock()
    lib.flagcxGetUniqueId.return_value = _FakeUniqueId()
    lib.flagcxCommInitRank.return_value = "comm-handle"
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: lib)
    monkeypatch.setattr(module, "_flagcx_wheel_library", None)
    monkeypatch.setenv("FLAGCX_PATH", "")

    synchronize_calls = []

    class _SyncStream:
        def synchronize(self):
            synchronize_calls.append(True)

    monkeypatch.setattr(module, "current_stream", Mock(return_value=_SyncStream()))
    with _patched_cuda_device(monkeypatch):
        comm = module.PyFlagcxCommunicator(_StubGroup(), "cpu")
    assert comm.available is True
    assert comm.disabled is False
    assert comm.comm == "comm-handle"
    assert comm.device == torch.device("cpu")
    # New (non-legacy) API: unique id passed directly, no ctypes.pointer.
    init_args = lib.flagcxCommInitRank.call_args.args
    assert init_args[0] == 2 and init_args[2] == 0
    assert not isinstance(init_args[1], int)
    # Warmup ran one all_reduce and synchronized the stream.
    assert lib.flagcxAllReduce.called
    assert synchronize_calls == [True]


@contextlib.contextmanager
def _patched_cuda_device(monkeypatch):
    """Make torch.cuda.device a nullcontext factory (CPU builds raise on enter)."""
    sentinel = object()

    def fake_device(device):
        assert device is not sentinel
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.cuda, "device", fake_device, raising=False)
    yield


def test_init_legacy_unique_id_and_stateful_group(load_communicator, monkeypatch):
    """A legacy-API library (handler attribute, no devHandle) reads .contents
    from flagcxGetUniqueId and inits the comm through ctypes.pointer; a
    non-StatelessProcessGroup group broadcasts the id through torch.distributed."""
    module = load_communicator()
    # spec= pins the attribute set: `handler` present, `devHandle` absent,
    # which is exactly the legacy unique-id API detection.
    lib = Mock(
        spec=[
            "handler",
            "flagcxGetUniqueId",
            "flagcxCommInitRank",
            "flagcxAllReduce",
            "adaptor_stream_copy",
            "adaptor_stream_free",
        ]
    )
    unique = _FakeUniqueId()
    lib.flagcxGetUniqueId.return_value = SimpleNamespace(contents=unique)
    lib.flagcxCommInitRank.return_value = "comm-legacy"

    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: lib)
    monkeypatch.setattr(module, "_flagcx_wheel_library", None)
    monkeypatch.setenv("FLAGCX_PATH", "")

    import torch.distributed as dist

    broadcast_tensors = []

    def fake_broadcast(tensor, src, group):
        broadcast_tensors.append((tensor, src))

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo", raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda group: 1, raising=False)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2, raising=False)
    monkeypatch.setattr(
        dist, "get_process_group_ranks", lambda group: [0, 1], raising=False
    )
    monkeypatch.setattr(dist, "broadcast", fake_broadcast, raising=False)

    synchronize = Mock()
    monkeypatch.setattr(module, "current_stream", Mock(return_value=synchronize))
    with _patched_cuda_device(monkeypatch):
        comm = module.PyFlagcxCommunicator(object(), "cpu")
    assert comm.rank == 1 and comm.world_size == 2
    assert comm.comm == "comm-legacy"
    # The unique id bytes travelled through the ByteTensor broadcast.
    assert len(broadcast_tensors) == 1
    assert broadcast_tensors[0][1] == 0  # src is the group's global rank 0
    # Legacy init: ctypes.pointer wraps the unique id.
    init_args = lib.flagcxCommInitRank.call_args.args
    assert isinstance(init_args[1], ctypes._Pointer)
    assert init_args[0] == 2 and init_args[2] == 1


def test_init_npu_device_with_wheel_uses_ascend_stream(load_communicator, monkeypatch):
    """An NPU device with a wheel library imports the Ascend all-reduce
    stream patch, uses the NPU device context and NPU current stream."""
    import types as types_mod

    fake_stream = SimpleNamespace(name="FakeAllReduceStream")
    ascend_patch = types_mod.ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.patches.flagcx"
    )
    ascend_patch.FlagCXAllReduceStream = lambda device: fake_stream
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.flagcx",
        ascend_patch,
    )

    module = load_communicator()
    lib = Mock()
    lib.flagcxGetUniqueId.return_value = _FakeUniqueId()
    lib.flagcxCommInitRank.return_value = "comm-npu"
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: lib)
    monkeypatch.setattr(module, "_flagcx_wheel_library", "/wheel/libflagcx.so")

    npu_stream = SimpleNamespace(
        npu_stream=5555,
        synchronize=Mock(),
    )
    fake_npu = SimpleNamespace(
        device=lambda device: device,  # _FakeDevice is its own context manager
        current_stream=lambda device: npu_stream,
        stream=lambda stream: contextlib.nullcontext(),
    )
    proxy = _TorchProxy(torch)
    object.__setattr__(proxy, "npu", fake_npu)
    object.__setattr__(proxy, "device", _FakeDevice)
    object.__setattr__(
        proxy, "zeros", lambda *args, **kwargs: torch.zeros(1, device="meta")
    )
    monkeypatch.setattr(module, "torch", proxy, raising=False)
    comm = module.PyFlagcxCommunicator(_StubGroup(), "npu:0")
    assert comm.device.entered == [True]  # npu device context entered
    assert comm._all_reduce_stream is fake_stream
    assert comm._current_stream() is npu_stream
    # Meta warmup tensor short-circuits the real collective (trace tensor).
    assert lib.flagcxAllReduce.called is False


def test_init_ptpu_device_uses_plain_device_ctx(load_communicator, monkeypatch):
    """The ptpu branch re-wraps through torch.device and enters the device
    itself as its context manager."""
    module = load_communicator()
    lib = Mock()
    lib.flagcxGetUniqueId.return_value = _FakeUniqueId()
    lib.flagcxCommInitRank.return_value = "comm-ptpu"
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: lib)
    monkeypatch.setattr(module, "_flagcx_wheel_library", None)
    monkeypatch.setenv("FLAGCX_PATH", "")

    synchronize = Mock()
    monkeypatch.setattr(module, "current_stream", Mock(return_value=synchronize))
    proxy = _TorchProxy(torch)
    object.__setattr__(proxy, "device", _FakeDevice)
    object.__setattr__(
        proxy, "zeros", lambda *args, **kwargs: torch.zeros(1, device="meta")
    )
    monkeypatch.setattr(module, "torch", proxy, raising=False)
    comm = module.PyFlagcxCommunicator(_StubGroup(), "ptpu:0")
    assert comm.device.entered  # entered once when the ptpu ctx wrapped it
    assert comm.comm == "comm-ptpu"


def test_init_txda_warmup_failure_destroys_and_raises(load_communicator, monkeypatch):
    """A failing warmup on TXDA destroys the communicator (abort path) and
    re-raises the original error."""
    module = load_communicator()
    lib = Mock()
    lib.flagcxGetUniqueId.return_value = _FakeUniqueId()
    lib.flagcxCommInitRank.return_value = "comm-txda"
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: lib)
    monkeypatch.setattr(module, "_flagcx_wheel_library", None)
    monkeypatch.setenv("FLAGCX_PATH", "")

    abort = Mock(return_value=0)
    lib.lib = {"flagcxCommAbort": abort}
    lib.FLAGCX_CHECK = Mock()

    fake_txda = SimpleNamespace(
        device=lambda device: device,  # _FakeDevice is its own ctx manager
        current_stream=lambda device: SimpleNamespace(synchronize=Mock()),
    )
    proxy = _TorchProxy(torch)
    object.__setattr__(proxy, "txda", fake_txda)
    object.__setattr__(proxy, "device", _FakeDevice)
    object.__setattr__(
        proxy, "zeros", lambda *args, **kwargs: torch.zeros(1, device="meta")
    )
    monkeypatch.setattr(module, "torch", proxy, raising=False)
    monkeypatch.setattr(module, "_TXDA_ABORT_TIMEOUT_SECONDS", 0.2)

    def failing_all_reduce(self, tensor, *args, **kwargs):
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(
        module.PyFlagcxCommunicator, "all_reduce", failing_all_reduce, raising=False
    )
    try:
        with pytest.raises(RuntimeError, match="warmup failed"):
            module.PyFlagcxCommunicator(_StubGroup(), "txda:0")
        assert abort.called  # destroy ran the abort path
    finally:
        del module.PyFlagcxCommunicator.all_reduce


# ---------------------------------------------------------------------------
# flagcx.py: destroy()
# ---------------------------------------------------------------------------


def test_destroy_txda_aborts_through_native_thread(load_communicator):
    """TXDA destroy runs flagcxCommAbort in a daemon thread, checks the
    return code, clears the comm handle and is idempotent."""
    module = load_communicator()
    abort = Mock(return_value=0)
    lib = Mock()
    lib.lib = {"flagcxCommAbort": abort}
    lib.FLAGCX_CHECK = Mock()

    comm = module.PyFlagcxCommunicator.__new__(module.PyFlagcxCommunicator)
    comm.device = _FakeDevice("txda:0")
    comm._destroy_lock = threading.Lock()
    comm._destroy_started = False
    comm._destroy_thread = None
    comm.flagcx = lib
    comm.comm = "comm-handle"
    comm.available = True
    comm.disabled = False

    fake_txda = SimpleNamespace(device=lambda device: device)
    proxy = _TorchProxy(torch)
    object.__setattr__(proxy, "txda", fake_txda)
    object.__setattr__(proxy, "device", _FakeDevice)
    module.torch = proxy
    module._TXDA_ABORT_TIMEOUT_SECONDS = 0.2

    comm.destroy()
    assert abort.called
    assert lib.FLAGCX_CHECK.called
    assert comm.available is False
    assert comm.disabled is True
    assert comm.comm is None
    assert comm.device.entered  # abort ran inside the device context
    # Second destroy: already started, no second abort.
    comm.destroy()
    assert abort.call_count == 1


# ---------------------------------------------------------------------------
# flagcx.py: collectives via __new__-constructed communicators
# ---------------------------------------------------------------------------


def _cpu_communicator(module, device_type="cpu", rank=0):
    """Build a communicator by __new__ with a mocked native library.

    Exotic device types get a ``_FakeDevice`` plus a module torch proxy so
    the npu/txda branches of ``_current_stream``/``_copy_flagcx_stream``
    resolve without registering a privateuse1 backend.
    """
    comm = module.PyFlagcxCommunicator.__new__(module.PyFlagcxCommunicator)
    comm.disabled = False
    if device_type == "cpu":
        comm.device = torch.device("cpu")
    else:
        comm.device = _FakeDevice(f"{device_type}:0")
        fake_vendor = SimpleNamespace(
            device=lambda device: device,
            current_stream=lambda device: SimpleNamespace(synchronize=Mock()),
            stream=lambda stream: contextlib.nullcontext(),
        )
        proxy = _TorchProxy(torch)
        object.__setattr__(proxy, device_type, fake_vendor)
        object.__setattr__(proxy, "device", _FakeDevice)
        module.torch = proxy
    comm.flagcx = Mock()
    comm.comm = "comm-handle"
    comm.rank = rank
    comm._all_reduce_stream = None
    return comm


def test_all_reduce_disabled_returns_none(load_communicator):
    """A disabled communicator returns None without touching the library."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    comm.disabled = True
    assert comm.all_reduce(torch.zeros(2)) is None
    assert not comm.flagcx.flagcxAllReduce.called


def test_all_reduce_trace_tensor_short_circuits(load_communicator):
    """Fake/meta inputs never reach the native library: an empty_like is
    returned instead."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    result = comm.all_reduce(torch.zeros(2, device="meta"))
    assert isinstance(result, torch.Tensor)
    assert result.is_meta
    assert not comm.flagcx.adaptor_stream_copy.called


def test_all_reduce_creates_output_and_uses_current_stream(load_communicator):
    """Without an out tensor one is allocated; without a stream the current
    stream helper is consulted."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    stream = Mock()
    monkey_stream = Mock(return_value=stream)
    module.current_stream = monkey_stream
    out = comm.all_reduce(torch.ones(4), stream=None)
    assert isinstance(out, torch.Tensor) and out.numel() == 4
    assert monkey_stream.called
    assert comm.flagcx.flagcxAllReduce.called
    # The stream copy was freed after the collective.
    assert comm.flagcx.adaptor_stream_free.called


def test_all_reduce_on_stream_frees_stream_on_error(load_communicator):
    """_all_reduce_on_stream frees the copied stream even when the native
    call raises (finally clause)."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    comm.flagcx.flagcxAllReduce.side_effect = RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        comm._all_reduce_on_stream(torch.ones(2), torch.zeros(2), "sum", "stream")
    assert comm.flagcx.adaptor_stream_free.called


def test_all_reduce_submits_through_all_reduce_stream(load_communicator):
    """With an _all_reduce_stream present the collective is submitted
    through it instead of running inline."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    submitted = []
    comm._all_reduce_stream = SimpleNamespace(
        submit=lambda *args: submitted.append(args)
    )
    comm.all_reduce(torch.ones(2), stream="caller")
    assert len(submitted) == 1
    assert submitted[0][0].__name__ == "_all_reduce_on_stream"
    assert not comm.flagcx.flagcxAllReduce.called


def test_all_gather_disabled_and_trace(load_communicator):
    """all_gather returns early when disabled or when tracing."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    comm.disabled = True
    comm.all_gather(torch.zeros(4), torch.zeros(2))
    assert not comm.flagcx.flagcxAllGather.called

    comm.disabled = False
    comm.all_gather(torch.zeros(4, device="meta"), torch.zeros(2, device="meta"))
    assert not comm.flagcx.flagcxAllGather.called


def test_all_gather_invokes_library(load_communicator):
    """all_gather passes input/output pointers, numel and the dtype enum to
    the library and frees the copied stream."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    inp, out = torch.zeros(2), torch.zeros(4)
    comm.all_gather(out, inp, stream="s")
    args = comm.flagcx.flagcxAllGather.call_args.args
    assert args[2] == 2  # numel
    assert comm.flagcx.adaptor_stream_free.called


def test_all_gatherv_broadcasts_each_root_slice(load_communicator):
    """all_gatherv wraps per-root broadcasts in group_start/end and slices
    the output at the running offset."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    out = torch.zeros(4)
    inp = torch.zeros(2)
    comm.all_gatherv(out, inp, [2, 2], stream="s")
    assert comm.flagcx.flagcxBroadcast.call_count == 2
    # Slices handed to the library advance by the running offset.
    first, second = comm.flagcx.flagcxBroadcast.call_args_list
    assert first.args[2] == 2 and second.args[2] == 2
    assert first.args[4] == 0 and second.args[4] == 1  # roots
    assert comm.flagcx.flagcxGroupStart.called
    assert comm.flagcx.flagcxGroupEnd.called
    assert comm.flagcx.adaptor_stream_free.called


def test_all_gatherv_shape_mismatch_rejected(load_communicator):
    """output rows must equal sum(sizes)."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    with pytest.raises(AssertionError):
        comm.all_gatherv(torch.zeros(3), torch.zeros(2), [2, 2], stream="s")


def test_reduce_scatter_invokes_library(load_communicator):
    """reduce_scatter forwards pointers, output numel and both enums."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    comm.reduce_scatter(torch.zeros(2), torch.zeros(4), stream="s")
    args = comm.flagcx.flagcxReduceScatter.call_args.args
    assert args[2] == 2
    assert comm.flagcx.adaptor_stream_free.called


def test_reduce_scatterv_reduces_per_root_chunks(load_communicator):
    """reduce_scatterv reduces each root's input chunk inside a group."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    comm.reduce_scatterv(torch.zeros(2), torch.zeros(4), [2, 2], stream="s")
    assert comm.flagcx.flagcxReduce.call_count == 2
    roots = [c.args[5] for c in comm.flagcx.flagcxReduce.call_args_list]
    assert roots == [0, 1]
    assert comm.flagcx.flagcxGroupStart.called and comm.flagcx.flagcxGroupEnd.called
    assert comm.flagcx.adaptor_stream_free.called


def test_send_recv_invoke_library(load_communicator):
    """send/recv forward the tensor pointer, numel, dtype enum and peer."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    comm.send(torch.zeros(3), 1, stream="s")
    send_args = comm.flagcx.flagcxSend.call_args.args
    assert send_args[1] == 3 and send_args[3] == 1
    comm.recv(torch.zeros(3), 2, stream="s")
    recv_args = comm.flagcx.flagcxRecv.call_args.args
    assert recv_args[1] == 3 and recv_args[3] == 2
    assert comm.flagcx.adaptor_stream_free.call_count == 2


@pytest.mark.parametrize("src_is_root", [True, False])
def test_broadcast_buffer_selection(load_communicator, src_is_root):
    """The root rank passes the tensor pointer as both sendbuff and
    recvbuff; non-roots send an empty buffer."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    src = comm.rank if src_is_root else comm.rank + 1
    comm.broadcast(torch.zeros(3), src, stream="s")
    sendbuff, recvbuff = comm.flagcx.flagcxBroadcast.call_args.args[:2]
    if src_is_root:
        assert sendbuff == recvbuff
    else:
        assert sendbuff != recvbuff
    assert comm.flagcx.adaptor_stream_free.called


@pytest.mark.parametrize("device_type", ["cpu", "npu"])
def test_group_start_end_argument_forms(load_communicator, device_type):
    """NPU (and AMD-source) pass the comm handle; other devices call the
    argument-less form."""
    module = load_communicator()
    comm = _cpu_communicator(module, device_type=device_type)
    comm.group_start()
    comm.group_end()
    if device_type == "npu":
        assert comm.flagcx.flagcxGroupStart.call_args.args == ("comm-handle",)
        assert comm.flagcx.flagcxGroupEnd.call_args.args == ("comm-handle",)
    else:
        assert comm.flagcx.flagcxGroupStart.call_args.args == ()
        assert comm.flagcx.flagcxGroupEnd.call_args.args == ()


def test_group_start_end_amd_source_form(load_communicator, monkeypatch):
    """An AMD-source module passes the comm even on non-NPU devices."""
    module = load_communicator()
    monkeypatch.setattr(module, "_flagcx_amd_source", True)
    comm = _cpu_communicator(module)
    comm.group_start()
    comm.group_end()
    assert comm.flagcx.flagcxGroupStart.call_args.args == ("comm-handle",)
    assert comm.flagcx.flagcxGroupEnd.call_args.args == ("comm-handle",)


def test_copy_flagcx_stream_npu_wraps_cuda_stream(load_communicator):
    """On NPU devices the stream is rewrapped as a cuda_stream namespace
    before reaching adaptor_stream_copy."""
    module = load_communicator()
    comm = _cpu_communicator(module, device_type="npu")
    npu_stream = SimpleNamespace(npu_stream=777)
    comm._copy_flagcx_stream(npu_stream)
    wrapped = comm.flagcx.adaptor_stream_copy.call_args.args[0]
    assert wrapped.cuda_stream == 777


def test_copy_flagcx_stream_passthrough_other_devices(load_communicator):
    """Non-NPU devices hand the stream to adaptor_stream_copy unchanged."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    stream = Mock()
    comm._copy_flagcx_stream(stream)
    assert comm.flagcx.adaptor_stream_copy.call_args.args[0] is stream


def _npu_labeled_tensor(module, data):
    """A CPU-backed tensor whose .device reports the communicator's NPU
    device, so the collectives' device asserts pass without any native NPU
    runtime. ``module.torch`` is already the test proxy from
    ``_cpu_communicator``; the NPU stream context is its ``npu.stream``.
    ``__torch_function__`` unwraps back to plain tensors so slicing and
    ``copy_`` (the NPU own-slice restore) work with the stock kernels."""

    class _FakeNpuTensor(torch.Tensor):
        @property
        def device(self):
            return _FakeDevice("npu:0")

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            unwrap = lambda a: a.as_subclass(torch.Tensor) if isinstance(a, cls) else a  # noqa: E731
            args = tuple(unwrap(a) for a in args)
            if kwargs:
                kwargs = {k: unwrap(v) for k, v in kwargs.items()}
            return func(*args, **(kwargs or {}))

    return _FakeNpuTensor(data)


def test_all_gatherv_npu_fills_own_slice(load_communicator):
    """The NPU broadcast copies only count bytes, so the rank's own slice is
    restored with a local copy under the npu stream context."""
    module = load_communicator()
    comm = _cpu_communicator(module, device_type="npu", rank=0)
    entered = []
    fake_npu = SimpleNamespace(
        stream=lambda s: entered.append(s) or contextlib.nullcontext()
    )
    object.__setattr__(module.torch, "npu", fake_npu)
    inp = _npu_labeled_tensor(module, torch.arange(1.0) + 7)
    out = _npu_labeled_tensor(module, torch.zeros(4))
    comm.all_gatherv(out, inp, [1, 3], stream=SimpleNamespace(npu_stream=9))
    assert comm.flagcx.flagcxBroadcast.call_count == 2
    assert comm.flagcx.flagcxGroupStart.call_args.args == ("comm-handle",)
    assert entered  # npu stream context used
    # Own slice restored by a local copy_: out[0:1] == inp (the mock broadcast
    # wrote nothing, so only this copy could have filled the slice).
    assert torch.equal(
        out[0:1].as_subclass(torch.Tensor), inp.as_subclass(torch.Tensor)
    )
    assert out[1:].as_subclass(torch.Tensor).equal(torch.zeros(3))


def test_reduce_scatterv_npu_group_forms(load_communicator):
    """reduce_scatterv on NPU uses the comm-carrying group forms."""
    module = load_communicator()
    comm = _cpu_communicator(module, device_type="npu", rank=0)
    inp = _npu_labeled_tensor(module, torch.zeros(4))
    out = _npu_labeled_tensor(module, torch.zeros(2))
    comm.reduce_scatterv(out, inp, [2, 2], stream=SimpleNamespace(npu_stream=9))
    assert comm.flagcx.flagcxReduce.call_count == 2
    assert comm.flagcx.flagcxGroupStart.call_args.args == ("comm-handle",)
    assert comm.flagcx.adaptor_stream_free.called

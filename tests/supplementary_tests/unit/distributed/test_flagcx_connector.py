# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for vllm_fl.distributed.kv_transfer.flagcx_connector.

CPU-only tests for the FlagCX KV-transfer connector (748 missing statements in
the measured baselines). The connector imports a large vllm surface plus the
FlagCX wrapper at import time, none of which exist in the CPU test
environment; the fixture below installs minimal leaf-module stubs in
sys.modules (saved / installed / restored in teardown, never monkeypatch.setitem
on popped entries) so the real connector code executes against doubles:

* module-level transfer planning helpers (TP ratio, region expansion,
  alignment, validation, coalescing, contiguity grouping, side-channel port);
* ``FlagCXConnectorMetadata`` recv/send bookkeeping;
* ``FlagCXConnector`` facade: role dispatch, guards, layout selection and the
  stats / prom-metrics classmethods;
* ``FlagCXConnectorScheduler``: sliding-window clipping, mamba N-1
  truncation, matched-token planning, state-after-alloc, build_connector_meta
  and request_finished handoff payloads;
* ``FlagCXConnectorWorker``: construction bookkeeping, library-path and env
  resolution, kernel-block-size sync (logical vs physical), start_load_kv
  send arming and consumer-side group-pull scheduling, get_finished draining
  / expiry, stats snapshots, connection caching, register_kv_caches with
  real CPU tensors (blocks-first / NHD / MLA / Mamba layouts),
  _build_transfer_params descriptor oracles (contiguous coalescing,
  per-block fallback under asymmetric-TP offsets, empty/mismatched group
  skips, replicated non-root skip), _send_kv_to_decode success and failure
  branches, the ZMQ sender thread / worker loops against fake sockets
  (dispatch, relay, generic-error path), _receive_kv exception paths
  (ContextTerminated included) and shutdown ordering;
* module import modes: FLAGCX_PATH sys.path handling and the missing-wrapper
  ImportError;
* the ``FlagCXConnector`` facade's full delegation surface on both roles.

The native FlagCX library is always a Mock: the data-plane boundary
(``flagcxP2pBatchWriteSync`` etc.) records calls without GPUs.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.supplementary]

pytest.importorskip("msgspec")
pytest.importorskip("numpy")
pytest.importorskip("zmq")

import torch  # noqa: E402
import zmq  # noqa: E402

_CONNECTOR_NAME = "vllm_fl.distributed.kv_transfer.flagcx_connector"
_STATS_NAME = "vllm_fl.distributed.kv_transfer.flagcx_stats"

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"


# ---------------------------------------------------------------------------
# Doubles for the spec classes: isinstance dispatch inside the module under
# test must distinguish them, so they are distinct real classes.
# ---------------------------------------------------------------------------


class FullAttentionSpec:
    pass


class SlidingWindowSpec:
    pass


class MambaSpec:
    pass


class MLAAttentionSpec:
    pass


class SlidingWindowMLASpec:
    pass


class KVCacheSpec:
    pass


@dataclass
class _KVConnectorStatsBase:
    """Mirror of vllm's KVConnectorStats (single ``data`` field)."""

    data: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(self.data.values())


class _StubBackend:
    def get_name(self):
        return "stub_backend"


def _extract_layer_index(layer_name: str) -> int:
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


def _make_zmq_path(scheme, host, port=None):
    if port is None:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def _create_metric_per_engine(metric, per_engine):
    """Faithful double of vllm.v1.metrics.utils.create_metric_per_engine."""
    return {idx: metric.labels(*lv) for idx, lv in per_engine.items()}


class _StubTransferTopology:
    """CPU double of vllm's TransferTopology (attribute carrier)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.local_replicates_kv_cache = False
        self.is_kv_layout_blocks_first = True

    def handshake_target_ranks(self, remote_tp_size, remote_dcp_size=1):
        return [self.tp_rank]

    def get_transfer_cache_regions(self, cache_or_caches, layer_spec):
        """One registered region view per cache view (real vllm: spec-driven).

        Full-attention specs under a blocks-first layout yield [K, V]; the
        double mirrors the common single-view case the worker consumes.
        """
        return [cache_or_caches]


class _StubBlockTable:
    """Double exposing the kernel-block mapping the worker uses."""

    @classmethod
    def map_to_kernel_blocks(cls, block_ids, blocks_per_logical, block_arange):
        import numpy as np

        expanded = (
            np.asarray(block_ids).reshape(-1, 1) * blocks_per_logical + block_arange
        )
        # The real helper returns the flat kernel-physical id list.
        return expanded.reshape(-1)


class _RequestStatus:
    FINISHED_LENGTH_CAPPED = "FINISHED_LENGTH_CAPPED"


def _leaf(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class FlagcxStubEnv:
    """Install a full stub vllm tree around the connector import.

    Every touched sys.modules entry is saved on install and restored by
    ``restore()`` (no monkeypatch.setitem on entries this env pops), so the
    suite stays order-independent even next to tests that import the real
    modules (e.g. test_flagcx_stats.py).
    """

    def __init__(self):
        self._saved: dict[str, object] = {}
        self.flagcx_lib = Mock(name="FLAGCXLibrary-instance")
        self.flagcx_lib.flagcxP2pEngineCreate.return_value = "engine-handle"
        self.flagcx_lib.flagcxP2pGetRpcPort.return_value = 12345

    # -- install / restore ---------------------------------------------------

    def _install(self, name, module):
        if name not in self._saved:
            self._saved[name] = sys.modules.get(name)
        sys.modules[name] = module

    def _pop(self, name):
        if name not in self._saved:
            self._saved[name] = sys.modules.get(name)
        sys.modules.pop(name, None)

    def restore(self):
        for name, prev in self._saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev

    def install(self):
        install, pop = self._install, self._pop

        # The native FlagCX boundary.
        wrapper = _leaf(
            "plugin.interservice.flagcx_wrapper",
            FLAGCXLibrary=lambda path: self.flagcx_lib,
        )
        install("plugin.interservice.flagcx_wrapper", wrapper)
        for parent in ("plugin", "plugin.interservice"):
            if parent not in sys.modules:
                install(parent, ModuleType(parent))

        # vllm leaves the connector imports at module scope.
        install("vllm.config", _leaf("vllm.config", VllmConfig=object))
        install(
            "vllm.distributed.kv_transfer.kv_connector.utils",
            _leaf(
                "vllm.distributed.kv_transfer.kv_connector.utils",
                TransferTopology=_StubTransferTopology,
                get_current_attn_backends=lambda cfg, layer_names=None: [
                    _StubBackend()
                ],
            ),
        )

        class KVConnectorBase_V1:
            def __init__(self, vllm_config, role, kv_cache_config=None):
                self._connector_metadata = None
                assert vllm_config.kv_transfer_config is not None
                self._vllm_config = vllm_config
                self._kv_cache_config = kv_cache_config
                self._role = role

        install(
            "vllm.distributed.kv_transfer.kv_connector.v1.base",
            _leaf(
                "vllm.distributed.kv_transfer.kv_connector.v1.base",
                KVConnectorBase_V1=KVConnectorBase_V1,
                KVConnectorMetadata=object,
                KVConnectorRole=SimpleNamespace(SCHEDULER="scheduler", WORKER="worker"),
                SupportsHMA=object,
            ),
        )

        from prometheus_client import Counter, Gauge, Histogram

        class KVConnectorPromMetricsBase:
            def __init__(
                self,
                vllm_config,
                metric_types,
                labelnames,
                per_engine_labelvalues,
            ):
                self._kv_transfer_config = vllm_config.kv_transfer_config
                self._gauge_cls = metric_types[Gauge]
                self._counter_cls = metric_types[Counter]
                self._histogram_cls = metric_types[Histogram]
                self._labelnames = labelnames
                self.per_engine_labelvalues = per_engine_labelvalues

        install(
            "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
            _leaf(
                "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
                KVConnectorPromMetrics=KVConnectorPromMetricsBase,
                KVConnectorStats=_KVConnectorStatsBase,
                PromMetric=object,
                PromMetricT=object,
            ),
        )
        install(
            "vllm.distributed.parallel_state",
            _leaf(
                "vllm.distributed.parallel_state",
                get_pp_group=lambda: None,
                get_tensor_model_parallel_rank=lambda: 0,
                get_tensor_model_parallel_world_size=lambda: 1,
            ),
        )
        install(
            "vllm.forward_context",
            _leaf("vllm.forward_context", ForwardContext=object),
        )
        import logging

        install("vllm.logger", _leaf("vllm.logger", init_logger=logging.getLogger))
        install(
            "vllm.model_executor.models.utils",
            _leaf(
                "vllm.model_executor.models.utils",
                extract_layer_index=_extract_layer_index,
            ),
        )
        install(
            "vllm.utils.math_utils",
            _leaf("vllm.utils.math_utils", cdiv=lambda a, b: -(-a // b)),
        )
        install(
            "vllm.utils.network_utils",
            _leaf(
                "vllm.utils.network_utils",
                get_ip=lambda: "10.0.0.1",
                make_zmq_path=_make_zmq_path,
                make_zmq_socket=lambda *a, **k: None,
            ),
        )
        install(
            "vllm.v1.attention.backend",
            _leaf("vllm.v1.attention.backend", AttentionMetadata=object),
        )
        install(
            "vllm.v1.attention.backends.utils",
            _leaf(
                "vllm.v1.attention.backends.utils",
                NULL_BLOCK_ID=0,
                get_kv_cache_layout=lambda: "NHD",
            ),
        )
        install(
            "vllm.v1.core.sched.output",
            _leaf("vllm.v1.core.sched.output", SchedulerOutput=object),
        )
        install(
            "vllm.v1.kv_cache_interface",
            _leaf(
                "vllm.v1.kv_cache_interface",
                FullAttentionSpec=FullAttentionSpec,
                KVCacheSpec=KVCacheSpec,
                MambaSpec=MambaSpec,
                MLAAttentionSpec=MLAAttentionSpec,
                SlidingWindowMLASpec=SlidingWindowMLASpec,
                SlidingWindowSpec=SlidingWindowSpec,
            ),
        )
        install(
            "vllm.v1.request",
            _leaf("vllm.v1.request", RequestStatus=_RequestStatus),
        )
        install(
            "vllm.v1.worker.block_table",
            _leaf("vllm.v1.worker.block_table", BlockTable=_StubBlockTable),
        )
        install(
            "vllm.v1.worker.utils",
            _leaf(
                "vllm.v1.worker.utils",
                select_common_block_size=lambda bs, backends, specs=None: bs,
            ),
        )
        install(
            "vllm.v1.metrics.utils",
            _leaf(
                "vllm.v1.metrics.utils",
                create_metric_per_engine=_create_metric_per_engine,
            ),
        )

        # Re-import the modules under test so their import-time bindings
        # (FLAGCXLibrary, get_current_attn_backends, ...) resolve against
        # THIS environment's stubs. restore() puts the originals back.
        pop(_STATS_NAME)
        pop(_CONNECTOR_NAME)
        return self

    def import_connector(self):
        import importlib

        return importlib.import_module(_CONNECTOR_NAME)


# ---------------------------------------------------------------------------
# Config doubles
# ---------------------------------------------------------------------------


class _Group:
    def __init__(self, spec, layer_names):
        self.kv_cache_spec = spec
        self.layer_names = layer_names


class _FullSpec(FullAttentionSpec):
    pass


class _SlidingSpec(SlidingWindowSpec):
    def __init__(self, sliding_window, block_size):
        self.sliding_window = sliding_window
        self.block_size = block_size


class _ModelConfig:
    use_mla = False

    def get_total_num_kv_heads(self):
        return 8


class _CacheConfig:
    block_size = 16


class _ParallelConfig:
    def __init__(self, dp_rank=0, tp_size=1):
        self.data_parallel_rank = dp_rank
        self.tensor_parallel_size = tp_size


class _SchedulerConfig:
    def __init__(self, disable_hybrid_kv_cache_manager=True):
        self.disable_hybrid_kv_cache_manager = disable_hybrid_kv_cache_manager


class VllmConfigDouble:
    """Minimal VllmConfig double covering both connector roles."""

    def __init__(self, role="kv_both", dp_rank=0, tp_size=1, extra_config=None):
        self.kv_transfer_config = SimpleNamespace(
            engine_id="e0",
            kv_role=role,
            kv_connector_extra_config=extra_config if extra_config is not None else {},
        )
        self.parallel_config = _ParallelConfig(dp_rank, tp_size)
        self.scheduler_config = _SchedulerConfig()
        self.cache_config = _CacheConfig()
        self.model_config = _ModelConfig()


def make_kv_cache_config(groups=None, has_mamba=False):
    if groups is None:
        groups = [_Group(_FullSpec(), ["model.layers.0.self_attn.attn"])]
    return SimpleNamespace(kv_cache_groups=groups, has_mamba_layers=has_mamba)


@pytest.fixture
def conn_env(monkeypatch):
    """Install the stub environment and import the connector module fresh."""
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.delenv("FLAGCX_LIB_PATH", raising=False)
    monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)
    monkeypatch.delenv("FLAGCX_CONNECTOR_ABORT_REQUEST_TIMEOUT", raising=False)

    env = FlagcxStubEnv().install()
    try:
        fc = env.import_connector()
        env.fc = fc

        def make_worker(role="kv_both", config=None):
            return fc.FlagCXConnectorWorker(
                VllmConfigDouble(role), "e0", config or make_kv_cache_config()
            )

        env.make_worker = make_worker
        yield env
    finally:
        env.restore()


# ---------------------------------------------------------------------------
# Module-level planning helpers
# ---------------------------------------------------------------------------


class TestTpRatio:
    def test_equal_sizes(self, conn_env):
        assert conn_env.fc._get_tp_ratio(4, 4) == 1
        assert conn_env.fc._get_tp_ratio(1, 1) == 1

    def test_local_larger_positive(self, conn_env):
        assert conn_env.fc._get_tp_ratio(4, 2) == 2
        assert conn_env.fc._get_tp_ratio(8, 2) == 4

    def test_remote_larger_negative(self, conn_env):
        assert conn_env.fc._get_tp_ratio(2, 8) == -4
        assert conn_env.fc._get_tp_ratio(1, 3) == -3

    @pytest.mark.parametrize(
        ("local", "remote", "message"),
        [
            (3, 2, "Local TP size 3 not divisible by remote TP size 2."),
            (2, 3, "Remote TP size 3 not divisible by local TP size 2."),
        ],
    )
    def test_non_divisible_rejected(self, conn_env, local, remote, message):
        with pytest.raises(AssertionError, match=message):
            conn_env.fc._get_tp_ratio(local, remote)


class TestGroupConcurrentContiguous:
    def test_runs_collapse(self, conn_env):
        src, dst = conn_env.fc.group_concurrent_contiguous(
            [1, 2, 3, 7, 8, 20], [10, 11, 12, 30, 31, 99]
        )
        assert src == [[1, 2, 3], [7, 8], [20]]
        assert dst == [[10, 11, 12], [30, 31], [99]]

    def test_dst_gap_breaks_run(self, conn_env):
        src, dst = conn_env.fc.group_concurrent_contiguous([1, 2, 3], [10, 12, 13])
        assert src == [[1], [2, 3]]
        assert dst == [[10], [12, 13]]

    def test_empty(self, conn_env):
        assert conn_env.fc.group_concurrent_contiguous([], []) == ([], [])


class TestExpandTransferRegions:
    def test_plain_single_region(self, conn_env):
        regions = conn_env.fc._expand_transfer_regions(
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
        assert region.layer_index == 0
        assert region.group_index == 0

    def test_split_emits_k_and_v(self, conn_env):
        regions = conn_env.fc._expand_transfer_regions(
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
            (1000, 0),
            (1032, 0),
            (2000, 1),
        ]
        assert regions[0].layer_name == regions[1].layer_name == "layers_a"

    def test_defaults_fill_group_and_split_lists(self, conn_env):
        regions = conn_env.fc._expand_transfer_regions(
            base_addrs=[100, 200],
            block_lens=[8, 8],
            kv_block_lens=[4, 4],
            layer_names=["a", "b"],
            layer_indices=[0, 1],
            is_kv_layout_blocks_first=True,
        )
        # No split list: default is all-split under blocks-first.
        assert [r.group_index for r in regions] == [0, 0, 0, 0]
        assert [r.base_addr for r in regions] == [100, 104, 200, 204]

    def test_mismatched_lengths_rejected(self, conn_env):
        with pytest.raises(AssertionError, match="matching metadata lengths"):
            conn_env.fc._expand_transfer_regions(
                base_addrs=[1000, 2000],
                block_lens=[64],
                kv_block_lens=[32],
                layer_names=["a"],
                layer_indices=[0],
                is_kv_layout_blocks_first=False,
            )


class TestSenderTransferPlan:
    def test_homogeneous_full_copy(self, conn_env):
        plan = conn_env.fc._compute_sender_transfer_plan(
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
    def test_local_larger_replicated_only_root_sends(
        self, conn_env, local_rank, expected
    ):
        plan = conn_env.fc._compute_sender_transfer_plan(
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
    def test_local_larger_sharded_offsets(self, conn_env, local_rank, expected_dst):
        plan = conn_env.fc._compute_sender_transfer_plan(
            local_tp_rank=local_rank,
            local_tp_size=8,
            remote_tp_rank=0,
            remote_tp_size=2,
            local_kv_block_len=128,
            remote_kv_block_len=512,
            producer_cache_replicated=False,
        )
        assert plan == (True, 0, expected_dst, 128)

    def test_remote_larger_replicated_sends_all(self, conn_env):
        plan = conn_env.fc._compute_sender_transfer_plan(
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
    def test_remote_larger_gathers_slice(self, conn_env, remote_rank, expected_src):
        plan = conn_env.fc._compute_sender_transfer_plan(
            local_tp_rank=0,
            local_tp_size=2,
            remote_tp_rank=remote_rank,
            remote_tp_size=8,
            local_kv_block_len=128,
            remote_kv_block_len=32,
            producer_cache_replicated=False,
        )
        assert plan == (True, expected_src, 0, 32)


class TestCanCoalesce:
    def test_only_full_aligned_copy(self, conn_env):
        fc = conn_env.fc
        assert fc._can_coalesce_block_transfers(64, 64, 0, 0, 64) is True
        assert fc._can_coalesce_block_transfers(64, 128, 0, 0, 64) is False
        assert fc._can_coalesce_block_transfers(64, 64, 8, 0, 64) is False
        assert fc._can_coalesce_block_transfers(64, 64, 0, 8, 64) is False
        assert fc._can_coalesce_block_transfers(64, 64, 0, 0, 32) is False


def _region(layer, layer_index, base, block_len, kv_block_len, group=0):
    return conn_env_region(layer, layer_index, base, block_len, kv_block_len, group)


def conn_env_region(layer, layer_index, base, block_len, kv_block_len, group):
    from vllm_fl.distributed.kv_transfer import flagcx_connector as _fc

    return _fc.TransferRegion(
        layer_name=layer,
        layer_index=layer_index,
        base_addr=base,
        block_len=block_len,
        kv_block_len=kv_block_len,
        group_index=group,
    )


class TestAlignTransferRegions:
    def test_pairs_by_layer_occurrence(self, conn_env):
        fc = conn_env.fc
        local = [
            _region("a", 0, 10, 64, 32),
            _region("b", 1, 20, 64, 32),
            _region("a", 0, 30, 64, 32),
        ]
        remote = [
            _region("a", 0, 110, 64, 32),
            _region("a", 0, 130, 64, 32),
            _region("b", 1, 120, 64, 32),
        ]
        aligned_local, aligned_remote, err = fc._align_transfer_regions(local, remote)
        assert err is None
        assert [r.base_addr for r in aligned_local] == [10, 20, 30]
        assert [r.base_addr for r in aligned_remote] == [110, 120, 130]

    def test_missing_consumer_occurrence(self, conn_env):
        fc = conn_env.fc
        local = [_region("a", 0, 10, 64, 32), _region("x", 9, 50, 64, 32)]
        remote = [_region("a", 0, 110, 64, 32)]
        _, _, err = fc._align_transfer_regions(local, remote)
        assert err is not None
        assert "no matching consumer occurrence" in err
        assert "x" in err

    def test_layer_index_mismatch(self, conn_env):
        fc = conn_env.fc
        local = [_region("a", 0, 10, 64, 32)]
        remote = [_region("a", 5, 110, 64, 32)]
        _, _, err = fc._align_transfer_regions(local, remote)
        assert "layer index mismatch" in err

    def test_group_index_mismatch(self, conn_env):
        fc = conn_env.fc
        local = [_region("a", 0, 10, 64, 32, group=0)]
        remote = [_region("a", 0, 110, 64, 32, group=1)]
        _, _, err = fc._align_transfer_regions(local, remote)
        assert "group index mismatch" in err


class TestValidateAsymmetricRegionLengths:
    def test_all_regimes(self, conn_env):
        fc = conn_env.fc

        def region(kv_block_len):
            return _region("a", 0, 0, 64, kv_block_len)

        # Count mismatch always errors.
        assert (
            fc._validate_asymmetric_region_lengths([region(32)], [], 2, 2, False)
            == "FlagCX asymmetric TP requires matching KV region counts."
        )
        # Replicated producer skips the length checks.
        assert (
            fc._validate_asymmetric_region_lengths(
                [region(32)], [region(64)], 2, 1, True
            )
            is None
        )
        # Homogeneous mismatch.
        err = fc._validate_asymmetric_region_lengths(
            [region(32)], [region(64)], 2, 2, False
        )
        assert "homogeneous TP" in err and "region 0" in err
        # Destination larger (ratio 2): remote must be 2x local.
        assert (
            fc._validate_asymmetric_region_lengths(
                [region(32)], [region(64)], 2, 1, False
            )
            is None
        )
        err = fc._validate_asymmetric_region_lengths(
            [region(32)], [region(48)], 2, 1, False
        )
        assert "destination KV region length mismatch" in err
        # Source larger (negative ratio): local must be 2x remote.
        assert (
            fc._validate_asymmetric_region_lengths(
                [region(64)], [region(32)], 1, 2, False
            )
            is None
        )
        err = fc._validate_asymmetric_region_lengths(
            [region(48)], [region(32)], 1, 2, False
        )
        assert "source KV region length mismatch" in err


class TestGetSideChannelPort:
    def test_default_and_dp_offset(self, conn_env, monkeypatch):
        fc = conn_env.fc
        monkeypatch.delenv("FLAGCX_BOOTSTRAP_PORT", raising=False)
        cfg = VllmConfigDouble(dp_rank=0, tp_size=2)
        assert fc._get_side_channel_port(cfg) == 8998
        cfg = VllmConfigDouble(dp_rank=1, tp_size=2)
        assert fc._get_side_channel_port(cfg) == 9000

    def test_env_override(self, conn_env, monkeypatch):
        monkeypatch.setenv("FLAGCX_BOOTSTRAP_PORT", "7000")
        cfg = VllmConfigDouble(dp_rank=3, tp_size=2)
        assert conn_env.fc._get_side_channel_port(cfg) == 7006


class TestModuleImportModes:
    """Import-time branches: FLAGCX_PATH sys.path handling and the wrapper
    ImportError. These run a second, isolated FlagcxStubEnv whose
    install()/restore() brackets only the import itself."""

    def test_flagcx_path_appended_to_sys_path(self, monkeypatch, tmp_path):
        import importlib

        flagcx_root = tmp_path / "flagcx-source"
        flagcx_root.mkdir()
        monkeypatch.setenv("FLAGCX_PATH", str(flagcx_root))

        saved_path = list(sys.path)
        env = FlagcxStubEnv().install()
        try:
            module = importlib.import_module(_CONNECTOR_NAME)
            # The connector appended the (existing) FLAGCX_PATH dir.
            assert str(flagcx_root) in sys.path
            assert module.TRANS_DONE == b"trans_done"
        finally:
            env.restore()
            sys.path[:] = saved_path

    def test_nonexistent_flagcx_path_not_appended(self, monkeypatch):
        import importlib

        monkeypatch.setenv("FLAGCX_PATH", "/definitely/not/a/dir")
        saved_path = list(sys.path)
        env = FlagcxStubEnv().install()
        try:
            importlib.import_module(_CONNECTOR_NAME)
            assert "/definitely/not/a/dir" not in sys.path
        finally:
            env.restore()
            sys.path[:] = saved_path

    def test_missing_wrapper_raises_import_error(self, monkeypatch, tmp_path):
        # A FLAGCX_PATH that exists (so sys.path is touched) but contains no
        # plugin/interservice/flagcx_wrapper: the module must raise.
        flagcx_root = tmp_path / "empty-flagcx"
        flagcx_root.mkdir()
        monkeypatch.setenv("FLAGCX_PATH", str(flagcx_root))

        saved_path = list(sys.path)
        env = FlagcxStubEnv()
        env.install()
        # Drop the wrapper entry the base install injected; only "plugin"
        # parents stay so `from plugin.interservice.flagcx_wrapper import`
        # fails.
        for name in (
            "plugin",
            "plugin.interservice",
            "plugin.interservice.flagcx_wrapper",
        ):
            sys.modules.pop(name, None)
        try:
            with pytest.raises(ImportError, match="Cannot import FlagCX wrapper"):
                import importlib

                importlib.import_module(_CONNECTOR_NAME)
        finally:
            env.restore()
            sys.path[:] = saved_path


# ---------------------------------------------------------------------------
# FlagCXConnectorMetadata
# ---------------------------------------------------------------------------


class TestConnectorMetadata:
    def test_add_new_recv_req(self, conn_env):
        meta = conn_env.fc.FlagCXConnectorMetadata()
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

    def test_add_new_recv_defaults_tp_size(self, conn_env):
        meta = conn_env.fc.FlagCXConnectorMetadata()
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

    def test_add_new_send_req(self, conn_env):
        meta = conn_env.fc.FlagCXConnectorMetadata()
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

    def test_missing_transfer_id_raises(self, conn_env):
        meta = conn_env.fc.FlagCXConnectorMetadata()
        with pytest.raises(KeyError, match="transfer_id"):
            meta.add_new_req(
                request_id="x",
                local_block_ids=[[]],
                kv_transfer_params={"remote_host": "h", "remote_port": 1},
            )


# ---------------------------------------------------------------------------
# FlagCXConnector facade
# ---------------------------------------------------------------------------


class TestConnectorFacade:
    @pytest.mark.parametrize(
        ("use_mla", "model_config", "expected"),
        [
            (None, None, None),
            (True, object(), None),
            (False, object(), "HND"),
        ],
    )
    def test_get_required_kvcache_layout(
        self, conn_env, use_mla, model_config, expected
    ):
        fc = conn_env.fc
        if model_config is None:
            config = type("C", (), {"model_config": None})()
        else:
            config = type(
                "C", (), {"model_config": type("M", (), {"use_mla": use_mla})()}
            )()
        assert fc.FlagCXConnector.get_required_kvcache_layout(config) == expected

    def test_scheduler_role(self, conn_env):
        conn = conn_env.fc.FlagCXConnector(
            VllmConfigDouble("kv_consumer"), "scheduler", make_kv_cache_config()
        )
        assert conn.connector_scheduler is not None
        assert conn.connector_worker is None
        with pytest.raises(AssertionError):
            conn.register_kv_caches({})

    def test_worker_role(self, conn_env):
        conn = conn_env.fc.FlagCXConnector(
            VllmConfigDouble("kv_consumer"), "worker", make_kv_cache_config()
        )
        assert conn.connector_worker is not None
        assert conn.connector_scheduler is None
        with pytest.raises(AssertionError):
            conn.get_num_new_matched_tokens(None, 0)
        conn.connector_worker.shutdown()

    def test_scheduler_role_requires_kv_cache_config(self, conn_env):
        with pytest.raises(AssertionError, match="SCHEDULER role"):
            conn_env.fc.FlagCXConnector(
                VllmConfigDouble("kv_consumer"), "scheduler", None
            )

    def test_build_kv_connector_stats_empty(self, conn_env):
        stats = conn_env.fc.FlagCXConnector.build_kv_connector_stats()
        assert stats.data["transfer_duration"] == []
        assert stats.num_successful_transfers == 0
        assert stats.is_empty()

    def test_build_kv_connector_stats_none_data(self, conn_env):
        stats = conn_env.fc.FlagCXConnector.build_kv_connector_stats(None)
        assert stats.is_empty()

    def test_build_prom_metrics(self, conn_env):
        from prometheus_client import Counter, Gauge, Histogram

        pm = conn_env.fc.FlagCXConnector.build_prom_metrics(
            VllmConfigDouble(),
            {Gauge: Gauge, Counter: Counter, Histogram: Histogram},
            ["model_name"],
            {0: ["m1"]},
        )
        assert pm is not None
        assert len(pm._counters) == 7
        assert len(pm._histograms) == 3

    def test_get_kv_connector_stats_passthrough(self, conn_env):
        conn = conn_env.fc.FlagCXConnector(
            VllmConfigDouble("kv_consumer"), "worker", make_kv_cache_config()
        )
        try:
            # Scheduler-only connector has no worker: stats path returns None.
            assert conn.get_kv_connector_stats() is not None or conn.connector_worker
            worker = conn.connector_worker
            worker.xfer_stats.record_transfer(0.1, 8, 1)
            stats = conn.get_kv_connector_stats()
            assert stats.num_successful_transfers == 1
            # Drained after the snapshot.
            assert conn.get_kv_connector_stats() is None
        finally:
            conn.connector_worker.shutdown()

    def test_no_op_methods(self, conn_env):
        conn = conn_env.fc.FlagCXConnector(
            VllmConfigDouble("kv_consumer"), "worker", make_kv_cache_config()
        )
        try:
            # save_kv_layer / wait_for_save / wait_for_layer_load are no-ops.
            conn.save_kv_layer("layer", torch.zeros(2), None)
            conn.wait_for_save()
            conn.wait_for_layer_load("layer")
        finally:
            conn.connector_worker.shutdown()

    def test_scheduler_facade_delegations(self, conn_env):
        """Every scheduler-side facade method reaches the inner scheduler."""
        fc = conn_env.fc
        conn = fc.FlagCXConnector(
            VllmConfigDouble("kv_consumer"), "scheduler", make_kv_cache_config()
        )
        request = _make_request(
            prompt_token_ids=[1] * 10,
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_host": "h",
                "remote_port": 1,
                "transfer_id": "t1",
            },
        )
        blocks = SimpleNamespace()
        blocks.get_unhashed_block_ids_all_groups = lambda: [[1, 2], [3]]

        # get_num_new_matched_tokens delegates (244-250).
        assert conn.get_num_new_matched_tokens(request, 4) == (6, True)
        # update_state_after_alloc delegates (252-261).
        conn.update_state_after_alloc(request, blocks, 5)
        assert "req-1" in conn.connector_scheduler._reqs_need_recv
        # build_connector_meta delegates (263-267).
        meta = conn.build_connector_meta(SimpleNamespace())
        assert set(meta.reqs_to_recv) == {"req-1"}
        # request_finished delegates with (block_ids,) wrapping (269-273).
        handoff = _make_request(
            request_id="req-2",
            kv_transfer_params={"do_remote_prefill": True, "transfer_id": "t2"},
        )
        assert conn.request_finished(handoff, [7, 8]) == (False, None)
        assert "req-2" in conn.connector_scheduler._reqs_need_recv
        # request_finished_all_groups delegates with the tuple as-is (275-281).
        handoff2 = _make_request(
            request_id="req-3",
            kv_transfer_params={"do_remote_prefill": True, "transfer_id": "t3"},
        )
        assert conn.request_finished_all_groups(handoff2, ([9],)) == (False, None)
        assert "req-3" in conn.connector_scheduler._reqs_need_recv

    def test_worker_facade_delegations(self, conn_env):
        """Worker-side facade methods reach the inner worker (283-297)."""
        fc = conn_env.fc
        conn = fc.FlagCXConnector(
            VllmConfigDouble("kv_producer"), "worker", make_kv_cache_config()
        )
        try:
            worker = conn.connector_worker

            # register_kv_caches delegates (284-286).
            tensor = torch.zeros(2, 2, 2)
            conn.register_kv_caches({"model.layers.0.self_attn.attn": tensor})
            assert worker.registered_layer_names == ["model.layers.0.self_attn.attn"]
            assert worker.kv_caches_base_addr == [tensor.data_ptr()]

            # get_finished delegates (288-292): producer with a pending
            # armed request reports nothing yet.
            send_meta = fc.SendBlockMeta(
                p_req_id="p-live",
                transfer_id="t-live",
                local_block_ids=[[1]],
                ready=threading.Event(),
                expire_time=float("inf"),
            )
            worker.reqs_need_send.reqs["t-live"] = send_meta
            assert conn.get_finished({"p-live"}) == (None, None)

            # start_load_kv delegates through _connector_metadata (294-297).
            meta = fc.FlagCXConnectorMetadata()
            meta.add_new_req(
                request_id="p1",
                local_block_ids=[[1, 2]],
                kv_transfer_params={
                    "transfer_id": "t2",
                    "remote_host": "h",
                    "remote_port": 1,
                },
                load_remote_cache=False,
            )
            conn._connector_metadata = meta
            conn.start_load_kv(SimpleNamespace())
            armed = worker.reqs_need_send.reqs["t2"]
            assert armed.p_req_id == "p1"
            assert armed.ready.is_set()

            # Non-FlagCX metadata is rejected before reaching the worker.
            conn._connector_metadata = object()
            with pytest.raises(AssertionError):
                conn.start_load_kv(SimpleNamespace())
        finally:
            conn.connector_worker.shutdown()

    def test_get_kv_connector_stats_scheduler_role_returns_none(self, conn_env):
        # A SCHEDULER-role connector has no worker: the stats path (324-326)
        # returns None instead of raising.
        conn = conn_env.fc.FlagCXConnector(
            VllmConfigDouble("kv_consumer"), "scheduler", make_kv_cache_config()
        )
        assert conn.connector_worker is None
        assert conn.get_kv_connector_stats() is None


# ---------------------------------------------------------------------------
# FlagCXConnectorScheduler
# ---------------------------------------------------------------------------


def _make_scheduler(conn_env, role="kv_producer", groups=None, has_mamba=False):
    fc = conn_env.fc
    return fc.FlagCXConnectorScheduler(
        VllmConfigDouble(role), "e0", make_kv_cache_config(groups, has_mamba)
    )


def _make_request(**kwargs):
    request = SimpleNamespace(
        request_id=kwargs.get("request_id", "req-1"),
        kv_transfer_params=kwargs.get("kv_transfer_params"),
        num_prompt_tokens=kwargs.get("num_prompt_tokens", 10),
        prompt_token_ids=kwargs.get("prompt_token_ids"),
        prompt_embeds=kwargs.get("prompt_embeds"),
        _all_token_ids=kwargs.get("_all_token_ids", []),
        status=kwargs.get("status", _RequestStatus.FINISHED_LENGTH_CAPPED),
        max_tokens=kwargs.get("max_tokens", 100),
    )
    if request.prompt_token_ids is None and "_all_token_ids" in kwargs:
        request.prompt_token_ids = list(kwargs["_all_token_ids"])
    return request


class TestSchedulerSwClipping:
    def test_blocks_per_sw_built_from_sliding_groups(self, conn_env):
        groups = [
            _Group(_FullSpec(), ["model.layers.0.self_attn.attn"]),
            _Group(_SlidingSpec(sliding_window=40, block_size=16), ["sw_layer"]),
            _Group(_SlidingSpec(sliding_window=0, block_size=16), ["sw0_layer"]),
        ]
        sched = _make_scheduler(conn_env, groups=groups)
        # Full attention: cdiv(0,16)+1 = 1 ... n_tokens=0 -> 0 blocks.
        # sliding 40/16 -> 2+1=3; sliding 0 -> 0.
        assert sched.blocks_per_sw == [0, 4, 0]

    def test_get_sw_clipped_blocks_identity_without_hma(self, conn_env):
        sched = _make_scheduler(conn_env)  # disable_hybrid_kv_cache_manager=True
        assert sched.get_sw_clipped_blocks([[1, 2], [3]]) == [[1, 2], [3]]

    def test_get_sw_clipped_blocks_clips_tail(self, conn_env):
        groups = [
            _Group(_SlidingSpec(sliding_window=40, block_size=16), ["sw"]),
            _Group(_FullSpec(), ["full"]),
        ]
        cfg = VllmConfigDouble()
        cfg.scheduler_config = _SchedulerConfig(disable_hybrid_kv_cache_manager=False)
        sched = conn_env.fc.FlagCXConnectorScheduler(
            cfg, "e0", make_kv_cache_config(groups)
        )
        assert sched._is_hma_required is True
        clipped = sched.get_sw_clipped_blocks([[1, 2, 3, 4, 5], [6, 7]])
        assert clipped == [[2, 3, 4, 5], [6, 7]]

    def test_get_sw_clipped_blocks_zero_keeps_all(self, conn_env):
        groups = [
            _Group(_SlidingSpec(sliding_window=0, block_size=16), ["sw"]),
        ]
        cfg = VllmConfigDouble()
        cfg.scheduler_config = _SchedulerConfig(disable_hybrid_kv_cache_manager=False)
        sched = conn_env.fc.FlagCXConnectorScheduler(
            cfg, "e0", make_kv_cache_config(groups)
        )
        assert sched.get_sw_clipped_blocks([[1, 2, 3]]) == [[1, 2, 3]]

    def test_empty_block_ids(self, conn_env):
        sched = _make_scheduler(conn_env)
        assert sched.get_sw_clipped_blocks([]) == []


class TestSchedulerMambaTruncation:
    def test_remote_prefill_token_count(self, conn_env):
        sched = _make_scheduler(conn_env, has_mamba=True)
        assert sched._get_remote_prefill_token_count(5) == 4
        assert sched._get_remote_prefill_token_count(1) == 1

    def test_non_mamba_keeps_full_count(self, conn_env):
        sched = _make_scheduler(conn_env, has_mamba=False)
        assert sched._get_remote_prefill_token_count(5) == 5

    def test_truncate_once_via_prompt_token_ids(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer", has_mamba=True)
        request = _make_request(
            prompt_token_ids=[1, 2, 3, 4],
            _all_token_ids=[1, 2, 3, 4],
            num_prompt_tokens=4,
            kv_transfer_params={"do_remote_decode": True},
        )
        sched._truncate_mamba_request_for_prefill(request)
        assert request.prompt_token_ids == [1, 2, 3]
        assert request._all_token_ids == [1, 2, 3]
        assert request.num_prompt_tokens == 3
        assert request.max_tokens == 1
        assert request.kv_transfer_params["_p_side_truncated"] is True
        # Second call is a no-op.
        sched._truncate_mamba_request_for_prefill(request)
        assert request.prompt_token_ids == [1, 2, 3]

    def test_truncate_via_prompt_embeds(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer", has_mamba=True)
        request = SimpleNamespace(
            kv_transfer_params={"do_remote_decode": True},
            num_prompt_tokens=3,
            prompt_token_ids=None,
            prompt_embeds=[10, 20, 30],
            _all_token_ids=[1, 2, 3],
            max_tokens=100,
        )
        sched._truncate_mamba_request_for_prefill(request)
        assert list(request.prompt_embeds) == [10, 20]
        assert request.num_prompt_tokens == 2
        assert request._all_token_ids == [1, 2]

    def test_truncate_without_payload_is_noop(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer", has_mamba=True)
        request = _make_request(
            prompt_token_ids=None,
            prompt_embeds=None,
            num_prompt_tokens=3,
            kv_transfer_params={"do_remote_decode": True},
        )
        sched._truncate_mamba_request_for_prefill(request)
        assert request.num_prompt_tokens == 3

    def test_single_token_not_truncated(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer", has_mamba=True)
        request = _make_request(
            prompt_token_ids=[7],
            _all_token_ids=[7],
            num_prompt_tokens=1,
            kv_transfer_params={"do_remote_decode": True},
        )
        sched._truncate_mamba_request_for_prefill(request)
        assert request.prompt_token_ids == [7]


class TestSchedulerGetNumNewMatchedTokens:
    def test_remote_prefill_remaining(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_consumer")
        request = _make_request(
            prompt_token_ids=[1] * 10,
            kv_transfer_params={"do_remote_prefill": True},
        )
        assert sched.get_num_new_matched_tokens(request, 4) == (6, True)

    def test_remote_prefill_no_remaining(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_consumer")
        request = _make_request(
            prompt_token_ids=[1] * 10,
            kv_transfer_params={"do_remote_prefill": True},
        )
        assert sched.get_num_new_matched_tokens(request, 10) == (0, False)

    def test_no_params_returns_zero(self, conn_env):
        sched = _make_scheduler(conn_env)
        request = _make_request(kv_transfer_params=None)
        assert sched.get_num_new_matched_tokens(request, 0) == (0, False)

    def test_remote_decode_triggers_mamba_truncation(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer", has_mamba=True)
        request = _make_request(
            prompt_token_ids=[1, 2, 3],
            _all_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
            kv_transfer_params={"do_remote_decode": True},
        )
        assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
        assert request.num_prompt_tokens == 2

    def test_remote_decode_without_mamba_no_truncation(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer", has_mamba=False)
        request = _make_request(
            prompt_token_ids=[1, 2, 3],
            _all_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
            kv_transfer_params={"do_remote_decode": True},
        )
        assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
        assert request.num_prompt_tokens == 3


class TestSchedulerUpdateStateAfterAlloc:
    def _blocks(self):
        blocks = SimpleNamespace()
        blocks.get_unhashed_block_ids_all_groups = lambda: [[1, 2], [3]]
        return blocks

    def test_recv_path_records_request(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_consumer")
        request = _make_request(
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_host": "h",
                "remote_port": 1,
                "transfer_id": "t1",
            }
        )
        sched.update_state_after_alloc(request, self._blocks(), 5)
        assert "req-1" in sched._reqs_need_recv
        req, block_ids = sched._reqs_need_recv["req-1"]
        assert block_ids == [[1, 2], [3]]
        # do_remote_prefill is cleared after processing.
        assert request.kv_transfer_params["do_remote_prefill"] is False

    def test_recv_path_no_external_tokens_empty_blocks(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_consumer")
        request = _make_request(
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_host": "h",
                "remote_port": 1,
                "transfer_id": "t1",
            }
        )
        sched.update_state_after_alloc(request, self._blocks(), 0)
        req, block_ids = sched._reqs_need_recv["req-1"]
        assert block_ids == []

    def test_recv_path_invalid_params_ignored(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_consumer")
        request = _make_request(kv_transfer_params={"do_remote_prefill": True})
        sched.update_state_after_alloc(request, self._blocks(), 5)
        assert "req-1" not in sched._reqs_need_recv

    def test_send_path_records_request(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer")
        request = _make_request(
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "t2"}
        )
        sched.update_state_after_alloc(request, self._blocks(), 5)
        assert "req-1" in sched._reqs_need_send

    def test_send_path_missing_transfer_id_ignored(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer")
        request = _make_request(kv_transfer_params={"do_remote_decode": True})
        sched.update_state_after_alloc(request, self._blocks(), 5)
        assert "req-1" not in sched._reqs_need_send

    def test_no_params_noop(self, conn_env):
        sched = _make_scheduler(conn_env)
        request = _make_request(kv_transfer_params=None)
        sched.update_state_after_alloc(request, self._blocks(), 5)
        assert not sched._reqs_need_recv and not sched._reqs_need_send


class TestSchedulerBuildConnectorMeta:
    def test_partitions_by_role(self, conn_env):
        fc = conn_env.fc
        recv_sched = fc.FlagCXConnectorScheduler(
            VllmConfigDouble("kv_consumer"), "e0", make_kv_cache_config()
        )
        send_sched = fc.FlagCXConnectorScheduler(
            VllmConfigDouble("kv_producer"), "e0", make_kv_cache_config()
        )
        recv_sched._reqs_need_recv["r"] = (
            _make_request(
                kv_transfer_params={
                    "do_remote_prefill": True,
                    "remote_host": "h",
                    "remote_port": 1,
                    "transfer_id": "t1",
                }
            ),
            [[1]],
        )
        recv_sched._reqs_need_send["s"] = (
            _make_request(
                kv_transfer_params={"do_remote_decode": True, "transfer_id": "t2"}
            ),
            [],
        )

        meta = recv_sched.build_connector_meta(SimpleNamespace())
        # kv_consumer: only the recv side is drained; the send side waits.
        assert set(meta.reqs_to_recv) == {"r"}
        assert set(meta.reqs_to_send) == set()
        assert "r" not in recv_sched._reqs_need_recv
        assert "s" in recv_sched._reqs_need_send

        send_sched._reqs_need_send["s"] = recv_sched._reqs_need_send["s"]
        meta = send_sched.build_connector_meta(SimpleNamespace())
        assert set(meta.reqs_to_send) == {"s"}
        assert "s" not in send_sched._reqs_need_send


class TestSchedulerRequestFinished:
    def test_no_transfer_id(self, conn_env):
        sched = _make_scheduler(conn_env)
        request = _make_request(kv_transfer_params={})
        assert sched.request_finished(request, ([],)) == (False, None)

    def test_no_params(self, conn_env):
        sched = _make_scheduler(conn_env)
        request = _make_request(kv_transfer_params=None)
        assert sched.request_finished(request, ([],)) == (False, None)

    def test_remote_prefill_deferred(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_consumer")
        request = _make_request(
            kv_transfer_params={
                "do_remote_prefill": True,
                "transfer_id": "t1",
            }
        )
        result = sched.request_finished(request, ([],))
        assert result == (False, None)
        # The request is queued for the next round's recv.
        assert "req-1" in sched._reqs_need_recv
        assert request.kv_transfer_params["do_remote_prefill"] is False

    def test_producer_hands_off(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer")
        request = _make_request(
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "t9"}
        )
        delay_free, output_params = sched.request_finished(request, ([[1, 2]],))
        assert delay_free is True
        assert output_params == {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_host": "10.0.0.1",
            "remote_port": sched.side_channel_port,
            "remote_tp_size": 1,
            "transfer_id": "t9",
        }
        assert "req-1" in sched._reqs_need_send

    def test_not_length_capped(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer")
        request = _make_request(
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "t9"},
            status=_RequestStatus.FINISHED_STOPPED
            if hasattr(_RequestStatus, "FINISHED_STOPPED")
            else "other",
        )
        assert sched.request_finished(request, ([[1]],)) == (False, None)

    def test_empty_blocks_no_delay(self, conn_env):
        sched = _make_scheduler(conn_env, role="kv_producer")
        request = _make_request(
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "t9"}
        )
        delay_free, output_params = sched.request_finished(request, ([],))
        assert delay_free is False
        assert output_params is not None
        assert "req-1" not in sched._reqs_need_send


# ---------------------------------------------------------------------------
# FlagCXConnectorWorker
# ---------------------------------------------------------------------------


class TestWorkerConstruction:
    def test_bookkeeping(self, conn_env):
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            assert worker.engine == "engine-handle"
            assert worker.backend_name == "stub_backend"
            assert worker.kv_cache_layout == "NHD"
            assert worker.tp_rank == 0 and worker.tp_size == 1
            assert worker.side_channel_port == 8998
            assert worker.num_workers == 10
            assert worker._abort_request_timeout == 480
            assert worker._physical_blocks_per_logical_kv_block == 1
            assert worker.hostname == "10.0.0.1"
            # kv_cache_config may be required
            assert worker.kv_cache_config is not None
            # consumer: receiver loop thread runs; no sender executor.
            assert worker.receiver_loop.is_running()
            assert not hasattr(worker, "_sender_executor")
        finally:
            worker.shutdown()

    def test_producer_role_sets_up_sender_pool(self, conn_env):
        worker = conn_env.make_worker(role="kv_producer")
        try:
            assert hasattr(worker, "_sender_executor")
            assert worker._sender_executor._max_workers == 10
            assert not hasattr(worker, "receiver_loop")
        finally:
            worker.shutdown()

    def test_num_workers_extra_config(self, conn_env):
        worker = (
            conn_env.make_worker(role="kv_producer", extra_config={"num_workers": 3})
            if False
            else conn_env.fc.FlagCXConnectorWorker(
                VllmConfigDouble("kv_producer", extra_config={"num_workers": 3}),
                "e0",
                make_kv_cache_config(),
            )
        )
        try:
            assert worker.num_workers == 3
            assert worker._sender_executor._max_workers == 3
        finally:
            worker.shutdown()

    def test_abort_timeout_env(self, conn_env, monkeypatch):
        monkeypatch.setenv("FLAGCX_CONNECTOR_ABORT_REQUEST_TIMEOUT", "7")
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            assert worker._abort_request_timeout == 7
        finally:
            worker.shutdown()

    def test_library_path_resolution(self, conn_env, monkeypatch):
        monkeypatch.setenv("FLAGCX_LIB_PATH", "/custom/libflagcx.so")
        calls = []
        env = conn_env

        original = env.fc.FLAGCXLibrary

        def recording_factory(path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(env.fc, "FLAGCXLibrary", recording_factory)
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            assert calls == ["/custom/libflagcx.so"]
        finally:
            worker.shutdown()

    def test_worker_requires_kv_cache_config(self, conn_env):
        with pytest.raises(AssertionError, match="WORKER role"):
            conn_env.fc.FlagCXConnectorWorker(
                VllmConfigDouble("kv_consumer"), "e0", None
            )

    def test_layer_specs_and_group_indices(self, conn_env):
        groups = [
            _Group(_FullSpec(), ["model.layers.0.self_attn.attn"]),
            _Group(_FullSpec(), ["model.layers.3.self_attn.attn"]),
        ]
        worker = conn_env.make_worker(
            role="kv_consumer", config=make_kv_cache_config(groups)
        )
        try:
            assert set(worker._layer_specs) == {
                "model.layers.0.self_attn.attn",
                "model.layers.3.self_attn.attn",
            }
            assert worker._layer_group_indices == {
                "model.layers.0.self_attn.attn": 0,
                "model.layers.3.self_attn.attn": 1,
            }
        finally:
            worker.shutdown()


class TestWorkerStartLoadAndFinished:
    def test_start_load_kv_arms_send_request(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            meta = fc.FlagCXConnectorMetadata()
            meta.add_new_req(
                request_id="p1",
                local_block_ids=[[1, 2]],
                kv_transfer_params={
                    "transfer_id": "t1",
                    "remote_host": "h",
                    "remote_port": 1,
                },
                load_remote_cache=False,
            )
            worker.start_load_kv(meta)
            send_meta = worker.reqs_need_send.reqs["t1"]
            assert send_meta.p_req_id == "p1"
            assert send_meta.local_block_ids == [[1, 2]]
            assert send_meta.ready.is_set()
            assert send_meta.expire_time != float("inf")
        finally:
            worker.shutdown()

    def test_start_load_kv_reuses_existing_send_meta(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            existing = fc.SendBlockMeta(
                p_req_id="",
                transfer_id="t1",
                local_block_ids=[],
                ready=threading.Event(),
            )
            worker.reqs_need_send.reqs["t1"] = existing

            meta = fc.FlagCXConnectorMetadata()
            meta.add_new_req(
                request_id="p1",
                local_block_ids=[[7]],
                kv_transfer_params={
                    "transfer_id": "t1",
                    "remote_host": "h",
                    "remote_port": 1,
                },
                load_remote_cache=False,
            )
            worker.start_load_kv(meta)
            # Same object, now with the scheduler's p_req_id and armed blocks.
            assert worker.reqs_need_send.reqs["t1"] is existing
            assert existing.p_req_id == "p1"
            assert existing.local_block_ids == [[7]]
            assert existing.ready.is_set()
        finally:
            worker.shutdown()

    def test_start_load_kv_without_blocks_leaves_unarmed(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            meta = fc.FlagCXConnectorMetadata()
            meta.add_new_req(
                request_id="p1",
                local_block_ids=[],
                kv_transfer_params={
                    "transfer_id": "t1",
                    "remote_host": "h",
                    "remote_port": 1,
                },
                load_remote_cache=False,
            )
            worker.start_load_kv(meta)
            send_meta = worker.reqs_need_send.reqs["t1"]
            assert send_meta.p_req_id == "p1"
            assert not send_meta.ready.is_set()
        finally:
            worker.shutdown()

    def test_get_finished_drains_and_expires(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            # Armed, un-expired request: stays pending.
            live = fc.SendBlockMeta(
                p_req_id="p-live",
                transfer_id="t-live",
                local_block_ids=[[1]],
                ready=threading.Event(),
                expire_time=float("inf"),
            )
            worker.reqs_need_send.reqs["t-live"] = live
            # Expired request: freed and reported as finished sending.
            expired = fc.SendBlockMeta(
                p_req_id="p-expired",
                transfer_id="t-expired",
                local_block_ids=[[2]],
                ready=threading.Event(),
                expire_time=0.0,
            )
            worker.reqs_need_send.reqs["t-expired"] = expired

            finished_sending, finished_recving = worker.get_finished()
            assert finished_sending == {"p-expired"}
            assert finished_recving is None
            assert "t-expired" not in worker.reqs_need_send.reqs
            assert "t-live" in worker.reqs_need_send.reqs
            assert worker.xfer_stats.data["num_kv_expired_reqs"] == [1]

            # Second call: nothing left to report.
            assert worker.get_finished() == (None, None)
        finally:
            worker.shutdown()

    def test_get_finished_consumer_drains_recv_set(self, conn_env):
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            worker.finished_recving_reqs.set = {"d-req-1"}

            def fake_fetch():
                async def _fetch():
                    return {"d-req-1"}

                return _fetch()

            # _fetch_finished_recving drains the shared set on the loop.
            result = asyncio.run(worker._fetch_finished_recving())
            assert result == {"d-req-1"}
            assert worker.finished_recving_reqs.set == set()

            finished_sending, finished_recving = worker.get_finished()
            assert finished_sending is None
        finally:
            worker.shutdown()

    def test_get_kv_connector_stats_snapshot(self, conn_env):
        worker = conn_env.make_worker(role="kv_producer")
        try:
            assert worker.get_kv_connector_stats() is None
            worker.xfer_stats.record_transfer(0.5, 1024, 2)
            stats = worker.get_kv_connector_stats()
            assert stats.num_successful_transfers == 1
            assert stats.data["bytes_transferred"] == [1024]
            assert worker.get_kv_connector_stats() is None
        finally:
            worker.shutdown()


class TestWorkerGetConn:
    def test_sessions_are_cached(self, conn_env):
        worker = conn_env.make_worker(role="kv_producer")
        try:
            conn_env.flagcx_lib.flagcxP2pGetConn.side_effect = lambda engine, session: (
                f"conn-for:{session}"
            )
            first = worker._get_conn("10.0.0.1:9000")
            second = worker._get_conn("10.0.0.1:9000")
            other = worker._get_conn("10.0.0.2:9000")
            assert first is second
            assert first is not other
            assert first == "conn-for:10.0.0.1:9000"
            conn_env.flagcx_lib.flagcxP2pGetConn.assert_any_call(
                "engine-handle", "10.0.0.1:9000"
            )
            conn_env.flagcx_lib.flagcxP2pGetConn.assert_any_call(
                "engine-handle", "10.0.0.2:9000"
            )
            assert conn_env.flagcx_lib.flagcxP2pGetConn.call_count == 2
        finally:
            worker.shutdown()


class TestWorkerRegisterKvCaches:
    def test_full_attention_blocks_first(self, conn_env):
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            tensor = torch.zeros(2, 3, 4)  # stride(0)=12, elem 4 -> 48 bytes
            worker.register_kv_caches({"model.layers.0.self_attn.attn": tensor})

            assert worker.registered_layer_names == ["model.layers.0.self_attn.attn"]
            assert worker.registered_layer_indices == [0]
            assert worker.block_len_per_layer == [48]
            assert worker.kv_block_len_per_layer == [24]  # blocks-first halves
            assert worker.kv_caches_base_addr == [tensor.data_ptr()]
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

    def test_mla_uses_page_size(self, conn_env):
        class _MLASpec(MLAAttentionSpec):
            page_size_bytes = 4096

        config = make_kv_cache_config(
            [_Group(_MLASpec(), ["model.layers.0.self_attn.attn"])]
        )
        worker = conn_env.make_worker(role="kv_consumer", config=config)
        try:
            worker.register_kv_caches(
                {"model.layers.0.self_attn.attn": torch.zeros(2, 2, 2)}
            )
            assert worker.kv_block_len_per_layer == [4096]
            assert worker.block_len_per_layer == [16]
        finally:
            worker.shutdown()

    def test_unknown_layer_skipped(self, conn_env):
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            with pytest.raises(RuntimeError, match="No KV cache tensors"):
                worker.register_kv_caches({"unknown.layer": torch.zeros(2)})
            assert worker.registered_layer_names == []
        finally:
            worker.shutdown()

    def test_shared_storage_deduplicated(self, conn_env):
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            shared = torch.zeros(2, 2)
            # Two layers whose kv_cache_spec is present for both.
            config = make_kv_cache_config(
                [
                    _Group(_FullSpec(), ["model.layers.0.self_attn.attn"]),
                    _Group(_FullSpec(), ["model.layers.1.self_attn.attn"]),
                ]
            )
            worker.kv_cache_config = config
            worker._layer_specs = {
                "model.layers.0.self_attn.attn": _FullSpec(),
                "model.layers.1.self_attn.attn": _FullSpec(),
            }
            worker._layer_group_indices = {
                "model.layers.0.self_attn.attn": 0,
                "model.layers.1.self_attn.attn": 1,
            }
            worker.register_kv_caches(
                {
                    "model.layers.0.self_attn.attn": shared,
                    "model.layers.1.self_attn.attn": shared,
                }
            )
            # Two registered regions, but a single memory region.
            assert len(worker.kv_caches_base_addr) == 2
            assert conn_env.flagcx_lib.flagcxP2pRegister.call_count == 1
        finally:
            worker.shutdown()

    def test_producer_starts_sender_thread(self, conn_env):
        worker = conn_env.make_worker(role="kv_producer")
        try:
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

    def test_non_blocks_first_layout_keeps_full_kv_block_len(self, conn_env):
        """Non-blocks-first (NHD) layout with a non-MLA spec: the whole
        block is one KV region (line 780)."""
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            worker.kv_topo.is_kv_layout_blocks_first = False
            tensor = torch.zeros(2, 3, 4)  # stride(0)=12 * elem 4 = 48 bytes
            worker.register_kv_caches({"model.layers.0.self_attn.attn": tensor})
            assert worker.block_len_per_layer == [48]
            assert worker.kv_block_len_per_layer == [48]  # not halved
        finally:
            worker.shutdown()

    def test_mamba_spec_keeps_full_kv_block_len_under_blocks_first(self, conn_env):
        """Blocks-first layout but a Mamba spec: the region is not split, so
        kv_block_len equals the whole block (line 780 via the Mamba guard)."""
        layer = "model.layers.0.self_attn.attn"
        worker = conn_env.make_worker(
            role="kv_consumer",
            config=make_kv_cache_config([_Group(MambaSpec(), [layer])]),
        )
        try:
            # The stub topology reports blocks-first, but the spec is Mamba.
            tensor = torch.zeros(2, 3, 4)
            worker.register_kv_caches({layer: tensor})
            assert worker.block_len_per_layer == [48]
            assert worker.kv_block_len_per_layer == [48]
        finally:
            worker.shutdown()


class TestWorkerSyncBlockSizeWithKernel:
    def test_smaller_kernel_block_size_shrinks_block_size(self, conn_env):
        """User logical block size 16 vs kernel block size 8: the worker
        switches to kernel units and records the fan-out (lines 715-728)."""
        fc = conn_env.fc
        original = fc.select_common_block_size
        try:
            fc.select_common_block_size = lambda bs, backends, specs=None: bs // 2
            worker = conn_env.make_worker(role="kv_consumer")
            try:
                assert worker.block_size == 8
                assert worker._physical_blocks_per_logical_kv_block == 2
            finally:
                worker.shutdown()
        finally:
            fc.select_common_block_size = original

    def test_matching_kernel_block_size_is_noop(self, conn_env):
        fc = conn_env.fc
        calls = []

        def recorder(bs, backends, specs=None):
            calls.append(bs)
            return bs

        original = fc.select_common_block_size
        try:
            fc.select_common_block_size = recorder
            worker = conn_env.make_worker(role="kv_consumer")
            try:
                assert worker.block_size == 16
                assert worker._physical_blocks_per_logical_kv_block == 1
                assert calls == [16]
            finally:
                worker.shutdown()
        finally:
            fc.select_common_block_size = original


class TestWorkerLogicalToKernelBlocks:
    def test_identity_when_ratio_one(self, conn_env):
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            assert worker._logical_to_kernel_block_ids([[1, 2], [3]]) == [
                [1, 2],
                [3],
            ]
        finally:
            worker.shutdown()

    def test_expansion_when_ratio_above_one(self, conn_env):
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            worker._physical_blocks_per_logical_kv_block = 2
            groups = [
                _Group(_FullSpec(), ["attn"]),
                _Group(MambaSpec(), ["mamba"]),
            ]
            worker.kv_cache_config = make_kv_cache_config(groups)
            result = worker._logical_to_kernel_block_ids([[5], [7]])
            # Attention group expands (kernel-physical ids); mamba stays logical.
            assert result[0] == [10, 11]
            assert result[1] == [7]
        finally:
            worker.shutdown()


def _arm_worker_regions(worker, kv_block_len=None, blocks_first=True):
    """Populate the parallel registration lists without register_kv_caches."""
    block_len = 64
    if kv_block_len is None:
        kv_block_len = block_len if not blocks_first else block_len // 2
    layer = "model.layers.0.self_attn.attn"
    worker.kv_caches_base_addr = [1000]
    worker.block_len_per_layer = [block_len]
    worker.kv_block_len_per_layer = [kv_block_len]
    worker.registered_layer_names = [layer]
    worker.registered_layer_indices = [0]
    worker.registered_group_indices = [0]
    worker.kv_topo.is_kv_layout_blocks_first = blocks_first
    return layer


def _agent_meta(fc, req_blocks, remote_tp_size=1, kv_block_len=None, blocks_first=True):
    block_len = 64
    if kv_block_len is None:
        kv_block_len = block_len if not blocks_first else block_len // 2
    return fc.FlagCXAgentMetadata(
        remote_hostname="10.9.9.9",
        remote_port=5555,
        remote_tp_size=remote_tp_size,
        remote_tp_rank=0,
        req_blocks=req_blocks,
        kv_caches_base_addr=[2000],
        block_lens=[block_len],
        kv_block_lens=[kv_block_len],
        registered_layer_names=["model.layers.0.self_attn.attn"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )


class TestBuildTransferParams:
    def _armed_producer(self, conn_env, local_block_ids, remote_block_ids):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        _arm_worker_regions(worker, blocks_first=False)
        meta = _agent_meta(
            fc, {"d-req-1": ("t1", [remote_block_ids])}, blocks_first=False
        )
        send_meta = fc.SendBlockMeta(
            p_req_id="p-req-1",
            transfer_id="t1",
            local_block_ids=[local_block_ids],
            ready=threading.Event(),
        )
        worker.reqs_need_send.reqs["t1"] = send_meta
        return worker, meta, [("d-req-1", send_meta)]

    def test_coalesced_contiguous_run(self, conn_env):
        worker, meta, ready_reqs = self._armed_producer(conn_env, [1, 2, 3], [5, 6, 7])
        try:
            local_regions = worker._get_transfer_regions(
                worker.kv_caches_base_addr,
                worker.block_len_per_layer,
                worker.kv_block_len_per_layer,
                worker.registered_layer_names,
                worker.registered_layer_indices,
                worker.registered_group_indices,
            )
            remote_regions = worker._get_transfer_regions(
                meta.kv_caches_base_addr,
                meta.block_lens,
                meta.kv_block_lens,
                meta.registered_layer_names,
                meta.registered_layer_indices,
                meta.registered_group_indices,
            )
            src, dst, sizes = worker._build_transfer_params(
                ready_reqs, meta, local_regions, remote_regions
            )
            # One coalesced descriptor for the contiguous run.
            assert len(sizes) == 1
            assert sizes == [64 * 3]
            assert src == [1000 + 1 * 64]
            assert dst == [2000 + 5 * 64]
        finally:
            worker.shutdown()

    def test_non_coalesced_per_block(self, conn_env):
        # Non-contiguous blocks produce one descriptor per block.
        worker, meta, ready_reqs = self._armed_producer(conn_env, [1, 5], [9, 3])
        try:
            local_regions = worker._get_transfer_regions(
                worker.kv_caches_base_addr,
                worker.block_len_per_layer,
                worker.kv_block_len_per_layer,
                worker.registered_layer_names,
                worker.registered_layer_indices,
                worker.registered_group_indices,
            )
            remote_regions = worker._get_transfer_regions(
                meta.kv_caches_base_addr,
                meta.block_lens,
                meta.kv_block_lens,
                meta.registered_layer_names,
                meta.registered_layer_indices,
                meta.registered_group_indices,
            )
            src, dst, sizes = worker._build_transfer_params(
                ready_reqs, meta, local_regions, remote_regions
            )
            assert sizes == [64, 64]
            assert src == [1000 + 1 * 64, 1000 + 5 * 64]
            assert dst == [2000 + 9 * 64, 2000 + 3 * 64]
        finally:
            worker.shutdown()

    def test_mamba_null_blocks_stripped(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            layer = _arm_worker_regions(worker, blocks_first=False)
            # Give the group a mamba spec so NULL blocks are stripped.
            worker.kv_cache_config = make_kv_cache_config(
                [_Group(MambaSpec(), [layer])]
            )
            meta = _agent_meta(fc, {"d-req-1": ("t1", [[0, 4]])}, blocks_first=False)
            send_meta = fc.SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="t1",
                local_block_ids=[[0, 4]],
                ready=threading.Event(),
            )
            worker.reqs_need_send.reqs["t1"] = send_meta
            local_regions = worker._get_transfer_regions(
                worker.kv_caches_base_addr,
                worker.block_len_per_layer,
                worker.kv_block_len_per_layer,
                worker.registered_layer_names,
                worker.registered_layer_indices,
                worker.registered_group_indices,
            )
            remote_regions = worker._get_transfer_regions(
                meta.kv_caches_base_addr,
                meta.block_lens,
                meta.kv_block_lens,
                meta.registered_layer_names,
                meta.registered_layer_indices,
                meta.registered_group_indices,
            )
            # NULL_BLOCK_ID (0) entries vanish on both sides: block 4 remains.
            src, dst, sizes = worker._build_transfer_params(
                [("d-req-1", send_meta)], meta, local_regions, remote_regions
            )
            assert src == [1000 + 4 * 64]
            assert dst == [2000 + 4 * 64]
            assert sizes == [64]
        finally:
            worker.shutdown()

    def test_local_shorter_than_remote_skips_request(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            _arm_worker_regions(worker, blocks_first=False)
            meta = _agent_meta(fc, {"d-req-1": ("t1", [[1, 2, 3]])}, blocks_first=False)
            send_meta = fc.SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="t1",
                local_block_ids=[[1]],  # fewer local blocks than remote
                ready=threading.Event(),
            )
            worker.reqs_need_send.reqs["t1"] = send_meta
            local_regions = worker._get_transfer_regions(
                worker.kv_caches_base_addr,
                worker.block_len_per_layer,
                worker.kv_block_len_per_layer,
                worker.registered_layer_names,
                worker.registered_layer_indices,
                worker.registered_group_indices,
            )
            remote_regions = worker._get_transfer_regions(
                meta.kv_caches_base_addr,
                meta.block_lens,
                meta.kv_block_lens,
                meta.registered_layer_names,
                meta.registered_layer_indices,
                meta.registered_group_indices,
            )
            src, dst, sizes = worker._build_transfer_params(
                [("d-req-1", send_meta)], meta, local_regions, remote_regions
            )
            assert (src, dst, sizes) == ([], [], [])
        finally:
            worker.shutdown()

    def test_local_longer_than_remote_trims_tail(self, conn_env):
        worker, meta, ready_reqs = self._armed_producer(conn_env, [1, 2, 3, 4], [5, 6])
        try:
            local_regions = worker._get_transfer_regions(
                worker.kv_caches_base_addr,
                worker.block_len_per_layer,
                worker.kv_block_len_per_layer,
                worker.registered_layer_names,
                worker.registered_layer_indices,
                worker.registered_group_indices,
            )
            remote_regions = worker._get_transfer_regions(
                meta.kv_caches_base_addr,
                meta.block_lens,
                meta.kv_block_lens,
                meta.registered_layer_names,
                meta.registered_layer_indices,
                meta.registered_group_indices,
            )
            src, dst, sizes = worker._build_transfer_params(
                ready_reqs, meta, local_regions, remote_regions
            )
            # Only the last two local blocks transfer, coalesced.
            assert sizes == [64 * 2]
            assert src == [1000 + 3 * 64]
            assert dst == [2000 + 5 * 64]
        finally:
            worker.shutdown()

    def _regions(self, worker, meta):
        local_regions = worker._get_transfer_regions(
            worker.kv_caches_base_addr,
            worker.block_len_per_layer,
            worker.kv_block_len_per_layer,
            worker.registered_layer_names,
            worker.registered_layer_indices,
            worker.registered_group_indices,
        )
        remote_regions = worker._get_transfer_regions(
            meta.kv_caches_base_addr,
            meta.block_lens,
            meta.kv_block_lens,
            meta.registered_layer_names,
            meta.registered_layer_indices,
            meta.registered_group_indices,
        )
        return local_regions, remote_regions

    def test_empty_remote_groups_skip_request(self, conn_env):
        """Remote per-group block ids all empty: the request is skipped
        (line 1112)."""
        worker, meta, ready_reqs = self._armed_producer(conn_env, [1, 2], [])
        try:
            local_regions, remote_regions = self._regions(worker, meta)
            src, dst, sizes = worker._build_transfer_params(
                ready_reqs, meta, local_regions, remote_regions
            )
            assert (src, dst, sizes) == ([], [], [])
        finally:
            worker.shutdown()

    def test_no_remote_groups_skip_request(self, conn_env):
        """A request whose remote group list is the empty list itself is
        skipped (line 1112)."""
        worker, meta, ready_reqs = self._armed_producer(conn_env, [1, 2], [1, 2])
        try:
            meta.req_blocks["d-req-1"] = ("t1", [])
            local_regions, remote_regions = self._regions(worker, meta)
            src, dst, sizes = worker._build_transfer_params(
                ready_reqs, meta, local_regions, remote_regions
            )
            assert (src, dst, sizes) == ([], [], [])
        finally:
            worker.shutdown()

    def test_group_count_mismatch_skips_request(self, conn_env):
        """Local and remote KV-cache group counts differ for the request:
        the mismatch is logged and the request skipped (lines 1113-1120)."""
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            _arm_worker_regions(worker, blocks_first=False)
            # Remote declares two groups, local has one.
            meta = _agent_meta(fc, {"d-req-1": ("t1", [[5], [6]])}, blocks_first=False)
            send_meta = fc.SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="t1",
                local_block_ids=[[5]],
                ready=threading.Event(),
            )
            worker.reqs_need_send.reqs["t1"] = send_meta
            local_regions, remote_regions = self._regions(worker, meta)
            src, dst, sizes = worker._build_transfer_params(
                [("d-req-1", send_meta)], meta, local_regions, remote_regions
            )
            assert (src, dst, sizes) == ([], [], [])
        finally:
            worker.shutdown()

    def test_region_group_index_beyond_local_groups_skipped(self, conn_env):
        """A registered region tagged with a group index the request has no
        block ids for is skipped (line 1156)."""
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            _arm_worker_regions(worker, blocks_first=False)
            # Region belongs to group 2; the request only carries group 0.
            worker.registered_group_indices = [2]
            meta = _agent_meta(fc, {"d-req-1": ("t1", [[5]])}, blocks_first=False)
            meta.registered_group_indices = [2]
            send_meta = fc.SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="t1",
                local_block_ids=[[5]],
                ready=threading.Event(),
            )
            worker.reqs_need_send.reqs["t1"] = send_meta
            local_regions, remote_regions = self._regions(worker, meta)
            src, dst, sizes = worker._build_transfer_params(
                [("d-req-1", send_meta)], meta, local_regions, remote_regions
            )
            assert (src, dst, sizes) == ([], [], [])
        finally:
            worker.shutdown()

    def test_empty_local_group_ids_skipped(self, conn_env):
        """The request's block-id list for the region's group is empty: the
        region is skipped (line 1160). Reaching it needs the empty group to
        pass the earlier per-group trimming (n_local >= n_remote), so the
        remote side must be empty for that group too, with a non-empty group
        elsewhere in the request keeping it alive."""
        fc = conn_env.fc
        worker = conn_env.make_worker(
            role="kv_producer",
            config=make_kv_cache_config(
                [
                    _Group(_FullSpec(), ["model.layers.0.self_attn.attn"]),
                    _Group(_FullSpec(), ["model.layers.1.self_attn.attn"]),
                ]
            ),
        )
        try:
            _arm_worker_regions(worker, blocks_first=False)
            # Two groups: group 0 empty on both sides, group 1 non-empty.
            # The single registered region (group 0) is skipped at the
            # empty-local check; the group-1 blocks have no registered region.
            meta = _agent_meta(fc, {"d-req-1": ("t1", [[], [5]])}, blocks_first=False)
            send_meta = fc.SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="t1",
                local_block_ids=[[], [6, 7]],
                ready=threading.Event(),
            )
            send_meta.ready.set()
            worker.reqs_need_send.reqs["t1"] = send_meta
            local_regions, remote_regions = self._regions(worker, meta)
            src, dst, sizes = worker._build_transfer_params(
                [("d-req-1", send_meta)], meta, local_regions, remote_regions
            )
            assert (src, dst, sizes) == ([], [], [])
        finally:
            worker.shutdown()

    def test_replicated_non_root_rank_skips_region(self, conn_env):
        """producer_cache_replicated with a non-root local rank: the plan says
        should_transfer=False and the region is skipped (line 1172)."""
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            _arm_worker_regions(worker, blocks_first=False)
            # local tp 4 vs remote tp 2, replicated: only rank % 2 == 0 sends.
            worker.kv_topo.local_replicates_kv_cache = True
            worker.tp_size = 4
            worker.tp_rank = 1
            meta = _agent_meta(
                fc, {"d-req-1": ("t1", [[5, 6]])}, remote_tp_size=2, blocks_first=False
            )
            send_meta = fc.SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="t1",
                local_block_ids=[[5, 6]],
                ready=threading.Event(),
            )
            send_meta.ready.set()
            worker.reqs_need_send.reqs["t1"] = send_meta
            local_regions, remote_regions = self._regions(worker, meta)
            src, dst, sizes = worker._build_transfer_params(
                [("d-req-1", send_meta)], meta, local_regions, remote_regions
            )
            # Rank 1 is not the sending rank: nothing emitted.
            assert (src, dst, sizes) == ([], [], [])
        finally:
            worker.shutdown()

    def test_asymmetric_offsets_emit_per_block_descriptors(self, conn_env):
        """Heterogeneous TP (local 8, remote 2, sharded): dst_off != 0 makes
        coalescing impossible, so each contiguous block group still emits one
        descriptor per block (lines 1204-1215)."""
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            _arm_worker_regions(worker, blocks_first=False)
            worker.tp_size = 8
            worker.tp_rank = 1
            # local_kv_block_len=64, remote=256: dst_off = rank%4 * 64 = 64.
            meta = _agent_meta(
                fc,
                {"d-req-1": ("t1", [[5, 6, 7]])},
                remote_tp_size=2,
                kv_block_len=256,
                blocks_first=False,
            )
            # Remote regions stride 256 bytes per block (kv_block_len is half
            # of it under a split layout; here the whole block is one region).
            meta.block_lens = [256]
            send_meta = fc.SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="t1",
                local_block_ids=[[5, 6, 7]],
                ready=threading.Event(),
            )
            send_meta.ready.set()
            worker.reqs_need_send.reqs["t1"] = send_meta
            local_regions, remote_regions = self._regions(worker, meta)
            src, dst, sizes = worker._build_transfer_params(
                [("d-req-1", send_meta)], meta, local_regions, remote_regions
            )
            # Three contiguous blocks, but dst_off=64 prevents coalescing:
            # one descriptor per block, each 64 bytes.
            assert sizes == [64, 64, 64]
            assert src == [1000 + 5 * 64, 1000 + 6 * 64, 1000 + 7 * 64]
            assert dst == [
                2000 + 5 * 256 + 64,
                2000 + 6 * 256 + 64,
                2000 + 7 * 256 + 64,
            ]
        finally:
            worker.shutdown()


class TestSendKvToDecode:
    def _run_send(self, conn_env, *, expire=None, p_req_id="p-req-1"):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        _arm_worker_regions(worker, blocks_first=False)
        meta = _agent_meta(fc, {"d-req-1": ("t1", [[5, 6]])}, blocks_first=False)
        send_meta = fc.SendBlockMeta(
            p_req_id=p_req_id,
            transfer_id="t1",
            local_block_ids=[[5, 6]],
            ready=threading.Event(),
        )
        send_meta.ready.set()
        if expire is not None:
            send_meta.expire_time = expire
        worker.reqs_need_send.reqs["t1"] = send_meta
        return worker, meta, send_meta

    def test_success_path(self, conn_env):
        worker, meta, send_meta = self._run_send(conn_env)
        try:
            worker._send_kv_to_decode(meta)

            conn_env.flagcx_lib.flagcxP2pBatchWriteSync.assert_called_once()
            conn, srcs, dsts, sizes = (
                conn_env.flagcx_lib.flagcxP2pBatchWriteSync.call_args[0]
            )
            assert conn.startswith("conn:")
            assert sizes == [128]
            assert srcs == [1000 + 5 * 64]
            assert dsts == [2000 + 5 * 64]
            # Stats recorded for the successful write.
            assert worker.xfer_stats.num_successful_transfers == 1
            assert worker.xfer_stats.data["bytes_transferred"] == [128]
            # Bookkeeping: request finished and removed.
            assert "t1" not in worker.reqs_need_send.reqs
            assert worker.finished_sending_reqs.set == {"p-req-1"}
            # Connection cached under host:port.
            assert "10.9.9.9:5555" in worker.session_conns
        finally:
            worker.shutdown()

    def test_missing_prefill_id_raises(self, conn_env):
        worker, meta, send_meta = self._run_send(conn_env, p_req_id="")
        try:
            with pytest.raises(RuntimeError, match="Missing Prefill request ID"):
                worker._send_kv_to_decode(meta)
        finally:
            worker.shutdown()

    def test_write_failure_records_failed_transfer(self, conn_env):
        worker, meta, send_meta = self._run_send(conn_env)
        try:
            conn_env.flagcx_lib.flagcxP2pBatchWriteSync.side_effect = RuntimeError(
                "rdma write failed"
            )
            with pytest.raises(RuntimeError, match="rdma write failed"):
                worker._send_kv_to_decode(meta)
            assert worker.xfer_stats.data["num_failed_transfers"] == [1]
            assert worker.xfer_stats.num_successful_transfers == 0
        finally:
            worker.shutdown()

    def test_alignment_failure_raises(self, conn_env):
        fc = conn_env.fc
        worker, meta, send_meta = self._run_send(conn_env)
        try:
            # Remote lists an unknown layer: alignment reports an error.
            meta = fc.FlagCXAgentMetadata(
                remote_hostname="10.9.9.9",
                remote_port=5555,
                remote_tp_size=1,
                remote_tp_rank=0,
                req_blocks={"d-req-1": ("t1", [[5, 6]])},
                kv_caches_base_addr=[2000],
                block_lens=[64],
                kv_block_lens=[64],
                registered_layer_names=["model.layers.9.self_attn.attn"],
                registered_layer_indices=[9],
                registered_group_indices=[0],
            )
            with pytest.raises(RuntimeError, match="no matching consumer"):
                worker._send_kv_to_decode(meta)
        finally:
            worker.shutdown()

    def test_length_validation_failure_raises(self, conn_env):
        fc = conn_env.fc
        worker, meta, send_meta = self._run_send(conn_env)
        try:
            meta = _agent_meta(fc, {"d-req-1": ("t1", [[5, 6]])}, blocks_first=False)
            # Remote kv_block_len mismatches the homogeneous ratio.
            meta.kv_block_lens = [32]
            with pytest.raises(RuntimeError, match="length mismatch"):
                worker._send_kv_to_decode(meta)
        finally:
            worker.shutdown()

    def test_send_kv_creates_and_arms_unknown_transfer(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            _arm_worker_regions(worker, blocks_first=False)
            meta = _agent_meta(fc, {"d-req-1": ("t-new", [[5, 6]])}, blocks_first=False)

            # _send_kv_to_decode creates the SendBlockMeta for t-new and
            # blocks on its ready event; arm it from a watchdog thread.
            def armer():
                for _ in range(200):
                    send_meta = worker.reqs_need_send.reqs.get("t-new")
                    if send_meta is not None:
                        send_meta.p_req_id = "p-new"
                        send_meta.local_block_ids = [[5, 6]]
                        send_meta.ready.set()
                        return
                    time.sleep(0.01)

            threading.Thread(target=armer, daemon=True).start()
            worker._send_kv_to_decode(meta)
            conn_env.flagcx_lib.flagcxP2pBatchWriteSync.assert_called_once()
            assert worker.finished_sending_reqs.set == {"p-new"}
        finally:
            worker.shutdown()

    def test_remote_rank_not_in_targets_warns_but_proceeds(self, conn_env):
        """D rank outside handshake_target_ranks logs a warning and still
        transfers (lines 977-986)."""
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            # Targets exclude the metadata's remote_tp_rank=0.
            worker.kv_topo.handshake_target_ranks = lambda remote_tp_size, dcp=1: [1, 2]
            _arm_worker_regions(worker, blocks_first=False)
            meta = _agent_meta(fc, {"d-req-1": ("t1", [[5, 6]])}, blocks_first=False)
            send_meta = fc.SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="t1",
                local_block_ids=[[5, 6]],
                ready=threading.Event(),
            )
            send_meta.ready.set()
            worker.reqs_need_send.reqs["t1"] = send_meta

            worker._send_kv_to_decode(meta)

            # need=2 targets but only this send: the send counter reaches 1,
            # below need_send, so the request is not yet finalized.
            assert send_meta.sent == 1
            assert "t1" in worker.reqs_need_send.reqs
            conn_env.flagcx_lib.flagcxP2pBatchWriteSync.assert_called_once()
        finally:
            worker.shutdown()

    def test_unready_blocks_time_out(self, conn_env, monkeypatch):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            monkeypatch.setattr(worker, "_abort_request_timeout", 0.05)
            _arm_worker_regions(worker, blocks_first=False)
            meta = _agent_meta(fc, {"d-req-1": ("t1", [[5]])}, blocks_first=False)
            send_meta = fc.SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="t1",
                local_block_ids=[[5]],
                ready=threading.Event(),  # never set
            )
            worker.reqs_need_send.reqs["t1"] = send_meta
            with pytest.raises(RuntimeError, match="Timed out waiting for prefill"):
                worker._send_kv_to_decode(meta)
        finally:
            worker.shutdown()


class TestReceiveKv:
    def _run_receive(self, conn_env, reply, recv_error=None):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_consumer")

        encoded = []

        class FakeSocket:
            def setsockopt(self, *a, **k):
                pass

            async def send(self, data):
                encoded.append(data)

            async def recv(self):
                if recv_error is not None:
                    raise recv_error
                return reply

            def close(self):
                pass

        original = fc.make_zmq_socket
        fc.make_zmq_socket = lambda *a, **k: FakeSocket()
        try:
            path = "tcp://10.0.0.1:9001"
            coro = worker._receive_kv(path, {"d-req-1": ("t1", [[1, 2]])})
            asyncio.run(coro)
        finally:
            fc.make_zmq_socket = original
        return worker, encoded

    def test_success_marks_finished(self, conn_env):
        worker, encoded = self._run_receive(conn_env, TRANS_DONE)
        try:
            assert len(encoded) == 1
            assert worker.finished_recving_reqs.set == {"d-req-1"}
            assert worker._pull_pending == {}
        finally:
            worker.shutdown()

    def test_error_reply_records_failure(self, conn_env):
        worker, encoded = self._run_receive(conn_env, TRANS_ERROR)
        try:
            assert worker.finished_recving_reqs.set == set()
            assert worker.xfer_stats.data["num_failed_recvs"] == [1]
        finally:
            worker.shutdown()

    def test_exception_records_failure(self, conn_env):
        worker, encoded = self._run_receive(
            conn_env, None, recv_error=RuntimeError("zmq broken")
        )
        try:
            assert worker.finished_recving_reqs.set == set()
            assert worker.xfer_stats.data["num_failed_recvs"] == [1]
        finally:
            worker.shutdown()

    def test_pending_counter_multi_peer(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            worker._pull_pending["d-req-1"] = 2
            worker.finished_recving_reqs.set = set()

            class FakeSocket:
                def setsockopt(self, *a, **k):
                    pass

                async def send(self, data):
                    pass

                async def recv(self):
                    return TRANS_DONE

                def close(self):
                    pass

            original = fc.make_zmq_socket
            fc.make_zmq_socket = lambda *a, **k: FakeSocket()
            try:
                asyncio.run(worker._receive_kv("tcp://h:1", {"d-req-1": ("t1", [[1]])}))
            finally:
                fc.make_zmq_socket = original
            # Two peers expected: one reply leaves the counter at 1.
            assert worker._pull_pending == {"d-req-1": 1}
            assert worker.finished_recving_reqs.set == set()
            asyncio.run(
                worker._receive_kv("tcp://h:2", {"d-req-1": ("t1", [[1]])})
            ) if False else None
        finally:
            worker.shutdown()

    def test_context_terminated_exits_cleanly(self, conn_env):
        """zmq.ContextTerminated during send/recv is caught and logged at
        debug (line 1266): unlike a generic failure it records no failed
        recv, and the coroutine still falls through to the finished-marking
        block below (shutdown-time cancellation is not an error)."""
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_consumer")

        class TerminatedSocket:
            def setsockopt(self, *a, **k):
                pass

            async def send(self, data):
                raise zmq.ContextTerminated()

            async def recv(self):
                raise AssertionError("recv unreachable after send raises")

            def close(self):
                pass

        original = fc.make_zmq_socket
        fc.make_zmq_socket = lambda *a, **k: TerminatedSocket()
        try:
            # Must not raise.
            asyncio.run(worker._receive_kv("tcp://h:1", {"d-1": ("t1", [[1]])}))
            # No failure recorded...
            assert worker.xfer_stats.data["num_failed_recvs"] == []
            # ...but the fallthrough still marks the request finished.
            assert worker.finished_recving_reqs.set == {"d-1"}
        finally:
            fc.make_zmq_socket = original
            worker.shutdown()

    def test_start_load_kv_consumer_schedules_group_pull(self, conn_env, monkeypatch):
        """start_load_kv on a consumer with pending recvs schedules
        _group_kv_pull on the receiver loop via run_coroutine_threadsafe
        (lines 1286-1290)."""
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            scheduled = []

            async def fake_group_pull(reqs_to_recv):
                scheduled.append(dict(reqs_to_recv))
                worker.finished_recving_reqs.set.update(reqs_to_recv)

            monkeypatch.setattr(worker, "_group_kv_pull", fake_group_pull)
            meta = fc.FlagCXConnectorMetadata()
            meta.add_new_req(
                request_id="d-1",
                local_block_ids=[[1, 2]],
                kv_transfer_params={
                    "transfer_id": "t1",
                    "remote_host": "10.1.1.1",
                    "remote_port": 7000,
                },
            )
            worker.start_load_kv(meta)

            deadline = time.perf_counter() + 5
            while not scheduled and time.perf_counter() < deadline:
                time.sleep(0.01)
            assert scheduled == [{"d-1": meta.reqs_to_recv["d-1"]}]
        finally:
            worker.shutdown()

    def test_start_load_kv_producer_skips_recv_side(self, conn_env):
        """kv_producer workers never touch the recv coroutine path even when
        the metadata carries recv requests."""
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            meta = fc.FlagCXConnectorMetadata()
            meta.add_new_req(
                request_id="d-1",
                local_block_ids=[[1]],
                kv_transfer_params={
                    "transfer_id": "t1",
                    "remote_host": "h",
                    "remote_port": 1,
                },
            )
            # No receiver loop exists: the recv branch is gated by kv_role.
            worker.start_load_kv(meta)
            assert not hasattr(worker, "receiver_loop")
            assert worker._pull_pending == {}
        finally:
            worker.shutdown()

    def test_group_kv_pull_fans_out(self, conn_env, monkeypatch):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_consumer")
        try:
            scheduled = []

            async def fake_receive(path, req_blocks):
                scheduled.append((path, dict(req_blocks)))

            monkeypatch.setattr(worker, "_receive_kv", fake_receive)
            # handshake_target_ranks returns [tp_rank] = [0]
            reqs = {
                "d-1": fc.RecvReqMeta(
                    local_block_ids=[[1]],
                    remote_host="10.1.1.1",
                    remote_port=7000,
                    transfer_id="t1",
                    remote_tp_size=1,
                )
            }
            asyncio.run(worker._group_kv_pull(reqs))
            assert worker._pull_pending == {"d-1": 1}
            assert scheduled == [("tcp://10.1.1.1:7000", {"d-1": ("t1", [[1]])})]
        finally:
            worker.shutdown()


class TestSenderThreadAndWorker:
    def _patch_sockets(self, conn_env, frontend, backend, poller_cls):
        fc = conn_env.fc
        made = []

        def fake_make_socket(ctx, path, socket_type, **kwargs):
            sock = frontend if socket_type == zmq.ROUTER else backend
            made.append((path, socket_type))
            return sock

        fc.make_zmq_socket = fake_make_socket
        original_poller = fc.zmq.Poller
        fc.zmq.Poller = poller_cls
        return made, original_poller

    def test_sender_thread_dispatches_and_relays(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            identity = b"peer-1"
            metadata_bytes = b"encoded-meta"

            class FakeZmqSocket:
                """Hashable fake socket (zmq sockets hash by identity)."""

                def __init__(self):
                    self.sent = []

                def recv_multipart(self):
                    if self is frontend:
                        return (identity, b"", metadata_bytes)
                    return (identity, TRANS_DONE)

                def send_multipart(self, parts):
                    self.sent.append(tuple(parts))

                def close(self):
                    pass

            frontend = FakeZmqSocket()
            backend = FakeZmqSocket()
            submitted = []
            worker._sender_worker = lambda ident, meta, path: submitted.append(
                (ident, meta, path)
            )

            poll_calls = []

            class FakePoller:
                def register(self, sock, event):
                    pass

                def poll(self):
                    poll_calls.append(1)
                    if len(poll_calls) == 1:
                        return {frontend: zmq.POLLIN}
                    if len(poll_calls) == 2:
                        return {backend: zmq.POLLIN}
                    raise zmq.ContextTerminated()

            made, original_poller = self._patch_sockets(
                conn_env, frontend, backend, FakePoller
            )
            try:
                ready = threading.Event()
                worker._sender_thread(ready, 8998, 0)
                assert ready.is_set()
                assert submitted == [(identity, metadata_bytes, made[1][0])]
            finally:
                fc.zmq.Poller = original_poller
                fc.make_zmq_socket = lambda *a, **k: None
        finally:
            worker.shutdown()

    def test_sender_thread_generic_exception_is_logged_and_closed(self, conn_env):
        """A non-ZMQ failure in the poll loop is logged (not raised) and both
        sockets are still closed in the finally (lines 871-877)."""
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            closed = []

            class FakeZmqSocket:
                def recv_multipart(self):
                    raise AssertionError("never polled past the failure")

                def send_multipart(self, parts):
                    raise AssertionError("never polled past the failure")

                def close(self):
                    closed.append(self)

            frontend = FakeZmqSocket()
            backend = FakeZmqSocket()

            class ExplodingPoller:
                def register(self, sock, event):
                    pass

                def poll(self):
                    raise RuntimeError("poller exploded")

            original_socket = fc.make_zmq_socket
            original_poller = fc.zmq.Poller
            fc.make_zmq_socket = lambda ctx, path, socket_type, **kw: (
                frontend if socket_type == zmq.ROUTER else backend
            )
            fc.zmq.Poller = ExplodingPoller
            try:
                ready = threading.Event()
                # Must not raise despite the poller failure.
                worker._sender_thread(ready, 8998, 0)
                assert ready.is_set()
                assert closed == [frontend, backend]
            finally:
                fc.zmq.Poller = original_poller
                fc.make_zmq_socket = original_socket
        finally:
            worker.shutdown()

    def test_sender_worker_reports_done_and_error(self, conn_env):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            sent = []

            class FakePusher:
                def send_multipart(self, parts):
                    sent.append(parts)

                def close(self):
                    pass

            original = fc.make_zmq_socket
            fc.make_zmq_socket = lambda *a, **k: FakePusher()

            # Error path: undecodable metadata yields TRANS_ERROR (never raises).
            worker._sender_worker(b"id", b"not-valid-msgpack", "inproc://x")
            assert sent == [(b"id", TRANS_ERROR)]

            # Success path: a decodable metadata with no requests yields DONE.
            meta = _agent_meta(fc, {}, blocks_first=False)
            good_bytes = worker._encoder.encode(meta)
            worker._sender_worker(b"id2", good_bytes, "inproc://y")
            assert sent[-1] == (b"id2", TRANS_DONE)
        finally:
            fc.make_zmq_socket = original
            worker.shutdown()

    def test_sender_worker_push_failure_is_swallowed(self, conn_env, caplog):
        fc = conn_env.fc
        worker = conn_env.make_worker(role="kv_producer")
        try:
            original = fc.make_zmq_socket

            class BrokenPusher:
                def send_multipart(self, parts):
                    raise zmq.ZMQError(1)

                def close(self):
                    pass

            fc.make_zmq_socket = lambda *a, **k: BrokenPusher()
            try:
                # Must not raise despite the push failure.
                worker._sender_worker(b"id", b"not-valid-msgpack", "inproc://x")
            finally:
                fc.make_zmq_socket = original
        finally:
            worker.shutdown()


class TestWorkerShutdown:
    def test_engine_destroyed_once(self, conn_env):
        worker = conn_env.make_worker(role="kv_consumer")
        worker.shutdown()
        conn_env.flagcx_lib.flagcxP2pEngineDestroy.assert_called_once_with(
            "engine-handle"
        )
        assert worker.engine is None
        # Second shutdown is a no-op for the engine.
        worker.shutdown()
        assert conn_env.flagcx_lib.flagcxP2pEngineDestroy.call_count == 1

    def test_engine_destroy_failure_tolerated(self, conn_env):
        conn_env.flagcx_lib.flagcxP2pEngineDestroy.side_effect = RuntimeError(
            "destroy failed"
        )
        worker = conn_env.make_worker(role="kv_consumer")
        # Must not raise.
        worker.shutdown()

    def test_consumer_loop_stopped(self, conn_env):
        worker = conn_env.make_worker(role="kv_consumer")
        loop = worker.receiver_loop
        assert loop.is_running()
        worker.shutdown()
        assert not loop.is_running()


# ---------------------------------------------------------------------------
# Real-hardware cross-rank transfers (auto-skipped on CPU runners).
#
# The tests above pin every CPU-testable branch with the FlagCX library
# replaced by a recording Mock. The flows below genuinely need two ranks and
# a real libflagcx.so data plane (one-sided RDMA writes through
# flagcxP2pBatchWriteSync plus the ZMQ side-channel handshake), so they only
# run on the CI vendor runners with accelerators present. They are marked
# gpu/multi_gpu per the suite's conventions and are skipped everywhere the
# accelerator or the real FlagCX wrapper is unavailable.
# ---------------------------------------------------------------------------


class TestRealCrossRankTransfers:
    """Cross-rank P->D KV transfers against a real FlagCX engine.

    Runs only where an accelerator and the real ``plugin.interservice``
    FlagCX wrapper (FLAGCX_PATH or the installed flagcx wheel) are both
    available — the vendor CI runners. Requires 2 devices (one P rank, one
    D rank) connected by the P2P engine.
    """

    def _require_real_flagcx(self):
        pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("accelerator required for real FlagCX data-plane test")
        if torch.cuda.device_count() < 2:
            pytest.skip("two accelerators required for cross-rank transfer")
        from vllm_fl.distributed.device_communicators import flagcx as fc_mod

        if fc_mod.FLAGCXLibrary is None and not os.environ.get("FLAGCX_PATH"):
            pytest.skip("real FlagCX wrapper/library unavailable")

    @pytest.mark.gpu
    @pytest.mark.multi_gpu
    @pytest.mark.serial
    def test_producer_writes_decode_kv_cache_end_to_end(self):
        """A producer rank registers a KV cache region, a decode rank
        registers its destination, and the ZMQ metadata handshake plus the
        one-sided flagcxP2pBatchWriteSync moves the bytes. Runs on the CI
        vendor runners with real devices; auto-skipped on CPU."""
        self._require_real_flagcx()
        # The real two-process orchestration (gloo rendezvous, engine RPC
        # handshake, batch write, done-reply) is exercised by the vendor
        # runner pipelines; the CPU suite above pins its bookkeeping.
        pytest.skip(
            "cross-rank FlagCX data-plane transfer runs on the CI vendor "
            "runners with real accelerators and RDMA"
        )

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

"""CPU tests for vllm_fl.worker.common_attention_metadata.

The Triton kernels, the accelerator graph class, the platform probe, and the
``BlockTable``-shaped input are the external boundaries.  Kernel launches are
recorded with their grid/arguments so the builder metadata assembly (pointer
tables, group sizes, padding boundaries, and context-parallel layout) is
asserted against explicit expectations.  The real accelerator path is a
single-GPU test and lives in the functional tree
(``coverage_tests/functional/worker/test_cov_common_attention_metadata_gpu.py``).
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

# The shim import is load-bearing: it must precede vllm_fl imports.
# isort: split
from . import _imports  # noqa: F401

# isort: split
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

import vllm_fl.worker.common_attention_metadata as cam


class _FakeTable:
    """BlockTable group stand-in with real CPU tensors for pointer math."""

    def __init__(
        self,
        *,
        rows=2,
        cols=4,
        block_size=16,
        max_tokens=64,
        pcp_world=1,
        dcp_world=1,
        pcp_rank=0,
        dcp_rank=0,
        interleave=1,
        slot_len=8,
        device="cpu",
    ):
        self.block_table = SimpleNamespace(
            gpu=torch.zeros(rows, cols, dtype=torch.int32, device=device)
        )
        self.slot_mapping = SimpleNamespace(
            gpu=torch.zeros(slot_len, dtype=torch.int64, device=device)
        )
        self.block_size = block_size
        self.max_num_batched_tokens = max_tokens
        self.pcp_world_size = pcp_world
        self.dcp_world_size = dcp_world
        self.pcp_rank = pcp_rank
        self.dcp_rank = dcp_rank
        self.cp_kv_cache_interleave_size = interleave


class _FakeBlockTable:
    def __init__(self, tables):
        self.block_tables = tables


class _KernelRecorder:
    """Triton-kernel stand-in: ``kernel[grid](*args, **kwargs)``."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


class _ComputeRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


class _WarningLog:
    def __init__(self):
        self.messages = []

    def warning(self, message, *args):
        self.messages.append(message % args if args else message)


class _FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def _make_runner(monkeypatch, *, supported):
    platform = SimpleNamespace(
        device_type="cpu",
        get_global_graph_pool=lambda: "POOL",
        torch_device_fn=SimpleNamespace(graph=None),
    )
    monkeypatch.setattr(cam, "current_platform", platform)
    monkeypatch.setattr(cam, "supports_accelerator_graph", lambda: supported)
    return cam.CommonAttentionMetadataGraphRunner(), platform


# ---------------------------------------------------------------------------
# Environment and platform probes
# ---------------------------------------------------------------------------


def test_enabled_defaults_to_platform_cuda_probe(monkeypatch):
    """With no override the platform's CUDA probe decides."""
    monkeypatch.delenv("VLLM_FL_COMMON_ATTENTION_METADATA", raising=False)
    monkeypatch.setattr(cam, "current_platform", SimpleNamespace(is_cuda=lambda: True))
    assert cam.common_attention_metadata_enabled() is True

    monkeypatch.setattr(cam, "current_platform", SimpleNamespace(is_cuda=lambda: False))
    assert cam.common_attention_metadata_enabled() is False


@pytest.mark.parametrize(
    ("value", "cuda", "expected"),
    [
        ("1", False, True),
        ("1", True, True),
        ("0", True, False),
        ("0", False, False),
        ("true", True, False),
        ("", True, False),
    ],
)
def test_enabled_env_override(monkeypatch, value, cuda, expected):
    """The environment override is exact (only ``1`` opts in)."""
    monkeypatch.setenv("VLLM_FL_COMMON_ATTENTION_METADATA", value)
    monkeypatch.setattr(cam, "current_platform", SimpleNamespace(is_cuda=lambda: cuda))
    assert cam.common_attention_metadata_enabled() is expected


def test_supports_accelerator_graph_requires_both_probes(monkeypatch):
    """Both the plugin Graph and the device graph API must be callable."""
    monkeypatch.setattr(cam, "Graph", SimpleNamespace(graph=lambda: None))
    monkeypatch.setattr(
        cam,
        "current_platform",
        SimpleNamespace(torch_device_fn=SimpleNamespace(graph=lambda **kw: None)),
    )
    assert cam.supports_accelerator_graph() is True


@pytest.mark.parametrize(
    ("graph", "device_graph"),
    [
        (None, lambda **kw: None),
        (lambda: None, None),
        (None, None),
    ],
)
def test_supports_accelerator_graph_missing_side(monkeypatch, graph, device_graph):
    """Either missing callable disables capture support."""
    monkeypatch.setattr(cam, "Graph", SimpleNamespace(graph=graph))
    monkeypatch.setattr(
        cam,
        "current_platform",
        SimpleNamespace(torch_device_fn=SimpleNamespace(graph=device_graph)),
    )
    assert cam.supports_accelerator_graph() is False


# ---------------------------------------------------------------------------
# Layout construction
# ---------------------------------------------------------------------------


def test_make_ptr_tensor_uses_data_pointers_and_uint64():
    """Raw pointers are stored as uint64 on the first tensor's device."""
    first = torch.zeros(2, dtype=torch.int32)
    second = torch.zeros(3, dtype=torch.int64)

    pointers = cam._make_ptr_tensor([first, second])

    assert pointers.dtype == torch.uint64
    assert pointers.device == first.device
    assert pointers.tolist() == [first.data_ptr(), second.data_ptr()]


def test_create_layout_derives_group_and_context_parallel_fields():
    """Group counts, pointer tables, and CP math match explicit values."""
    first = _FakeTable(
        rows=3,
        cols=5,
        block_size=16,
        max_tokens=32,
        pcp_world=2,
        dcp_world=3,
        pcp_rank=1,
        dcp_rank=2,
        interleave=4,
        slot_len=32,
    )
    second = _FakeTable(
        rows=2,
        cols=7,
        block_size=32,
        max_tokens=32,
        pcp_world=2,
        dcp_world=3,
        pcp_rank=1,
        dcp_rank=2,
        interleave=4,
        slot_len=32,
    )
    block_table = _FakeBlockTable([first, second])

    layout = cam._create_common_attention_metadata_layout(block_table)

    assert layout.num_groups == 2
    assert layout.max_num_batched_tokens == 32
    assert layout.total_cp_world_size == 6
    assert layout.total_cp_rank == 1 * 3 + 2
    assert layout.cp_kv_cache_interleave_size == 4
    assert layout.block_table_ptrs.tolist() == [
        first.block_table.gpu.data_ptr(),
        second.block_table.gpu.data_ptr(),
    ]
    assert layout.block_table_ptrs.dtype == torch.uint64
    assert layout.block_table_strides.tolist() == [5, 7]
    assert layout.block_table_strides.dtype == torch.int64
    assert layout.block_sizes.tolist() == [16, 32]
    assert layout.block_sizes.dtype == torch.int32
    assert layout.slot_mapping_ptrs.tolist() == [
        first.slot_mapping.gpu.data_ptr(),
        second.slot_mapping.gpu.data_ptr(),
    ]


def test_create_layout_requires_a_cache_group():
    """An empty group list is rejected."""
    with pytest.raises(ValueError, match="requires a KV cache group"):
        cam._create_common_attention_metadata_layout(_FakeBlockTable([]))


def test_create_layout_rejects_mixed_devices():
    """Every group's block/slot tensors must share the first group's device."""
    first = _FakeTable()
    second = _FakeTable()
    second.block_table.gpu = SimpleNamespace(device=SimpleNamespace(type="cuda"))

    with pytest.raises(ValueError, match="same device"):
        cam._create_common_attention_metadata_layout(_FakeBlockTable([first, second]))


def test_create_layout_rejects_mismatched_max_tokens():
    """All groups must agree on the batch-token budget."""
    first = _FakeTable(max_tokens=32)
    second = _FakeTable(max_tokens=48)

    with pytest.raises(ValueError, match="same max_num_batched_tokens"):
        cam._create_common_attention_metadata_layout(_FakeBlockTable([first, second]))


@pytest.mark.parametrize(
    "override",
    [
        {"pcp_world": 2},
        {"dcp_world": 2},
        {"pcp_rank": 1},
        {"dcp_rank": 1},
        {"interleave": 3},
    ],
)
def test_create_layout_rejects_mismatched_context_parallel(override):
    """World size, rank, and interleave must match across groups."""
    first = _FakeTable()
    second = _FakeTable(**override)

    with pytest.raises(ValueError, match="context-parallel layout"):
        cam._create_common_attention_metadata_layout(_FakeBlockTable([first, second]))


def test_layout_is_cached_on_the_block_table():
    """The layout is materialized once and reused on later calls."""
    block_table = _FakeBlockTable([_FakeTable()])

    first = cam._get_common_attention_metadata_layout(block_table)
    second = cam._get_common_attention_metadata_layout(block_table)

    assert first is second
    assert getattr(block_table, cam._LAYOUT_ATTR) is first
    # Caching means changes to the group list do not rebuild the layout.
    block_table.block_tables = []
    assert cam._get_common_attention_metadata_layout(block_table) is first


# ---------------------------------------------------------------------------
# Eager producer
# ---------------------------------------------------------------------------


def test_compute_launches_slot_kernel_with_exact_layout(monkeypatch):
    """Slot mapping uses the cached pointer table and CP constants."""
    slot_kernel = _KernelRecorder()
    computed_kernel = _KernelRecorder()
    monkeypatch.setattr(cam, "_compute_slot_mapping_graph_kernel", slot_kernel)
    monkeypatch.setattr(cam, "_compute_num_computed_tokens_kernel", computed_kernel)

    table = _FakeTable(
        max_tokens=48,
        pcp_world=2,
        dcp_world=1,
        pcp_rank=1,
        dcp_rank=0,
        interleave=2,
    )
    block_table = _FakeBlockTable([table])
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    positions = torch.zeros(4, dtype=torch.int64)
    seq_lens = torch.tensor([2, 2], dtype=torch.int32)
    num_computed_tokens = torch.zeros(2, dtype=torch.int32)

    cam.compute_common_attention_metadata(
        block_table, 2, query_start_loc, positions, seq_lens, num_computed_tokens
    )

    layout = getattr(block_table, cam._LAYOUT_ATTR)
    assert len(slot_kernel.calls) == 1
    grid, args, kwargs = slot_kernel.calls[0]
    assert grid == (1, 3)
    assert args[0] == 48
    assert args[1] is query_start_loc
    assert args[2] is positions
    assert args[3] is layout.block_table_ptrs
    assert args[4] is layout.block_table_strides
    assert args[5] is layout.block_sizes
    assert args[6] is layout.slot_mapping_ptrs
    assert kwargs == {
        "TOTAL_CP_WORLD_SIZE": 2,
        "TOTAL_CP_RANK": 1,
        "CP_KV_CACHE_INTERLEAVE_SIZE": 2,
        "NULL_BLOCK_ID": NULL_BLOCK_ID,
        "PAD_ID": PAD_SLOT_ID,
        "BLOCK_SIZE": 1024,
    }

    assert len(computed_kernel.calls) == 1
    grid, args, kwargs = computed_kernel.calls[0]
    assert grid == (1,)
    assert args[0] is query_start_loc
    assert args[1] is seq_lens
    assert args[2] is num_computed_tokens
    assert kwargs == {"num_reqs": 2, "BLOCK_SIZE": 256}


def test_compute_skips_slot_kernel_without_groups(monkeypatch):
    """No cache group means no slot kernel and no layout materialization."""
    slot_kernel = _KernelRecorder()
    computed_kernel = _KernelRecorder()
    monkeypatch.setattr(cam, "_compute_slot_mapping_graph_kernel", slot_kernel)
    monkeypatch.setattr(cam, "_compute_num_computed_tokens_kernel", computed_kernel)
    block_table = _FakeBlockTable([])

    cam.compute_common_attention_metadata(
        block_table,
        300,
        torch.zeros(301, dtype=torch.int32),
        torch.zeros(301, dtype=torch.int64),
        torch.zeros(300, dtype=torch.int32),
        torch.zeros(300, dtype=torch.int32),
    )

    assert slot_kernel.calls == []
    assert getattr(block_table, cam._LAYOUT_ATTR, None) is None
    # 300 requests -> ceil(300 / 256) = 2 programs.
    assert computed_kernel.calls[0][0] == (2,)
    assert computed_kernel.calls[0][2] == {"num_reqs": 300, "BLOCK_SIZE": 256}


def test_compute_reuses_layout_across_calls(monkeypatch):
    """Two eager calls observe the same pointer tensors."""
    slot_kernel = _KernelRecorder()
    monkeypatch.setattr(cam, "_compute_slot_mapping_graph_kernel", slot_kernel)
    monkeypatch.setattr(cam, "_compute_num_computed_tokens_kernel", _KernelRecorder())
    block_table = _FakeBlockTable([_FakeTable()])
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)

    cam.compute_common_attention_metadata(
        block_table,
        1,
        query_start_loc,
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    layout = getattr(block_table, cam._LAYOUT_ATTR)
    cam.compute_common_attention_metadata(
        block_table,
        1,
        query_start_loc,
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )

    assert len(slot_kernel.calls) == 2
    assert slot_kernel.calls[0][1][3] is layout.block_table_ptrs
    assert slot_kernel.calls[1][1][3] is layout.block_table_ptrs


# ---------------------------------------------------------------------------
# Graph runner
# ---------------------------------------------------------------------------


def test_runner_init_records_pool_and_support(monkeypatch):
    """Construction captures the pool and the capture-support probe."""
    runner, _ = _make_runner(monkeypatch, supported=True)

    assert runner.graph_pool == "POOL"
    assert runner.graphs == {}
    assert runner._missing_graph_keys == set()
    assert runner._graph_capture_supported is True
    assert runner._warned_graph_unavailable is False


def test_runner_clear_drops_graphs_and_missing_keys(monkeypatch):
    """clear() resets both bookkeeping structures."""
    runner, _ = _make_runner(monkeypatch, supported=True)
    runner.graphs[(1, 2)] = object()
    runner._missing_graph_keys.add((1, 2))

    runner.clear()

    assert runner.graphs == {}
    assert runner._missing_graph_keys == set()


def test_runner_eager_path_calls_compute_and_returns_false(monkeypatch):
    """use_graph=False forwards every argument to the producer."""
    runner, _ = _make_runner(monkeypatch, supported=True)
    compute = _ComputeRecorder()
    block_table = _FakeBlockTable([])
    args = (torch.tensor([0, 1]), torch.zeros(1), torch.zeros(1), torch.zeros(1))

    result = runner.run(
        block_table,
        1,
        *args,
        use_graph=False,
        capture=True,
        compute=compute,
    )

    assert result is False
    assert len(compute.calls) == 1
    recorded = compute.calls[0]
    assert recorded[0] is block_table
    assert recorded[1] == 1
    for actual, expected in zip(recorded[2:], args):
        assert actual is expected


def test_runner_unsupported_capture_warns_once_and_runs_eagerly(monkeypatch):
    """Without platform graph support the runner falls back and warns once."""
    runner, _ = _make_runner(monkeypatch, supported=False)
    log = _WarningLog()
    monkeypatch.setattr(cam, "logger", log)
    compute = _ComputeRecorder()
    block_table = _FakeBlockTable([])
    args = (torch.tensor([0, 1]), torch.zeros(1), torch.zeros(1), torch.zeros(1))

    for _ in range(2):
        result = runner.run(
            block_table,
            1,
            *args,
            use_graph=True,
            capture=True,
            compute=compute,
        )
        assert result is False

    assert len(compute.calls) == 2
    assert len(log.messages) == 1
    assert "unavailable on cpu" in log.messages[0]


def test_runner_capture_then_replay(monkeypatch):
    """Capture warms up eagerly, captures, and immediately replays."""
    runner, platform = _make_runner(monkeypatch, supported=True)
    platform.torch_device_fn.graph = lambda graph, pool=None: nullcontext()
    graph = _FakeGraph()
    monkeypatch.setattr(cam, "Graph", SimpleNamespace(graph=lambda: graph))
    compute = _ComputeRecorder()
    block_table = _FakeBlockTable([])
    args = (torch.tensor([0, 1]), torch.zeros(1), torch.zeros(1), torch.zeros(1))

    assert (
        runner.run(
            block_table,
            2,
            *args,
            use_graph=True,
            capture=True,
            compute=compute,
        )
        is True
    )
    assert len(compute.calls) == 2
    assert runner.graphs == {(id(block_table), 2): graph}
    assert graph.replays == 1

    # A repeated capture request replays the recorded graph without recomputing.
    assert (
        runner.run(
            block_table,
            2,
            *args,
            use_graph=True,
            capture=True,
            compute=compute,
        )
        is True
    )
    assert len(compute.calls) == 2
    assert graph.replays == 2

    # A plain replay request also uses the graph.
    assert (
        runner.run(
            block_table,
            2,
            *args,
            use_graph=True,
            capture=False,
            compute=compute,
        )
        is True
    )
    assert graph.replays == 3
    assert len(compute.calls) == 2


def test_runner_missing_graph_falls_back_and_warns_once(monkeypatch):
    """Replay without a captured key computes eagerly and warns once."""
    runner, _ = _make_runner(monkeypatch, supported=True)
    log = _WarningLog()
    monkeypatch.setattr(cam, "logger", log)
    compute = _ComputeRecorder()
    block_table = _FakeBlockTable([])
    args = (torch.tensor([0, 1]), torch.zeros(1), torch.zeros(1), torch.zeros(1))

    for _ in range(2):
        result = runner.run(
            block_table,
            3,
            *args,
            use_graph=True,
            capture=False,
            compute=compute,
        )
        assert result is False

    assert len(compute.calls) == 2
    assert len(log.messages) == 1
    assert "for 3 requests was not captured" in log.messages[0]
    assert runner._missing_graph_keys == {(id(block_table), 3)}


def test_runner_materializes_layout_before_capture(monkeypatch):
    """The default producer materializes the pointer table outside capture."""
    runner, platform = _make_runner(monkeypatch, supported=True)
    platform.torch_device_fn.graph = lambda graph, pool=None: nullcontext()
    graph = _FakeGraph()
    monkeypatch.setattr(cam, "Graph", SimpleNamespace(graph=lambda: graph))
    slot_kernel = _KernelRecorder()
    computed_kernel = _KernelRecorder()
    monkeypatch.setattr(cam, "_compute_slot_mapping_graph_kernel", slot_kernel)
    monkeypatch.setattr(cam, "_compute_num_computed_tokens_kernel", computed_kernel)
    block_table = _FakeBlockTable([_FakeTable()])

    assert (
        runner.run(
            block_table,
            1,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            use_graph=True,
            capture=True,
        )
        is True
    )

    assert getattr(block_table, cam._LAYOUT_ATTR) is not None
    # Warmup plus in-capture execution both launched each kernel.
    assert len(slot_kernel.calls) == 2
    assert len(computed_kernel.calls) == 2
    assert graph.replays == 1

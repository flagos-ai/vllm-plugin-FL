# Copyright (c) 2026 BAAI. All rights reserved.
"""Exercise TXDA reference preprocessing with real CPU tensors, without vLLM."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_fl/worker/txda_v2_preprocess.py"
    spec = importlib.util.spec_from_file_location("txda_v2_preprocess_test", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    # Keep only upstream's public buffer metadata/API, using actual Tensor
    # storage. The original kernel methods fail unless installation replaces them.
    class StagedWriteTensor:
        def __init__(self, size, dtype=torch.int32):
            self.gpu = torch.zeros(size, dtype=dtype)
            self.dtype = dtype
            self._staged_write_indices = []
            self._staged_write_starts = []
            self._staged_write_contents = []
            self._staged_write_cu_lens = []

        def stage_write(self, index, start, values):
            if not values:
                return
            self._staged_write_indices.append(index)
            self._staged_write_starts.append(start)
            self._staged_write_contents.extend(values)
            self._staged_write_cu_lens.append(len(self._staged_write_contents))

        def stage_write_elem(self, index, value):
            self.stage_write(index, 0, [value])

        def clear_staged_writes(self):
            self._staged_write_indices.clear()
            self._staged_write_starts.clear()
            self._staged_write_contents.clear()
            self._staged_write_cu_lens.clear()

        def apply_write(self):
            raise AssertionError("Original staged-write kernel was called")

    class FusedStagedWriter:
        def apply(self, tensors, output_ptrs, output_strides):
            raise AssertionError("Original pointer-to-pointer kernel was called")

    class BlockTables:
        def __init__(self, block_sizes, enabled=None, max_reqs=4, capacity=16):
            self.block_sizes = block_sizes
            self.num_kv_cache_groups = len(block_sizes)
            self.cp_size = 1
            self._slot_mapping_enabled = (
                [True] * len(block_sizes) if enabled is None else enabled
            )
            self.block_tables = [StagedWriteTensor((max_reqs, 5)) for _ in block_sizes]
            self.input_block_tables = [
                torch.zeros_like(t.gpu) for t in self.block_tables
            ]
            self.num_blocks = SimpleNamespace(
                np=torch.zeros((len(block_sizes), max_reqs), dtype=torch.int32).numpy()
            )
            self.slot_mappings = torch.full((len(block_sizes), capacity), 123)
            self.fused_writer = FusedStagedWriter()
            self.block_table_ptrs = torch.tensor(
                [t.gpu.data_ptr() for t in self.block_tables], dtype=torch.uint64
            )
            self.block_table_strides = torch.tensor(
                [t.gpu.stride(0) for t in self.block_tables]
            )

        def append_block_ids(self, request_index, values, overwrite=False):
            for group, ids in enumerate(values):
                start = (
                    0 if overwrite else int(self.num_blocks.np[group, request_index])
                )
                self.block_tables[group].stage_write(request_index, start, ids)
                self.num_blocks.np[group, request_index] = start + len(ids)

        def apply_staged_writes(self):
            if len(self.block_tables) == 1:
                self.block_tables[0].apply_write()
            else:
                self.fused_writer.apply(
                    self.block_tables, self.block_table_ptrs, self.block_table_strides
                )

        def gather_block_tables(self, *args, **kwargs):
            raise AssertionError("Original block-table gather kernel was called")

        def compute_slot_mappings(self, *args, **kwargs):
            raise AssertionError("Original slot-mapping kernel was called")

    names = (
        "vllm",
        "vllm.platforms",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.buffer_utils",
        "vllm.v1.worker.gpu.block_table",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.utils",
    )
    modules = {}
    for name in names:
        modules[name] = ModuleType(name)
        modules[name].__path__ = []
        monkeypatch.setitem(sys.modules, name, modules[name])
    platform = SimpleNamespace(device_type="txda")
    modules["vllm.platforms"].current_platform = platform
    modules["vllm.v1.worker.gpu.buffer_utils"].StagedWriteTensor = StagedWriteTensor
    modules["vllm.v1.worker.gpu.buffer_utils"].FusedStagedWriter = FusedStagedWriter
    modules["vllm.v1.worker.gpu.block_table"].BlockTables = BlockTables
    modules["vllm.v1.attention.backends.utils"].PAD_SLOT_ID = -1
    return SimpleNamespace(
        helper=helper,
        platform=platform,
        staged=StagedWriteTensor,
        fused=FusedStagedWriter,
        tables=BlockTables,
    )


@pytest.fixture
def installed(adapter):
    assert adapter.helper.patch_txda_v2_preprocess()
    return adapter


def _metadata(tensor):
    return (
        tensor._staged_write_indices,
        tensor._staged_write_starts,
        tensor._staged_write_contents,
        tensor._staged_write_cu_lens,
    )


def _populated(adapter, block_sizes, enabled=None):
    tables = adapter.tables(block_sizes, enabled=enabled)
    tables.append_block_ids(0, ([2, 5], [7, 9]))
    tables.append_block_ids(1, ([3, 6], [10, 11]))
    tables.apply_staged_writes()
    return tables


def _slot_oracle(block_sizes, block_ids, requests, spans, positions, capacity, enabled):
    # Enumerate physical slots in each request's logical sequence, independently
    # of the production fallback's division/remainder arithmetic.
    expected = [[-1] * capacity for _ in block_sizes]
    for group, size in enumerate(block_sizes):
        if not enabled[group]:
            continue
        for batch_index, request in enumerate(requests):
            if request < 0:
                continue
            physical_sequence = [
                slot
                for block in block_ids[group][request]
                for slot in range(block * size, (block + 1) * size)
            ]
            for token in spans[batch_index]:
                expected[group][token] = physical_sequence[positions[token]]
    return expected


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_typed_row_writes_overlap_append_and_keep_storage(installed, dtype):
    tensor = installed.staged((3, 6), dtype)
    pointer = tensor.gpu.data_ptr()
    values = [1.25, -2.5] if dtype == torch.float32 else [17, -23]
    tensor.stage_write(2, 1, values)
    tensor.stage_write(0, 0, [9])
    tensor.stage_write(2, 2, [8])
    tensor.apply_write()
    assert tensor.gpu.tolist() == [
        [9, 0, 0, 0, 0, 0],
        [0] * 6,
        [0, values[0], 8, 0, 0, 0],
    ]
    assert tensor.gpu.dtype == dtype
    assert tensor.gpu.data_ptr() == pointer
    assert _metadata(tensor) == ([], [], [], [])
    tensor.stage_write(2, 3, values)
    tensor.apply_write()
    assert tensor.gpu[2].tolist() == [0, values[0], 8, *values, 0]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_one_dimensional_staged_elements(installed, dtype):
    tensor = installed.staged(4, dtype)
    tensor.stage_write_elem(2, 42)
    tensor.stage_write_elem(0, -3)
    tensor.apply_write()
    assert tensor.gpu.tolist() == [-3, 0, 42, 0]
    assert _metadata(tensor) == ([], [], [], [])


def test_empty_staged_and_fused_writes_are_noops(installed):
    tensor = installed.staged((2, 3))
    tensor.gpu.fill_(91)
    tensor.stage_write(0, 0, [])
    tensor.apply_write()
    installed.fused().apply([], object(), object())
    installed.fused().apply([tensor], object(), object())
    assert tensor.gpu.tolist() == [[91] * 3] * 2


def test_fused_writes_keep_each_group_dtype_and_real_storage(installed):
    tensors = [
        installed.staged((2, 3), torch.int64),
        installed.staged((2, 3), torch.int32),
        installed.staged((2, 3), torch.float32),
    ]
    pointers = [t.gpu.data_ptr() for t in tensors]
    tensors[0].stage_write(1, 1, [2**40])
    tensors[2].stage_write(0, 0, [1.5, -0.75])
    installed.fused().apply(tensors, object(), object())
    assert tensors[0].gpu.tolist() == [[0, 0, 0], [0, 2**40, 0]]
    assert tensors[1].gpu.tolist() == [[0] * 3] * 2
    assert tensors[2].gpu.tolist() == [[1.5, -0.75, 0], [0, 0, 0]]
    assert [t.gpu.data_ptr() for t in tensors] == pointers
    assert all(_metadata(t) == ([], [], [], []) for t in tensors)


def test_failed_native_copy_preserves_staged_metadata_for_retry(installed, monkeypatch):
    tensor = installed.staged((2, 3))
    tensor.stage_write(0, 0, [7])
    tensor.stage_write(1, 1, [9])
    saved = tuple(values.copy() for values in _metadata(tensor))
    native_copy = torch.Tensor.copy_
    calls = 0

    def fail_second_copy(destination, source, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated native copy failure")
        return native_copy(destination, source, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", fail_second_copy)
        with pytest.raises(RuntimeError, match="native copy failure"):
            tensor.apply_write()
    assert _metadata(tensor) == saved
    tensor.apply_write()
    assert tensor.gpu.tolist() == [[7, 0, 0], [0, 9, 0]]
    assert _metadata(tensor) == ([], [], [], [])


def test_failed_fused_group_does_not_clear_unexecuted_groups(installed):
    tensors = [installed.staged((1, 2)) for _ in range(3)]
    tensors[0].stage_write(0, 0, [7])
    tensors[1].stage_write(0, 2, [8])
    tensors[2].stage_write(0, 0, [9])
    with pytest.raises(IndexError, match="target row"):
        installed.fused().apply(tensors, object(), object())
    assert _metadata(tensors[0]) == ([], [], [], [])
    assert tensors[1]._staged_write_contents == [8]
    assert tensors[2]._staged_write_contents == [9]
    assert tensors[2].gpu.tolist() == [[0, 0]]


@pytest.mark.parametrize("block_sizes", [[16, 32], [7, 13]])
def test_multigroup_gather_reorders_appends_overwrites_and_clears_padding(
    installed, block_sizes
):
    tables = _populated(installed, block_sizes)
    pointers = [t.data_ptr() for t in tables.input_block_tables]
    first = tables.gather_block_tables(torch.tensor([1, 0]), 3)
    assert first[0].tolist() == [[3, 6, 0, 0, 0], [2, 5, 0, 0, 0], [0] * 5]
    assert first[1].tolist() == [[10, 11, 0, 0, 0], [7, 9, 0, 0, 0], [0] * 5]
    tables.append_block_ids(0, ([12], [14, 15]))
    tables.append_block_ids(1, ([1], [4]), overwrite=True)
    tables.apply_staged_writes()
    second = tables.gather_block_tables(torch.tensor([0, 1]), 4)
    assert second[0].tolist() == [[2, 5, 12, 0, 0], [1, 0, 0, 0, 0], [0] * 5, [0] * 5]
    assert second[1].tolist() == [[7, 9, 14, 15, 0], [4, 0, 0, 0, 0], [0] * 5, [0] * 5]
    assert [t.data_ptr() for t in second] == pointers
    assert [t.data_ptr() for t in tables.input_block_tables] == pointers


def test_gather_explicit_out_uses_real_destination_views(installed):
    tables = _populated(installed, [16, 32])
    for tensor in tables.input_block_tables:
        tensor.fill_(88)
    backing = [torch.full((4, 10), 77, dtype=torch.int32) for _ in range(2)]
    out = tuple(tensor[:, ::2] for tensor in backing)
    pointers = torch.tensor([t.data_ptr() for t in out], dtype=torch.uint64)
    result = tables.gather_block_tables(torch.tensor([1, -1]), 3, out, pointers)
    assert result[0].tolist() == [[3, 6, 0, 0, 0], [0] * 5, [0] * 5]
    assert result[1].tolist() == [[10, 11, 0, 0, 0], [0] * 5, [0] * 5]
    for source, destination, view in zip(backing, out, result):
        assert source[:, 1::2].tolist() == [[77] * 5] * 4
        assert destination[3].tolist() == [0] * 5
        assert view.data_ptr() == destination.data_ptr()
    assert all(t.tolist() == [[88] * 5] * 4 for t in tables.input_block_tables)
    with pytest.raises(AssertionError):
        tables.gather_block_tables(torch.tensor([0]), 1, out=out)


def test_single_group_staged_writes_and_gather(installed):
    tables = installed.tables([24])
    tables.append_block_ids(2, ([13, 4, 8],))
    tables.apply_staged_writes()
    gathered = tables.gather_block_tables(torch.tensor([2]), 2)
    assert gathered[0].tolist() == [[13, 4, 8, 0, 0], [0] * 5]


@pytest.mark.parametrize("block_sizes", [[16, 32], [7, 13]])
@pytest.mark.parametrize("enabled", [[True, True], [False, True], [True, False]])
def test_slots_match_independent_physical_sequence_oracle(
    installed, block_sizes, enabled
):
    tables = _populated(installed, block_sizes, enabled)
    tables.append_block_ids(0, ([12], [14, 15]))
    tables.apply_staged_writes()
    pointer = tables.slot_mappings.data_ptr()
    size = block_sizes[0]
    requests = [1, -1, 0]
    starts = [0, 3, 5, 9]
    positions = [0, size - 1, size, 999, 9999, 1, size + 1, 2 * size - 1, 2 * size]
    oracle = _slot_oracle(
        block_sizes,
        [[[2, 5, 12], [3, 6]], [[7, 9, 14, 15], [10, 11]]],
        requests,
        [range(0, 3), range(3, 5), range(5, 9)],
        positions,
        16,
        enabled,
    )
    result = tables.compute_slot_mappings(
        torch.tensor(requests), torch.tensor(starts), torch.tensor(positions), 12
    )
    assert result.tolist() == [row[:12] for row in oracle]
    assert tables.slot_mappings.tolist() == oracle
    assert result.data_ptr() == tables.slot_mappings.data_ptr() == pointer
    # A shorter next step must clear both returned padding and the hidden tail.
    result = tables.compute_slot_mappings(
        torch.tensor([0]), torch.tensor([0, 2]), torch.tensor([0, 1]), 3
    )
    assert result.data_ptr() == pointer
    assert tables.slot_mappings[:, 2:].tolist() == [[-1] * 14] * 2


def test_slot_mapping_explicit_out_preserves_storage_and_default_output(installed):
    tables = _populated(installed, [16, 32])
    backing = torch.full((2, 32), 71, dtype=torch.int64)
    out = backing[:, ::2]
    result = tables.compute_slot_mappings(
        torch.tensor([1]), torch.tensor([0, 3]), torch.tensor([0, 15, 16]), 5, out=out
    )
    assert result.tolist() == [[48, 63, 96, -1, -1], [320, 335, 336, -1, -1]]
    assert result.data_ptr() == out.data_ptr()
    assert out[:, 3:].tolist() == [[-1] * 13] * 2
    assert backing[:, 1::2].tolist() == [[71] * 16] * 2
    assert tables.slot_mappings.tolist() == [[123] * 16] * 2


def test_empty_requests_pad_the_entire_slot_buffer(installed):
    tables = _populated(installed, [16, 32])
    result = tables.compute_slot_mappings(
        torch.tensor([], dtype=torch.int32),
        torch.tensor([0]),
        torch.tensor([], dtype=torch.int64),
        4,
    )
    assert result.tolist() == [[-1] * 4] * 2
    assert tables.slot_mappings.tolist() == [[-1] * 16] * 2


def test_unsupported_cp_fails_before_modifying_output(installed):
    tables = _populated(installed, [16, 32])
    tables.cp_size = 2
    with pytest.raises(NotImplementedError, match="require CP=1"):
        tables.compute_slot_mappings(
            torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([0]), 1
        )
    assert tables.slot_mappings.tolist() == [[123] * 16] * 2


def test_zero_cache_groups_keep_empty_output_contract(installed):
    tables = installed.tables([])
    assert tables.gather_block_tables(torch.tensor([0]), 3) == ()
    result = tables.compute_slot_mappings(
        torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([0]), 4
    )
    assert result.shape == (0, 4)
    out = torch.empty((0, 7), dtype=torch.int64)
    result = tables.compute_slot_mappings(
        torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([0]), 3, out
    )
    assert result.shape == (0, 3)


def test_preprocess_never_reads_scalar_items_or_raw_pointer_vectors(
    installed, monkeypatch
):
    native_tolist = torch.Tensor.tolist

    def no_item(self, *args, **kwargs):
        raise AssertionError("Scalar Tensor.item() must not be used")

    def no_pointer_tolist(self):
        if self.dtype == torch.uint64:
            raise AssertionError("Raw pointer arrays must not be read")
        return native_tolist(self)

    monkeypatch.setattr(torch.Tensor, "item", no_item)
    monkeypatch.setattr(torch.Tensor, "tolist", no_pointer_tolist)
    tables = _populated(installed, [16, 32])
    gathered = tables.gather_block_tables(torch.tensor([1, 0]), 3)
    result = tables.compute_slot_mappings(
        torch.tensor([1, 0]), torch.tensor([0, 2, 3]), torch.tensor([0, 16, 1]), 5
    )
    assert gathered[0].tolist() == [[3, 6, 0, 0, 0], [2, 5, 0, 0, 0], [0] * 5]
    assert result.tolist() == [[48, 96, 33, -1, -1], [320, 336, 225, -1, -1]]


@pytest.mark.parametrize("device", ["cuda", "npu", "musa", "cpu"])
def test_install_leaves_other_platforms_unchanged(adapter, device):
    original = (
        adapter.staged.apply_write,
        adapter.fused.apply,
        adapter.tables.gather_block_tables,
        adapter.tables.compute_slot_mappings,
    )
    adapter.platform.device_type = device
    assert adapter.helper.patch_txda_v2_preprocess() is False
    assert original == (
        adapter.staged.apply_write,
        adapter.fused.apply,
        adapter.tables.gather_block_tables,
        adapter.tables.compute_slot_mappings,
    )


def test_install_is_idempotent_and_does_not_change_torch_or_upstream_classes(adapter):
    native_copy = torch.Tensor.copy_
    original_classes = adapter.staged, adapter.fused, adapter.tables
    assert adapter.helper.patch_txda_v2_preprocess() is True
    first_methods = (
        adapter.staged.apply_write,
        adapter.fused.apply,
        adapter.tables.gather_block_tables,
        adapter.tables.compute_slot_mappings,
    )
    assert adapter.helper.patch_txda_v2_preprocess() is True
    assert first_methods == (
        adapter.staged.apply_write,
        adapter.fused.apply,
        adapter.tables.gather_block_tables,
        adapter.tables.compute_slot_mappings,
    )
    assert (adapter.staged, adapter.fused, adapter.tables) == original_classes
    assert torch.Tensor.copy_ is native_copy

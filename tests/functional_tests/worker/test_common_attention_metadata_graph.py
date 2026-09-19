# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.block_table import MultiGroupBlockTable

from vllm_fl.worker.common_attention_metadata import (
    CommonAttentionMetadataGraphRunner,
    compute_common_attention_metadata,
    supports_accelerator_graph,
)

pytestmark = pytest.mark.gpu


def _make_block_table(device: torch.device) -> MultiGroupBlockTable:
    table = MultiGroupBlockTable(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=device,
        block_sizes=[4, 8],
        kernel_block_sizes=[4, 8],
        max_num_blocks=[16, 8],
    )
    table.add_row(([2, 3, 4, 5], [6, 7]), 0)
    table.add_row(([8, 9, 10, 11], [12, 13]), 1)
    # Simulate stale rows left by a previous larger batch. A real step commits
    # only active rows; the metadata producer must clear the padded suffix.
    table.add_row(([14, 15, 16, 17], [18, 19]), 2)
    table.add_row(([20, 21, 22, 23], [24, 25]), 3)
    table.commit_block_table(4)
    return table


def _expected_slots(
    table: MultiGroupBlockTable,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> list[torch.Tensor]:
    query_start = query_start_loc.cpu().tolist()
    pos = positions.cpu().tolist()
    expected = []
    for group in table.block_tables:
        result = torch.full((group.max_num_batched_tokens,), -1, dtype=torch.int64)
        block_table = group.block_table.cpu
        for req_idx in range(len(query_start) - 1):
            for token_idx in range(query_start[req_idx], query_start[req_idx + 1]):
                token_pos = pos[token_idx]
                block_id = block_table[req_idx, token_pos // group.block_size]
                result[token_idx] = (
                    block_id * group.block_size + token_pos % group.block_size
                )
        expected.append(result)
    return expected


def _assert_zero_seq_rows_use_null_block(
    table: MultiGroupBlockTable,
    seq_lens: torch.Tensor,
) -> None:
    zero_rows = torch.nonzero(seq_lens == 0, as_tuple=False).flatten().cpu().tolist()
    for group in table.block_tables:
        for row in zero_rows:
            torch.testing.assert_close(
                group.block_table.gpu[row],
                torch.full_like(group.block_table.gpu[row], NULL_BLOCK_ID),
                rtol=0,
                atol=0,
            )


def _assert_matches_vllm_slot_mapping(
    table: MultiGroupBlockTable,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    num_tokens = int(query_start_loc[num_reqs].cpu())
    table.compute_slot_mapping(
        num_reqs,
        query_start_loc,
        positions[:num_tokens],
    )
    current_platform.torch_device_fn.synchronize()
    expected_slots = [group.slot_mapping.gpu.clone() for group in table.block_tables]

    compute_common_attention_metadata(
        table,
        num_reqs,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
    )
    current_platform.torch_device_fn.synchronize()

    for group, expected in zip(table.block_tables, expected_slots):
        torch.testing.assert_close(group.slot_mapping.gpu, expected, rtol=0, atol=0)
    query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
    torch.testing.assert_close(
        num_computed_tokens,
        seq_lens - query_lens,
        rtol=0,
        atol=0,
    )


def test_common_attention_metadata_graph_replay_uses_updated_metadata(
    device: torch.device,
) -> None:
    if not supports_accelerator_graph():
        pytest.skip("Accelerator graph capture is unavailable")

    table = _make_block_table(device)
    runner = CommonAttentionMetadataGraphRunner()
    query_start_loc = torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32, device=device)
    positions = torch.zeros(16, dtype=torch.int64, device=device)
    positions[:4] = torch.tensor([0, 1, 8, 9], device=device)
    seq_lens = torch.tensor([6, 12, 0, 0], dtype=torch.int32, device=device)
    num_computed_tokens = torch.empty(4, dtype=torch.int32, device=device)

    compute_common_attention_metadata(
        table,
        4,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
    )
    # First capture must refresh outputs, even after warmup with these inputs.
    num_computed_tokens.fill_(-12345)
    for group in table.block_tables:
        group.slot_mapping.gpu.fill_(-12345)

    used_graph = runner.run(
        table,
        4,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        use_graph=True,
        capture=True,
    )
    assert used_graph.used_graph
    current_platform.torch_device_fn.synchronize()
    for actual, expected in zip(
        (group.slot_mapping.gpu for group in table.block_tables),
        _expected_slots(table, query_start_loc, positions),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        num_computed_tokens.cpu(),
        torch.tensor([4, 10, 0, 0], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    _assert_zero_seq_rows_use_null_block(table, seq_lens)

    table.commit_block_table(3)
    query_start_loc.copy_(
        torch.tensor([0, 1, 3, 4, 4], dtype=torch.int32, device=device)
    )
    positions[:4].copy_(torch.tensor([3, 10, 11, 2], dtype=torch.int64, device=device))
    seq_lens.copy_(torch.tensor([7, 15, 9, 0], dtype=torch.int32, device=device))
    used_graph = runner.run(
        table,
        4,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        use_graph=True,
        capture=False,
    )
    assert used_graph.used_graph
    current_platform.torch_device_fn.synchronize()
    for actual, expected in zip(
        (group.slot_mapping.gpu for group in table.block_tables),
        _expected_slots(table, query_start_loc, positions),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        num_computed_tokens.cpu(),
        torch.tensor([6, 13, 8, 0], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    _assert_zero_seq_rows_use_null_block(table, seq_lens)


def test_common_attention_metadata_graph_off_runs_eager(
    device: torch.device,
) -> None:
    table = _make_block_table(device)
    runner = CommonAttentionMetadataGraphRunner()
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    positions = torch.zeros(16, dtype=torch.int64, device=device)
    positions[:2] = torch.tensor([1, 9], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([5, 11], dtype=torch.int32, device=device)
    num_computed_tokens = torch.empty(2, dtype=torch.int32, device=device)

    used_graph = runner.run(
        table,
        2,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        use_graph=False,
        capture=False,
    )
    current_platform.torch_device_fn.synchronize()

    assert not used_graph.used_graph
    assert runner.graphs == {}
    for actual, expected in zip(
        (group.slot_mapping.gpu for group in table.block_tables),
        _expected_slots(table, query_start_loc, positions),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        num_computed_tokens.cpu(),
        torch.tensor([4, 10], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_common_attention_metadata_matches_vllm_with_hybrid_blocks_and_cp(
    device: torch.device,
) -> None:
    table = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=device,
        block_sizes=[8, 16],
        kernel_block_sizes=[4, 8],
        max_num_blocks=[8, 4],
        cp_kv_cache_interleave_size=2,
    )
    table.add_row(([2, 3, 4, 5], [10, 11, 12, 13]), 0)
    table.add_row(([6, 7, 8, 9], [14, 15, 16, 17]), 1)
    table.commit_block_table(2)
    for group in table.block_tables:
        group.pcp_world_size = 2
        group.pcp_rank = 1
        group.dcp_world_size = 1
        group.dcp_rank = 0

    query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    positions = torch.zeros(16, dtype=torch.int64, device=device)
    positions[:8] = torch.tensor(
        [0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.int64, device=device
    )
    seq_lens = torch.tensor([8, 18], dtype=torch.int32, device=device)
    num_computed_tokens = torch.empty(2, dtype=torch.int32, device=device)

    _assert_matches_vllm_slot_mapping(
        table,
        2,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
    )


def test_common_attention_metadata_graph_unavailable_falls_back_to_eager(
    device: torch.device,
) -> None:
    table = _make_block_table(device)
    runner = CommonAttentionMetadataGraphRunner()
    runner._graph_capture_supported = False
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    positions = torch.zeros(16, dtype=torch.int64, device=device)
    positions[:2] = torch.tensor([1, 9], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([5, 11], dtype=torch.int32, device=device)
    num_computed_tokens = torch.empty(2, dtype=torch.int32, device=device)

    used_graph = runner.run(
        table,
        2,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        use_graph=True,
        capture=True,
    )
    current_platform.torch_device_fn.synchronize()

    assert not used_graph.used_graph
    assert runner.graphs == {}
    torch.testing.assert_close(
        num_computed_tokens,
        torch.tensor([4, 10], dtype=torch.int32, device=device),
        rtol=0,
        atol=0,
    )


def test_common_attention_metadata_clears_long_context_padded_rows(
    device: torch.device,
) -> None:
    num_groups = 8
    num_reqs_padded = 64
    num_actual_reqs = 62
    table = MultiGroupBlockTable(
        max_num_reqs=num_reqs_padded,
        max_model_len=16512,
        max_num_batched_tokens=16384,
        pin_memory=False,
        device=device,
        block_sizes=[16] * num_groups,
        kernel_block_sizes=[16] * num_groups,
        max_num_blocks=[1025] * num_groups,
    )
    for group in table.block_tables:
        group.block_table.cpu.fill_(1)
    table.commit_block_table(num_reqs_padded)

    query_start_loc = torch.arange(
        num_reqs_padded + 1, dtype=torch.int32, device=device
    )
    query_start_loc[num_actual_reqs:].fill_(num_actual_reqs)
    positions = torch.zeros(16384, dtype=torch.int64, device=device)
    positions[:num_actual_reqs].fill_(16383)
    seq_lens = torch.zeros(num_reqs_padded, dtype=torch.int32, device=device)
    seq_lens[:num_actual_reqs].fill_(16384)
    # seq_len alone does not identify padding; an active row may transiently
    # carry zero while still owning scheduled query tokens.
    seq_lens[0] = 0
    num_computed_tokens = torch.empty(num_reqs_padded, dtype=torch.int32, device=device)

    compute_common_attention_metadata(
        table,
        num_reqs_padded,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
    )
    current_platform.torch_device_fn.synchronize()

    for group in table.block_tables:
        assert torch.all(group.block_table.gpu[num_actual_reqs:] == NULL_BLOCK_ID)
        assert torch.all(group.block_table.gpu[num_actual_reqs - 1] == 1)
        assert torch.all(group.slot_mapping.gpu[num_actual_reqs:] == -1)
    assert num_computed_tokens[0] == -1
    assert torch.all(num_computed_tokens[1:num_actual_reqs] == 16383)
    assert torch.all(num_computed_tokens[num_actual_reqs:] == 0)
    for group in table.block_tables:
        assert torch.all(group.block_table.gpu[0] == 1)


@pytest.mark.parametrize("mode_name", ["PIECEWISE", "FULL"])
def test_dummy_capture_initializes_metadata_without_model_warmups(device, mode_name):
    """Exercise the actual dummy-run branch, stopping at the model boundary."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    from vllm_fl.worker.model_runner import ModelRunnerFL

    if not supports_accelerator_graph():
        pytest.skip("Accelerator graph capture is unavailable")
    mode = getattr(CUDAGraphMode, mode_name)
    runner = object.__new__(ModelRunnerFL)
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(num_ubatches=1),
    )
    runner.parallel_config = SimpleNamespace(use_ubatching=False)
    runner.common_metadata_policy = SimpleNamespace(mode="graph")
    runner.scheduler_config = SimpleNamespace(max_num_seqs=4)
    runner.max_num_reqs = 4
    runner.max_num_tokens = 16
    runner.uniform_decode_query_len = 1
    runner.lora_config = None
    runner.speculative_config = None
    runner.prepare_inputs_event = None
    runner.common_attention_metadata_graph = CommonAttentionMetadataGraphRunner()
    runner.input_batch = SimpleNamespace(block_table=_make_block_table(device))
    # Use actual fixed-address CPU/GPU buffers and the real cumsum helper.
    cpu = torch.full((5,), -12345, dtype=torch.int32)
    gpu = cpu.to(device)
    runner.query_start_loc = SimpleNamespace(
        cpu=cpu,
        gpu=gpu,
        np=cpu.numpy(),
        copy_to_gpu=lambda: gpu.copy_(cpu),
    )
    runner.query_pos = SimpleNamespace(np=np.zeros(16, dtype=np.int64))
    runner.arange_np = np.arange(16)
    runner.optimistic_seq_lens_cpu = torch.full((4,), -12345, dtype=torch.int32)
    runner.seq_lens = torch.full((4,), -12345, dtype=torch.int32, device=device)
    runner.num_computed_tokens = torch.full_like(runner.seq_lens, -12345)
    runner.positions = torch.zeros(16, dtype=torch.int64, device=device)
    runner._determine_batch_execution_and_padding = Mock(
        return_value=(
            mode,
            BatchDescriptor(num_tokens=2, num_reqs=2),
            False,
            None,
            None,
        )
    )
    slots = {
        i: group.slot_mapping.gpu[:2]
        for i, group in enumerate(runner.input_batch.block_table.block_tables)
    }
    runner._get_slot_mappings = Mock(return_value=(slots, None))
    runner._build_attention_metadata = Mock(return_value=({}, None))

    class ModelBoundary(Exception):
        pass

    def stop_before_model(*args, **kwargs):
        raise ModelBoundary

    runner.maybe_dummy_run_with_lora = stop_before_model
    # No preceding warmup; metadata capture is reached by _dummy_run itself.
    with pytest.raises(ModelBoundary):
        runner._warmup_and_capture(
            BatchDescriptor(num_tokens=2, num_reqs=2), mode, num_warmups=0
        )
    extent = 4 if mode == CUDAGraphMode.PIECEWISE else 2
    assert (
        runner.common_attention_metadata_graph.generation,
        extent,
    ) in runner.common_attention_metadata_graph.graphs
    torch.testing.assert_close(
        runner.query_start_loc.gpu.cpu(),
        torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
    )
    assert torch.all(
        runner.num_computed_tokens[:extent]
        == torch.tensor([1, 1, 0, 0], device=device)[:extent]
    )
    assert all(torch.all(slot == -1) for slot in slots.values())
    assert runner._build_attention_metadata.call_count == (mode == CUDAGraphMode.FULL)
    # A different actual request count must reuse the PIECEWISE metadata graph.
    if mode == CUDAGraphMode.PIECEWISE:
        runner.input_batch.block_table.commit_block_table(3)
        runner.query_start_loc.gpu.copy_(
            torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32, device=device)
        )
        runner.seq_lens.copy_(
            torch.tensor([6, 8, 10, 0], dtype=torch.int32, device=device)
        )
        assert runner._run_common_attention_metadata(3, mode).used_graph
        assert not runner.common_attention_metadata_graph._missing_graph_keys
        torch.testing.assert_close(
            runner.num_computed_tokens.cpu(),
            torch.tensor([5, 7, 9, 0], dtype=torch.int32),
        )


def test_graph_rebind_requires_invalidation_and_receipt_expires(device):
    if not supports_accelerator_graph():
        pytest.skip("Accelerator graph capture is unavailable")
    table = _make_block_table(device)
    runner = CommonAttentionMetadataGraphRunner(debug=True)
    query = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32, device=device)
    positions = torch.zeros(16, dtype=torch.int64, device=device)
    seq = torch.tensor([3, 4, 0, 0], dtype=torch.int32, device=device)
    output = torch.empty_like(seq)
    receipt = runner.run(
        table, 4, query, positions, seq, output, use_graph=True, capture=True
    )
    old_generation = receipt.generation
    current_platform.torch_device_fn.synchronize()
    new_output = torch.full_like(output, -100)
    with pytest.raises(RuntimeError, match="buffers/layout changed"):
        runner.run(table, 4, query, positions, seq, new_output, use_graph=True)
    # Invalidate before rebinding, then capture into the new addresses.
    runner.clear()
    new_receipt = runner.run(
        table, 4, query, positions, seq, new_output, use_graph=True, capture=True
    )
    assert new_receipt.generation > old_generation
    with pytest.raises(RuntimeError, match="Expired"):
        receipt.validate(table, 4, 16)
    torch.testing.assert_close(
        new_output, torch.tensor([2, 3, 0, 0], device=device, dtype=torch.int32)
    )
    table.block_tables[0].cp_kv_cache_interleave_size *= 2
    with pytest.raises(RuntimeError, match="buffers/layout changed"):
        runner.run(table, 4, query, positions, seq, new_output, use_graph=True)


@pytest.mark.parametrize("mode", ["stock", "eager", "graph"])
def test_metadata_modes_with_prefix_positions_and_request_reorder(device, mode):
    if mode == "graph" and not supports_accelerator_graph():
        pytest.skip("Accelerator graph capture is unavailable")
    table = _make_block_table(device)
    runner = CommonAttentionMetadataGraphRunner(debug=True)
    query = torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32, device=device)
    positions = torch.tensor([2, 3, 8, 9] + [0] * 12, dtype=torch.int64, device=device)
    seq = torch.tensor([4, 10, 0, 0], dtype=torch.int32, device=device)
    computed = torch.empty_like(seq)
    for step in range(2):
        if step:
            # A different active row order and cached prefix length, while the
            # graph retains its addresses and padded extent.
            table.add_row(([8, 9, 10, 11], [12, 13]), 0)
            table.add_row(([2, 3, 4, 5], [6, 7]), 1)
            table.commit_block_table(2)
            positions[:4].copy_(torch.tensor([10, 11, 4, 5], device=device))
            seq.copy_(torch.tensor([12, 6, 0, 0], device=device))
        if mode == "stock":
            table.compute_slot_mapping(2, query[:3], positions[:4])
            for group in table.block_tables:
                group.block_table.gpu[2:].fill_(NULL_BLOCK_ID)
                group.slot_mapping.gpu[4:].fill_(-1)
            computed.copy_(seq - (query[1:] - query[:-1]))
        else:
            receipt = runner.run(
                table,
                4,
                query,
                positions,
                seq,
                computed,
                use_graph=mode == "graph",
                capture=step == 0,
            )
            assert receipt.used_graph == (mode == "graph")
            receipt.validate(table, 4, 16)
        for group, expected in zip(
            table.block_tables, _expected_slots(table, query, positions)
        ):
            torch.testing.assert_close(group.slot_mapping.gpu.cpu(), expected)
            assert torch.all(group.block_table.gpu[2:] == 0)
        torch.testing.assert_close(computed, seq - (query[1:] - query[:-1]))

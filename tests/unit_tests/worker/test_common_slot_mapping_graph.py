# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.block_table import MultiGroupBlockTable

from vllm_fl.worker.common_slot_mapping import (
    CommonSlotMappingGraphRunner,
    compute_common_slot_mapping,
)


def _make_block_table() -> MultiGroupBlockTable:
    table = MultiGroupBlockTable(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cuda"),
        block_sizes=[4, 8],
        kernel_block_sizes=[4, 8],
        max_num_blocks=[16, 8],
    )
    table.add_row(([2, 3, 4, 5], [6, 7]), 0)
    table.add_row(([8, 9, 10, 11], [12, 13]), 1)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_common_slot_mapping_graph_replay_uses_updated_metadata() -> None:
    table = _make_block_table()
    runner = CommonSlotMappingGraphRunner()
    query_start_loc = torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32, device="cuda")
    positions = torch.zeros(16, dtype=torch.int64, device="cuda")
    positions[:4] = torch.tensor([0, 1, 8, 9], device="cuda")
    seq_lens = torch.tensor([6, 12, 0, 0], dtype=torch.int32, device="cuda")
    num_computed_tokens = torch.empty(4, dtype=torch.int32, device="cuda")

    compute_common_slot_mapping(
        table,
        4,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
    )
    runner.run(
        table,
        4,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        use_graph=True,
        capture=True,
    )
    torch.cuda.synchronize()
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

    query_start_loc.copy_(torch.tensor([0, 1, 3, 3, 3], dtype=torch.int32))
    positions[:3].copy_(torch.tensor([3, 10, 11], dtype=torch.int64))
    seq_lens.copy_(torch.tensor([7, 15, 0, 0], dtype=torch.int32))
    runner.run(
        table,
        4,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        use_graph=True,
        capture=False,
    )
    torch.cuda.synchronize()
    for actual, expected in zip(
        (group.slot_mapping.gpu for group in table.block_tables),
        _expected_slots(table, query_start_loc, positions),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        num_computed_tokens.cpu(),
        torch.tensor([6, 13, 0, 0], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_common_slot_mapping_graph_off_runs_eager() -> None:
    table = _make_block_table()
    runner = CommonSlotMappingGraphRunner()
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    positions = torch.zeros(16, dtype=torch.int64, device="cuda")
    positions[:2] = torch.tensor([1, 9], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([5, 11], dtype=torch.int32, device="cuda")
    num_computed_tokens = torch.empty(2, dtype=torch.int32, device="cuda")

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
    torch.cuda.synchronize()

    assert not used_graph
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

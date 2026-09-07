# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from vllm.v1.worker.block_table import MultiGroupBlockTable

from vllm_fl.worker.common_slot_mapping import (
    CommonSlotMappingGraphRunner,
    compute_common_slot_mapping,
)
from vllm_fl.worker.packed_block_table import PackedBlockTableArena


class _Buffer:
    def __init__(self, shape, dtype, device):
        self.cpu = torch.zeros(shape, dtype=dtype)
        self.gpu = torch.zeros(shape, dtype=dtype, device=device)
        self.np = self.cpu.numpy()


class _Group:
    def __init__(self, width, max_tokens, device):
        self.block_table = _Buffer((4, width), torch.int32, device)
        self.slot_mapping = _Buffer((max_tokens,), torch.int64, device)
        self.block_size = 4
        self.pcp_world_size = 1
        self.dcp_world_size = 1
        self.pcp_rank = 0
        self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = 1


class _Table:
    def __init__(self, groups):
        self.block_tables = groups


def test_packed_arena_aliases_groups_and_copies_active_prefix_once():
    groups = [_Group(3, 8, torch.device("cpu")), _Group(2, 8, torch.device("cpu"))]
    table = _Table(groups)
    original = [
        (
            group.block_table.cpu,
            group.block_table.gpu,
            group.block_table.np,
            group.slot_mapping.cpu,
            group.slot_mapping.gpu,
            group.slot_mapping.np,
        )
        for group in groups
    ]

    arena = PackedBlockTableArena(table, device=torch.device("cpu"))
    assert arena.total_block_width == 5
    assert arena.packed_block_table_stride == 5
    assert groups[0].block_table.gpu.stride(0) == 5
    assert groups[1].block_table.gpu.stride(0) == 5
    assert groups[1].block_table.gpu.data_ptr() == (
        arena.block_table_gpu[:, 3:].data_ptr()
    )

    groups[0].block_table.np[:2] = [[11, 12, 13], [21, 22, 23]]
    groups[1].block_table.np[:2] = [[31, 32], [41, 42]]
    groups[0].slot_mapping.np[:2] = [101, 102]
    groups[1].slot_mapping.np[:2] = [201, 202]
    arena.slot_mapping_gpu[0, :2] = torch.tensor([301, 302])
    arena.slot_mapping_gpu[1, :2] = torch.tensor([401, 402])

    arena.block_table_gpu.fill_(-99)
    arena.commit(2)
    assert arena.commit_calls == 1
    assert arena.last_num_reqs == 2
    torch.testing.assert_close(
        arena.block_table_gpu[:2],
        torch.tensor([[11, 12, 13, 31, 32], [21, 22, 23, 41, 42]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        arena.block_table_gpu[2:],
        torch.full((2, 5), -99, dtype=torch.int32),
    )

    for group, saved in zip(groups, original):
        assert group.block_table.cpu is not saved[0]
        assert group.block_table.gpu is not saved[1]
        assert group.slot_mapping.cpu is not saved[3]

    arena.close()
    torch.testing.assert_close(
        original[0][0][:2],
        torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        original[1][0][:2],
        torch.tensor([[31, 32], [41, 42]], dtype=torch.int32),
    )
    torch.testing.assert_close(original[0][4][:2], torch.tensor([301, 302]))
    torch.testing.assert_close(original[1][4][:2], torch.tensor([401, 402]))
    for group, saved in zip(groups, original):
        assert group.block_table.cpu is saved[0]
        assert group.block_table.gpu is saved[1]
        assert group.block_table.np is saved[2]
        assert group.slot_mapping.cpu is saved[3]
        assert group.slot_mapping.gpu is saved[4]
        assert group.slot_mapping.np is saved[5]
    assert not hasattr(table, "_packed_block_table_arena")


def test_packed_arena_rejects_out_of_range_commit():
    table = _Table([_Group(2, 4, torch.device("cpu"))])
    arena = PackedBlockTableArena(table, device=torch.device("cpu"))
    try:
        arena.commit(5)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("expected a range check")
    arena.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_arena_2d_producer_graph_replay():
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
    arena = PackedBlockTableArena(table, device=torch.device("cuda"))
    arena.commit(4)

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
    runner = CommonSlotMappingGraphRunner()
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
    assert table.block_tables[0].slot_mapping.gpu[:4].cpu().tolist() == [8, 9, 40, 41]
    assert table.block_tables[1].slot_mapping.gpu[:4].cpu().tolist() == [48, 49, 104, 105]

    query_start_loc.copy_(
        torch.tensor([0, 1, 3, 3, 3], dtype=torch.int32, device="cuda")
    )
    positions[:3].copy_(torch.tensor([3, 10, 11], dtype=torch.int64, device="cuda"))
    seq_lens.copy_(torch.tensor([7, 15, 0, 0], dtype=torch.int32, device="cuda"))
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
    assert table.block_tables[0].slot_mapping.gpu[:3].cpu().tolist() == [11, 42, 43]
    assert table.block_tables[1].slot_mapping.gpu[:3].cpu().tolist() == [51, 106, 107]
    arena.close()

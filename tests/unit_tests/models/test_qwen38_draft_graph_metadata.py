# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from _qwen38_cpu import ROOT, declarations

graph = declarations(
    ROOT / "vllm_fl/spec_decode/qwen38_draft_graph.py",
    "qwen38_graph_metadata_cpu",
    namespace={"copy": __import__("copy").copy},
    names={"fixed_qsa_metadata", "copy_qsa_metadata", "draft_graph_max_tokens"},
)


def source(order=(0, 1), actual=3):
    return SimpleNamespace(
        num_actual_tokens=actual,
        storage_block_size=400,
        compress_ratio=1,
        score_columns=None,
        block_table=torch.tensor([[9, 10], [20, 21]], dtype=torch.int32)[list(order)],
        seq_lens=torch.tensor([20, 11], dtype=torch.int32)[list(order)],
        query_start_loc=torch.tensor([0, 1, actual], dtype=torch.int32),
        slot_mapping=torch.tensor([3620, 8009, 8010], dtype=torch.int64),
        token_to_req=torch.tensor([0, 1, 1], dtype=torch.int32),
        logical_positions=torch.tensor([19, 9, 10], dtype=torch.int64),
    )


def test_fixed_metadata_updates_values_without_address_or_padding_leaks():
    src = source()
    dst = graph.fixed_qsa_metadata(src, 8, 4)
    names = (
        "block_table",
        "seq_lens",
        "query_start_loc",
        "slot_mapping",
        "token_to_req",
        "logical_positions",
    )
    pointers = {name: getattr(dst, name).data_ptr() for name in names}
    graph.copy_qsa_metadata(dst, src)
    assert dst.num_actual_tokens == 8
    assert torch.equal(dst.slot_mapping[:3], src.slot_mapping)
    assert torch.all(dst.slot_mapping[3:] == -1)
    assert torch.all(dst.logical_positions[3:] == -1)
    assert not torch.count_nonzero(dst.block_table[2:])
    swapped = source((1, 0))
    swapped.slot_mapping[:] = torch.tensor([8000, 3600, 3601])
    graph.copy_qsa_metadata(dst, swapped)
    assert torch.equal(dst.block_table[:2], swapped.block_table)
    assert torch.equal(dst.slot_mapping[:3], swapped.slot_mapping)
    assert torch.all(dst.query_start_loc[3:] == 3)
    assert pointers == {name: getattr(dst, name).data_ptr() for name in names}
    assert dst.block_table.data_ptr() != src.block_table.data_ptr()


def test_fixed_metadata_rejects_new_capacity_or_cache_geometry():
    src = source()
    dst = graph.fixed_qsa_metadata(src, 8, 4)
    src.num_actual_tokens = 9
    with pytest.raises(ValueError, match="capacity"):
        graph.copy_qsa_metadata(dst, src)
    src = source()
    src.block_table = torch.zeros(2, 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="width changed"):
        graph.copy_qsa_metadata(dst, src)


def test_fixed_metadata_rejects_changed_score_bucket():
    src = source()
    src.score_columns = 512
    dst = graph.fixed_qsa_metadata(src, 8, 4)
    graph.copy_qsa_metadata(dst, src)
    src.score_columns = 2048
    with pytest.raises(ValueError, match="score width"):
        graph.copy_qsa_metadata(dst, src)


@pytest.mark.parametrize(
    "drafts,requests,limit,expected",
    [
        (1, 3, 64, 6),
        (2, 3, 64, 9),
        (2, 64, 4096, 192),
        (2, 64, 128, 128),
        (3, 3, 64, 12),
        (1, 64, 4096, 128),
        (3, 64, 4096, 256),
        (3, 64, 192, 192),
    ],
)
def test_graph_token_capacity_covers_all_accepted_drafts(
    drafts, requests, limit, expected
):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=drafts),
        scheduler_config=SimpleNamespace(
            max_num_seqs=requests, max_num_batched_tokens=limit
        ),
    )
    assert graph.draft_graph_max_tokens(config) == expected


def test_k3_first_pass_and_later_steps_share_fixed_graph_metadata_without_stale_rows():
    src = source()
    src.num_actual_tokens = 12
    src.block_table = torch.tensor([[9, 10], [20, 21], [30, 31]], dtype=torch.int32)
    src.seq_lens = torch.tensor([403, 1603, 6403], dtype=torch.int32)
    src.query_start_loc = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    src.slot_mapping = torch.arange(12, dtype=torch.int64) + 100
    src.token_to_req = torch.arange(3, dtype=torch.int32).repeat_interleave(4)
    src.logical_positions = torch.tensor(
        [399, 400, 401, 402, 1599, 1600, 1601, 1602, 6399, 6400, 6401, 6402],
        dtype=torch.int64,
    )
    dst = graph.fixed_qsa_metadata(src, 16, 4)
    graph.copy_qsa_metadata(dst, src)
    assert dst.slot_mapping.tolist() == list(range(100, 112)) + [-1] * 4
    pointers = {
        name: getattr(dst, name).data_ptr()
        for name in (
            "block_table",
            "seq_lens",
            "query_start_loc",
            "slot_mapping",
            "token_to_req",
            "logical_positions",
        )
    }
    for step in (1, 2):
        src.num_actual_tokens = 3
        src.seq_lens = torch.tensor(
            [403 + step, 1603 + step, 6403 + step], dtype=torch.int32
        )
        src.query_start_loc = torch.arange(4, dtype=torch.int32)
        src.slot_mapping = torch.tensor([201, 301, 401], dtype=torch.int64) + step
        src.token_to_req = torch.arange(3, dtype=torch.int32)
        src.logical_positions = src.seq_lens.long() - 1
        graph.copy_qsa_metadata(dst, src)
        assert torch.equal(dst.logical_positions[:3], src.logical_positions)
        assert dst.logical_positions[3:].tolist() == [-1] * 13
        assert dst.slot_mapping[3:].tolist() == [-1] * 13
        assert dst.query_start_loc.tolist() == [0, 1, 2, 3, 3]
        assert dst.seq_lens[3:].tolist() == [0]
        assert not torch.count_nonzero(dst.block_table[3:])
        assert pointers == {name: getattr(dst, name).data_ptr() for name in pointers}

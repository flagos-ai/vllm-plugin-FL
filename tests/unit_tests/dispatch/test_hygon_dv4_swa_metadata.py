"""Exercise paged SWA metadata, invalid rows, batch changes and graph replay."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.amd.rocm import DeepseekV4ROCMAiterSparseSWAMetadata
from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.swa_decode_metadata import (
    try_build_swa_decode,
)


def make_builder(window, block_size):
    def empty(*shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device="cuda")
    return SimpleNamespace(
        num_speculative_tokens=0, window_size=window, block_size=block_size,
        token_to_req_indices=empty(256), is_valid_token=empty(256, dtype=torch.bool),
        decode_swa_indices=empty(256, 1, window), decode_swa_lens=empty(256),
        decode_swa_ragged_indices_buffer=empty(256 * window),
        decode_swa_ragged_indptr_buffer=empty(257))


def make_common(rows, block_size):
    loc = torch.arange(rows + 1, dtype=torch.int32)
    lens = torch.randint(0, 4097, (rows,), dtype=torch.int32, device="cuda")
    slots = torch.randint(-1, 4, (rows,), dtype=torch.int64, device="cuda")
    blocks = torch.randint(1, 10000, (rows, 4096 // block_size + 1), dtype=torch.int32, device="cuda")
    return SimpleNamespace(num_reqs=rows, num_actual_tokens=rows, max_query_len=1,
                           query_start_loc_cpu=loc, query_start_loc=loc.cuda(),
                           seq_lens=lens, slot_mapping=slots, block_table_tensor=blocks)


def check(builder, common, result):
    lens, slots, blocks = common.seq_lens.cpu(), common.slot_mapping.cpu(), common.block_table_tensor.cpu()
    expected_flat, expected_ptr, expected_lens = [], [0], []
    dense = result.decode_swa_indices.cpu().reshape(common.num_reqs, builder.window_size)
    for row in range(common.num_reqs):
        seq_len = int(lens[row])
        values = []
        if slots[row] >= 0:
            for pos in range(max(0, seq_len - builder.window_size), seq_len):
                values.append(int(blocks[row, pos // builder.block_size]) * builder.block_size + pos % builder.block_size)
        assert dense[row, :len(values)].tolist() == values
        expected_lens.append(len(values))
        expected_flat.extend(values)
        expected_ptr.append(len(expected_flat))
    assert result.decode_swa_lens.cpu().tolist() == expected_lens
    assert result.decode_swa_ragged_indptr.cpu().tolist() == expected_ptr
    assert result.decode_swa_ragged_indices[:len(expected_flat)].cpu().tolist() == expected_flat
    assert result.is_valid_token.cpu().tolist() == (slots >= 0).tolist()
    assert result.token_to_req_indices.cpu().tolist() == list(range(common.num_reqs))
    assert torch.count_nonzero(builder.decode_swa_lens[common.num_reqs:]).item() == 0


@pytest.mark.parametrize("window", [7, 128, 257])
@pytest.mark.parametrize("block_size", [64, 256])
def test_batch_changes(window, block_size):
    builder = make_builder(window, block_size)
    for rows in [64, 1, 3, 256, 65]:
        common = make_common(rows, block_size)
        result = try_build_swa_decode(builder, common, DeepseekV4ROCMAiterSparseSWAMetadata)
        check(builder, common, result)
        assert result.decode_swa_ragged_indices.data_ptr() == builder.decode_swa_ragged_indices_buffer.data_ptr()


def test_graph_replay_changes_pages_lengths_and_validity():
    builder = make_builder(128, 256)
    common = make_common(64, 256)
    try_build_swa_decode(builder, common, DeepseekV4ROCMAiterSparseSWAMetadata)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = try_build_swa_decode(builder, common, DeepseekV4ROCMAiterSparseSWAMetadata)
    for n in [0, 13, 128, 1025, 4096]:
        common.seq_lens.fill_(n)
        common.block_table_tensor.add_(3)
        common.slot_mapping.neg_()
        graph.replay()
        check(builder, common, result)


def test_mixed_and_speculative_batches_fall_back():
    builder = make_builder(128, 256)
    common = make_common(4, 256)
    common.query_start_loc_cpu[1] = 0
    assert try_build_swa_decode(builder, common, DeepseekV4ROCMAiterSparseSWAMetadata) is None
    common = make_common(4, 256)
    builder.num_speculative_tokens = 1
    assert try_build_swa_decode(builder, common, DeepseekV4ROCMAiterSparseSWAMetadata) is None

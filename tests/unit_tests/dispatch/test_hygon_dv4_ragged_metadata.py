"""Check ragged semantics and mutable inputs with persistent graph addresses."""

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.ragged_metadata import (
    build_decode_ragged_into,
)


def check(dense, lengths, flat, indptr):
    dense = dense.cpu().reshape(dense.shape[0], -1).to(torch.int32)
    counts = lengths.cpu().clamp(0, dense.shape[1]).tolist()
    expected_ptr = [0]
    values = []
    for row, count in zip(dense, counts):
        values.extend(row[:count].tolist())
        expected_ptr.append(len(values))
    assert indptr.cpu().tolist() == expected_ptr
    assert flat[:len(values)].cpu().tolist() == values


@pytest.mark.parametrize("rows,width", [(1, 1), (3, 7), (64, 128), (65, 257), (256, 129)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_ragged_clamp_and_pack(rows, width, dtype, strided):
    torch.manual_seed(103)
    storage = torch.randint(-1, 100000, (rows, width * 2), dtype=dtype, device="cuda")
    dense = storage[:, ::2] if strided else storage[:, :width].contiguous()
    lengths = torch.randint(-5, width + 20, (rows,), dtype=torch.int32, device="cuda")
    flat_buffer = torch.full((rows * width + 19,), -123, dtype=torch.int32, device="cuda")
    ptr_buffer = torch.full((rows + 1 + 19,), -123, dtype=torch.int32, device="cuda")
    flat, ptr = build_decode_ragged_into(dense, lengths, flat_buffer, ptr_buffer)
    check(dense, lengths, flat, ptr)
    assert flat.data_ptr() == flat_buffer.data_ptr()
    assert ptr.data_ptr() == ptr_buffer.data_ptr()
    assert (flat_buffer[rows * width:] == -123).all().item()
    assert (ptr_buffer[rows + 1:] == -123).all().item()


def test_graph_replay_updates_lengths_and_indices():
    dense = torch.arange(64 * 128, dtype=torch.int32, device="cuda").reshape(64, 1, 128)
    lengths = torch.full((64,), 128, dtype=torch.int32, device="cuda")
    flat_buffer = torch.empty(64 * 128, dtype=torch.int32, device="cuda")
    ptr_buffer = torch.empty(65, dtype=torch.int32, device="cuda")
    build_decode_ragged_into(dense, lengths, flat_buffer, ptr_buffer)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flat, ptr = build_decode_ragged_into(dense, lengths, flat_buffer, ptr_buffer)
    for n in [0, 3, 128, 200, -5]:
        lengths.fill_(n)
        dense.add_(1)
        graph.replay()
        check(dense, lengths, flat, ptr)


def test_large_batch_falls_back():
    dense = torch.zeros((257, 128), device="cuda", dtype=torch.int32)
    lengths = torch.ones(257, device="cuda", dtype=torch.int32)
    assert build_decode_ragged_into(dense, lengths, dense.flatten(), lengths) is None

"""Indexer gather must not read GPU lengths on the host; replay reads new lengths."""
import pytest
import torch

import flag_gems
from vllm.platforms import current_platform
from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer import (
    cp_gather_indexer_k_quant_cache_hygon as gather,
)


def inputs(lengths):
    total = sum(lengths)
    width = max(total, 1)
    blocks = len(lengths) * width
    dtype = current_platform.fp8_dtype()
    values = (torch.arange(blocks * 128).reshape(blocks, 128) % 7).to(dtype)
    scales = torch.arange(1, blocks + 1, dtype=torch.float32)
    cache = torch.cat((values.view(torch.uint8), scales[:, None].view(torch.uint8)), 1)
    table = torch.arange(blocks, dtype=torch.int32).reshape(len(lengths), width)
    cu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).int()
    selected = torch.cat([table[i, :n].long() for i, n in enumerate(lengths)])
    expected = values.float()[selected], scales[selected]
    return (cache[:, None, :].cuda(),
            torch.empty(total, 128, dtype=dtype, device="cuda"),
            torch.empty(total, dtype=torch.float32, device="cuda"),
            table.cuda(), cu.cuda()), expected


@pytest.mark.parametrize("lengths", [(8,), (3, 0, 5), (0, 8, 0), (4096,)])
def test_gather_capture_and_values(lengths):
    args, expected = inputs(lengths)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with flag_gems.use_gems(include=["repeat_interleave_self_tensor", "searchsorted"]):
        with torch.cuda.stream(stream):
            gather(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gather(*args)
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(args[1].float().cpu(), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(args[2].cpu(), expected[1], rtol=0, atol=0)


def test_gather_graph_reads_changed_sequence_lengths():
    args, _ = inputs((4, 4))
    selected = torch.tensor([0, 1, 8, 9, 10, 11, 12, 13])
    cache_cpu = args[0].cpu().squeeze(1)
    expected_k = cache_cpu[:, :128].contiguous().view(args[1].dtype).float()[selected]
    expected_s = cache_cpu[:, 128:].contiguous().view(torch.float32).flatten()[selected]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with flag_gems.use_gems(include=["repeat_interleave_self_tensor", "searchsorted"]):
        with torch.cuda.stream(stream):
            gather(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gather(*args)
        args[4].copy_(torch.tensor([0, 2, 8], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(args[1].float().cpu(), expected_k, rtol=0, atol=0)
    torch.testing.assert_close(args[2].cpu(), expected_s, rtol=0, atol=0)

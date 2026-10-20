# Copyright (c) 2026 BAAI. All rights reserved.

"""Real Hopper indexer/WO-A numerics, including packed pages and graph replay."""

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA DSV4.1 bridge"
)


@pytest.fixture
def ops():
    from vllm_fl.ops import nvidia_dsv41_indexer, nvidia_dsv41_ops

    assert nvidia_dsv41_ops.register_nvidia_dsv41_ops()
    return nvidia_dsv41_indexer


def _fp8(shape, seed):
    return (
        torch.randn(shape, generator=torch.Generator().manual_seed(seed))
        .clamp(-3, 3)
        .to(torch.float8_e4m3fn)
        .cuda()
    )


def _packed_cache(keys, scales, block):
    pages = keys.shape[0] // block
    cache = torch.empty(pages, block * 132, dtype=torch.uint8, device="cuda")
    cache[:, : block * 128].copy_(keys.view(torch.uint8).reshape(pages, -1))
    cache[:, block * 128 :].copy_(scales.reshape(pages, block).view(torch.uint8))
    return cache.view(pages, block, 1, 132)


def _logits_reference(q, keys, scales, weights):
    dot = torch.einsum("mhd,nd->mhn", q.float().cpu(), keys.float().cpu())
    return (
        torch.relu(dot * scales.cpu()[None, None, :]) * weights.cpu()[:, :, None]
    ).sum(1)


@pytest.mark.gpu
@pytest.mark.parametrize("clean", [True, False])
def test_prefill_fp8_indexer_and_topk_beyond_budget(ops, clean):
    q, keys = _fp8((4, 32, 128), 1), _fp8((1100, 128), 2)
    scales = torch.linspace(0.01, 0.03, 1100, device="cuda")
    weights = torch.linspace(-0.5, 1, 128, device="cuda").reshape(4, 32)
    starts = torch.tensor([0, 8, 32, 511], dtype=torch.int32, device="cuda")
    ends = torch.tensor([0, 25, 1061, 1065], dtype=torch.int32, device="cuda")
    actual = ops.fp8_mqa_logits((q, None), (keys, scales), weights, starts, ends, clean)
    reference = _logits_reference(q, keys, scales, weights)
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        torch.testing.assert_close(
            actual[row, start:end].cpu(),
            reference[row, start:end],
            rtol=3e-4,
            atol=3e-4,
        )
        if clean:
            assert torch.isneginf(actual[row, :start]).all()
            assert torch.isneginf(actual[row, end:]).all()
    selected = torch.full((4, 512), -77, dtype=torch.int32, device="cuda")
    torch.ops._C.top_k_per_row_prefill(
        actual, starts, ends, selected, 4, *actual.stride(), 512
    )
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        count = min(512, end - start)
        indices = selected[row, :count].cpu().long()
        assert (selected[row, count:] == -1).all()
        assert indices.unique().numel() == count
        assert ((indices >= 0) & (indices < end - start)).all()
        obtained = actual[row].cpu()[indices + start].sort().values
        expected = actual[row, start:end].cpu().topk(count).values.sort().values
        torch.testing.assert_close(obtained, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("block", [32, 64, 128])
def test_paged_indexer_and_gather_packed_pages(ops, block):
    pages, max_len = 12, block * 6
    keys = _fp8((pages * block, 128), 8)
    scales = torch.linspace(0.01, 0.05, pages * block, device="cuda")
    cache = _packed_cache(keys, scales, block)
    untouched = cache.clone()
    table_storage = torch.tensor(
        [[8, 1, 5, 2, 10, 0, -77], [6, 11, 3, 9, 4, 7, -77]],
        dtype=torch.int32,
        device="cuda",
    )
    table = table_storage[:, :6]
    lengths = torch.tensor(
        [[max_len - 7], [block + 3]], dtype=torch.int32, device="cuda"
    )
    q = _fp8((2, 1, 32, 128), 9)
    weights = torch.linspace(-0.3, 0.7, 64, device="cuda").reshape(2, 32)
    metadata = ops.get_paged_mqa_logits_metadata(lengths, block, 132)
    assert metadata.dtype == torch.int32 and metadata.shape == (133, 2)
    actual = ops.fp8_paged_mqa_logits(
        (q, None), cache, weights, lengths, table, metadata, max_len, clean_logits=True
    )
    selected_keys, selected_scales = [], []
    for row, length in enumerate(lengths.flatten().tolist()):
        logical = torch.cat(
            [keys[int(p) * block : (int(p) + 1) * block] for p in table[row].tolist()]
        )[:length]
        logical_scale = torch.cat(
            [scales[int(p) * block : (int(p) + 1) * block] for p in table[row].tolist()]
        )[:length]
        expected = _logits_reference(
            q[row], logical, logical_scale, weights[row : row + 1]
        )
        torch.testing.assert_close(
            actual[row, :length].cpu(), expected[0], rtol=3e-4, atol=3e-4
        )
        assert torch.isneginf(actual[row, length:]).all()
        selected_keys.append(logical)
        selected_scales.append(logical_scale)
    total = int(lengths.sum())
    dst = torch.empty(total, 136, dtype=torch.float8_e4m3fn, device="cuda")[:, :128]
    dst_scale = torch.empty(total, 4, dtype=torch.uint8, device="cuda")
    cumulative = torch.tensor(
        [0, lengths[0].item(), total], dtype=torch.int32, device="cuda"
    )
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        cache, dst, dst_scale, table, cumulative
    )
    torch.testing.assert_close(
        dst.float(), torch.cat(selected_keys).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        dst_scale.view(torch.float32).flatten(),
        torch.cat(selected_scales),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(cache, untouched, rtol=0, atol=0)


@pytest.mark.gpu
def test_paged_indexer_graph_replays_updated_lengths(ops):
    block, max_len = 64, 1024
    keys, q = _fp8((max_len, 128), 10), _fp8((1, 1, 32, 128), 11)
    scales = torch.full((max_len,), 0.02, device="cuda")
    cache = _packed_cache(keys, scales, block)
    weights = torch.ones(1, 32, device="cuda") / 32
    lengths = torch.tensor([[max_len]], dtype=torch.int32, device="cuda")
    table = torch.arange(max_len // block, dtype=torch.int32, device="cuda").view(1, -1)
    metadata = ops.get_paged_mqa_logits_metadata(lengths, block, 132)

    def run():
        return ops.fp8_paged_mqa_logits(
            (q, None),
            cache,
            weights,
            lengths,
            table,
            metadata,
            max_len,
            clean_logits=True,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    lengths.fill_(513)
    q.copy_(_fp8(q.shape, 12))
    graph.replay()
    expected = _logits_reference(q[0], keys[:513], scales[:513], weights)
    torch.testing.assert_close(actual[0, :513].cpu(), expected[0], rtol=3e-4, atol=3e-4)
    assert torch.isneginf(actual[0, 513:]).all()


@pytest.mark.gpu
@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("heads, width, output_dim", [(2, 256, 256), (1, 4096, 1024)])
def test_wo_a_fp8_einsum_out_buffer(ops, tokens, heads, width, output_dim):
    x, y = _fp8((tokens, heads, width), 13), _fp8((heads, output_dim, width), 14)
    xs = torch.linspace(
        0.01, 0.03, tokens * heads * (width // 128), device="cuda"
    ).reshape(tokens, heads, width // 128)
    ys = torch.linspace(
        0.01, 0.03, heads * (output_dim // 128) * (width // 128), device="cuda"
    ).reshape(heads, output_dim // 128, width // 128)
    out = torch.full(
        (tokens, heads, output_dim), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    ptr = out.data_ptr()
    ops.fp8_einsum("bhr,hdr->bhd", (x, xs), (y, ys), out)
    xd = x.float().cpu() * xs.cpu().repeat_interleave(128, -1)
    yd = y.float().cpu() * ys.cpu().repeat_interleave(128, 1).repeat_interleave(128, 2)
    expected = torch.einsum("bhr,hdr->bhd", xd, yd).bfloat16()
    torch.testing.assert_close(out.cpu(), expected, rtol=0.01, atol=1e-4)
    assert out.data_ptr() == ptr


@pytest.mark.gpu
def test_indexer_rejects_fp4_and_varlen(ops):
    q = _fp8((1, 1, 32, 128), 20)
    with pytest.raises(NotImplementedError, match="varlen"):
        ops.fp8_paged_mqa_logits(
            (q, None), None, None, None, None, None, 1024, indices=torch.tensor([0])
        )
    with pytest.raises(NotImplementedError, match="separate scales"):
        ops.fp8_mqa_logits(
            (q[0], torch.ones(1)), (q[0], torch.ones(1)), None, None, None
        )


@pytest.mark.gpu
def test_wo_a_fp8_einsum_graph_replay(ops):
    x, y = _fp8((1, 1, 4096), 23), _fp8((1, 1024, 4096), 24)
    xs = torch.full((1, 1, 32), 0.02, dtype=torch.float32, device="cuda")
    ys = torch.full((1, 8, 32), 0.03, dtype=torch.float32, device="cuda")
    out = torch.empty((1, 1, 1024), dtype=torch.bfloat16, device="cuda")

    def run():
        ops.fp8_einsum("bhr,hdr->bhd", (x, xs), (y, ys), out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(_fp8(x.shape, 25))
    xs.fill_(0.04)
    graph.replay()
    xd = x.float().cpu() * xs.cpu().repeat_interleave(128, -1)
    yd = y.float().cpu() * ys.cpu().repeat_interleave(128, 1).repeat_interleave(128, 2)
    expected = torch.einsum("bhr,hdr->bhd", xd, yd).bfloat16()
    torch.testing.assert_close(out.cpu(), expected, rtol=0.01, atol=1e-4)

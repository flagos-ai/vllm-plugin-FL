# Copyright (c) 2026 BAAI. All rights reserved.

"""Sparse prefill/decode math, real Gems scheduling state, and graph replay."""

from types import SimpleNamespace

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "nvidia", reason="NVIDIA DSV4.1 MLA")


def _bf16(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).bfloat16()


def _reference(
    q,
    keys,
    ids,
    lengths,
    scale,
    sink,
    extra_keys=None,
    extra_ids=None,
    extra_lengths=None,
):
    q, keys, ids, lengths, sink = [x.cpu() for x in (q, keys, ids, lengths, sink)]
    outputs, lses = [], []
    for row in range(q.shape[0]):
        selected = ids[row].flatten()[: int(lengths[row])].long()
        selected = selected[(selected >= 0) & (selected < keys.shape[0])]
        values = keys[selected].float()
        if extra_keys is not None:
            extra = extra_ids[row].cpu().flatten()[: int(extra_lengths[row])].long()
            extra = extra[(extra >= 0) & (extra < extra_keys.shape[0])]
            values = torch.cat((values, extra_keys.cpu()[extra].float()), dim=0)
        if not values.shape[0]:
            outputs.append(torch.zeros(q.shape[1], 512))
            # FlashMLA returns +inf for an empty row so downstream normalized
            # merges give that row zero weight. Verified against native 0.32.
            lses.append(torch.full((q.shape[1],), float("inf")))
            continue
        scores = q[row].float() @ values.T * scale
        lse = torch.logsumexp(scores, -1)
        probs = torch.softmax(scores, -1)
        output = probs @ values
        output *= torch.sigmoid(lse - sink)[:, None]
        outputs.append(output)
        lses.append(lse)
    return torch.stack(outputs).bfloat16(), torch.stack(lses)


def _cache_and_decoded_keys(pages, block, seed):
    original = _bf16((pages * block, 512), seed)
    groups = original[:, :448].float().reshape(-1, 7, 64)
    exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
    quantized = (
        (groups * torch.exp2(-exponent)[..., None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    decoded = torch.cat(
        (
            (quantized.float() * torch.exp2(exponent)[..., None]).reshape(-1, 448),
            original[:, 448:].float(),
        ),
        -1,
    ).bfloat16()
    stride = ((block * 584 + 575) // 576) * 576
    storage = torch.full((pages, stride), 193, dtype=torch.uint8)
    for page in range(pages):
        a, b = page * block, (page + 1) * block
        data = torch.cat(
            (
                quantized[a:b].view(torch.uint8).reshape(block, 448),
                original[a:b, 448:].contiguous().view(torch.uint8),
            ),
            -1,
        )
        storage[page, : block * 576] = data.flatten()
        scales = torch.cat(
            ((exponent[a:b] + 127).byte(), torch.zeros(block, 1, dtype=torch.uint8)), -1
        )
        storage[page, block * 576 : block * 584] = scales.flatten()
    cache = storage.cuda()[:, : block * 584].view(pages, block, 1, 584)
    return cache, decoded


@pytest.mark.gpu
@pytest.mark.parametrize("heads", [64, 128])
def test_sparse_prefill_out_sink_lengths_and_invalid_indices(heads):
    from vllm_fl.dispatch.backends.flaggems.impl.dsv41_mla import flash_mla_sparse_fwd

    q, kv = _bf16((3, heads, 512), 1), _bf16((160, 512), 2)
    indices = torch.arange(128, dtype=torch.int32).repeat(3, 1).unsqueeze(1)
    indices[0, 0, 3], indices[1, 0, 4], indices[2] = -1, 200, -1
    lengths = torch.tensor([67, 99, 0], dtype=torch.int32)
    sink = torch.linspace(-2, 3, heads)
    sink[0], sink[1] = float("-inf"), float("inf")
    expected, expected_lse = _reference(q, kv, indices, lengths, 0.05, sink)
    storage = torch.empty(3, heads, 520, dtype=torch.bfloat16, device="cuda")
    output = storage[..., :512]
    result, _, lse = flash_mla_sparse_fwd(
        q.cuda(),
        kv[:, None].cuda(),
        indices.cuda(),
        0.05,
        attn_sink=sink.cuda(),
        topk_length=lengths.cuda(),
        out=output,
    )
    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result.cpu(), expected, rtol=0.02, atol=0.008)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("extra", [False, True])
def test_sparse_decode_two_caches_and_real_metadata(extra):
    from vllm_fl.dispatch.backends.flaggems.impl import dsv41_mla as ops

    cache, keys = _cache_and_decoded_keys(3, 64, 5)
    extra_cache, extra_keys = _cache_and_decoded_keys(3, 32, 6)
    q = _bf16((2, 64, 512), 7)
    ids = torch.arange(64, dtype=torch.int32).repeat(2, 1).unsqueeze(1)
    ids[1] += 80
    ids[0, 0, 2] = -1
    extra_ids = torch.arange(64, dtype=torch.int32).flip(0).repeat(2, 1).unsqueeze(1)
    lengths = torch.tensor([47, 64], dtype=torch.int32)
    extra_lengths = torch.tensor([17, 33], dtype=torch.int32)
    sink = torch.linspace(-1, 2, 64)
    sink[0], sink[1] = float("-inf"), float("inf")
    kwargs = (
        {
            "extra_keys": extra_keys,
            "extra_ids": extra_ids,
            "extra_lengths": extra_lengths,
        }
        if extra
        else {}
    )
    expected, expected_lse = _reference(q, keys, ids, lengths, 0.05, sink, **kwargs)
    builder = SimpleNamespace(_layer_types={"swaonly", "c1a", "c2a"})
    metadata = ops.build_tile_scheduler(builder, 2)
    assert metadata["swaonly"] is not metadata["c1a"]
    assert all(value is None for value in ops.build_tile_scheduler(builder, 0).values())
    state = metadata["c1a"]
    assert not state.have_initialized
    output = torch.empty(2, 1, 64, 520, dtype=torch.bfloat16, device="cuda")[..., :512]
    kwargs = (
        {
            "extra_k_cache": extra_cache,
            "extra_indices_in_kvcache": extra_ids.cuda(),
            "extra_topk_length": extra_lengths.cuda(),
        }
        if extra
        else {}
    )
    result, lse = ops.flash_mla_with_kvcache(
        q.cuda().unsqueeze(1),
        cache,
        None,
        None,
        512,
        state,
        softmax_scale=0.05,
        is_fp8_kvcache=True,
        indices=ids.cuda(),
        attn_sink=sink.cuda(),
        topk_length=lengths.cuda(),
        out=output,
        **kwargs,
    )
    assert result.data_ptr() == output.data_ptr() and state.have_initialized
    torch.testing.assert_close(result[:, 0].cpu(), expected, rtol=0.02, atol=0.008)
    torch.testing.assert_close(lse[..., 0].cpu(), expected_lse, rtol=3e-4, atol=3e-4)


@pytest.mark.gpu
def test_sparse_decode_cuda_graph_replays_indices():
    from vllm_fl.dispatch.backends.flaggems.impl import dsv41_mla as ops

    cache, keys = _cache_and_decoded_keys(3, 64, 11)
    q = _bf16((1, 64, 512), 12).cuda()
    ids = torch.arange(64, dtype=torch.int32, device="cuda").view(1, 1, 64)
    lengths = torch.tensor([53], dtype=torch.int32, device="cuda")
    sink = torch.zeros(64, device="cuda")
    state = ops.get_mla_metadata()[0]
    output = torch.empty(1, 1, 64, 512, dtype=torch.bfloat16, device="cuda")

    def run():
        return ops.flash_mla_with_kvcache(
            q.unsqueeze(1),
            cache,
            None,
            None,
            512,
            state,
            softmax_scale=0.05,
            is_fp8_kvcache=True,
            indices=ids,
            attn_sink=sink,
            topk_length=lengths,
            out=output,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    ids.add_(80)
    lengths.fill_(39)
    graph.replay()
    expected, _ = _reference(q, keys, ids, lengths, 0.05, sink)
    torch.testing.assert_close(output[:, 0].cpu(), expected, rtol=0.02, atol=0.008)

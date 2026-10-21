# Copyright (c) 2026 BAAI. All rights reserved.

"""Real upstream Triton query RoPE/FP8 numerics without bundled FlashAttention."""

import pytest
import torch

from tests.utils.device_utils import get_backend, requires_cuda_fp8

pytestmark = pytest.mark.skipif(get_backend() != "nvidia", reason="NVIDIA DSV4.1 query")


def query_reference(positions, query, cache, weights, softmax_scale, head_scale):
    # Independent CPU oracle: trailing GPT-J rotation, BF16 materialization,
    # power-of-two per-head FP8 scale, then its fold into scoring weights.
    q = query.detach().float().cpu().clone()
    angles = cache.detach().float().cpu()[positions.cpu()]
    cos, sin = angles.chunk(2, dim=-1)
    even, odd = q[..., 64::2].clone(), q[..., 65::2].clone()
    q[..., 64::2] = (even * cos[:, None] - odd * sin[:, None]).bfloat16().float()
    q[..., 65::2] = (odd * cos[:, None] + even * sin[:, None]).bfloat16().float()
    scale = torch.exp2(torch.ceil(torch.log2(q.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    fp8 = (q / scale[..., None]).to(torch.float8_e4m3fn)
    folded = weights.detach().float().cpu() * scale * softmax_scale * head_scale
    return fp8, folded


def query_inputs(tokens, heads, strided=False, weights_dtype=torch.bfloat16):
    rng = torch.Generator().manual_seed(920 + tokens + heads)
    q = torch.randn(tokens, heads, 128, generator=rng).bfloat16()
    if tokens:
        q[0, 0].zero_()  # Covers the scale floor without invalid values.
    weights = torch.randn(tokens, heads, generator=rng).to(weights_dtype)
    angles = torch.randn(1024, 32, generator=rng)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
    positions = (torch.arange(tokens, dtype=torch.int64) * 17 + 3) % 1024
    if strided:
        q_storage = torch.empty(tokens, heads, 144, dtype=q.dtype, device="cuda")
        q_storage[..., :128].copy_(q)
        q = q_storage[..., :128]
        w_storage = torch.empty(tokens, heads + 7, dtype=weights.dtype, device="cuda")
        w_storage[:, :heads].copy_(weights)
        weights = w_storage[:, :heads]
        cache_storage = torch.empty(1024, 72, device="cuda")
        cache_storage[:, :64].copy_(cache)
        cache = cache_storage[:, :64]
    return positions.cuda(), q.cuda(), cache.cuda(), weights.cuda()


def assert_query_equal(actual, expected):
    # FP8 is checked as exact represented values, including its rounding.
    torch.testing.assert_close(
        actual[0].float().cpu(), expected[0].float(), rtol=0, atol=0
    )
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=2e-6, atol=1e-8)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "tokens,heads,strided,weights_dtype",
    [
        (0, 32, False, torch.bfloat16),
        pytest.param(1, 32, False, torch.bfloat16, marks=requires_cuda_fp8),
        pytest.param(7, 32, True, torch.bfloat16, marks=requires_cuda_fp8),
        pytest.param(33, 64, True, torch.float32, marks=requires_cuda_fp8),
        pytest.param(513, 32, False, torch.float32, marks=requires_cuda_fp8),
    ],
)
def test_query_rope_fp8_matches_cpu(tokens, heads, strided, weights_dtype):
    from vllm_fl.ops.nvidia_dsv41_indexer import fused_indexer_q_rope_quant

    args = query_inputs(tokens, heads, strided, weights_dtype)
    scales = (128**-0.5, heads**-0.5)
    actual = fused_indexer_q_rope_quant(*args, *scales)
    expected = query_reference(*args, *scales)
    assert_query_equal(actual, expected)
    assert actual[0].dtype == torch.float8_e4m3fn
    assert actual[1].dtype == torch.float32


@pytest.mark.gpu
@requires_cuda_fp8
def test_query_graph_replay_reads_updated_values_positions_and_weights():
    from vllm_fl.ops.nvidia_dsv41_indexer import fused_indexer_q_rope_quant

    args = query_inputs(7, 32, True)
    scales = (128**-0.5, 32**-0.5)
    for _ in range(2):
        fused_indexer_q_rope_quant(*args, *scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_indexer_q_rope_quant(*args, *scales)
    with torch.no_grad():
        args[0].add_(5)
        args[1].mul_(0.5)
        args[3].mul_(-0.25)
    graph.replay()
    torch.cuda.synchronize()
    assert_query_equal(actual, query_reference(*args, *scales))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kwargs", [{"use_fp4": True}, {"weights_out_dtype": torch.bfloat16}]
)
def test_query_rejects_unsupported_format(kwargs):
    from vllm_fl.ops.nvidia_dsv41_indexer import fused_indexer_q_rope_quant

    with pytest.raises(NotImplementedError, match="FP8 and FP32"):
        fused_indexer_q_rope_quant(*query_inputs(1, 32), 0.125, 0.25, **kwargs)

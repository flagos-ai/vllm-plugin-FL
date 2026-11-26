# Copyright (c) 2026 BAAI. All rights reserved.
"""C550 RoPE values and mutation for packed QKV and graph replay."""

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_backend() != "metax" or not torch.cuda.is_available(),
        reason="MetaX GPU and MACA runtime required",
    ),
]


@pytest.fixture(autouse=True)
def rotary_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    config.compilation_config.custom_ops = ["all"]
    with set_current_vllm_config(config):
        yield


def _reference(value, cache, positions, rotary_dim, neox):
    value = value.float()
    c, s = cache.float()[positions].chunk(2, -1)
    if neox:
        c, s = torch.cat((c, c), -1), torch.cat((s, s), -1)
        left, right = value[..., :rotary_dim].chunk(2, -1)
        rotated = torch.cat((-right, left), -1)
    else:
        c, s = c.repeat_interleave(2, -1), s.repeat_interleave(2, -1)
        rotated = torch.stack(
            (-value[..., 1:rotary_dim:2], value[..., :rotary_dim:2]), -1
        ).flatten(-2)
    result = value.clone()
    result[..., :rotary_dim] = (
        value[..., :rotary_dim] * c[:, None] + rotated * s[:, None]
    )
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,heads,kv_heads,head_size,rotary_dim,neox,packed",
    [
        (1, 10, 10, 128, 128, True, True),
        (551, 10, 10, 128, 128, True, True),
        (1917, 10, 10, 128, 128, True, True),
        (1917, 8, 2, 128, 64, True, True),
        (551, 8, 2, 64, 64, True, False),
        (551, 10, 10, 128, 128, False, True),
        (1, 2, 2, 256, 256, True, False),
    ],
)
@torch.inference_mode()
def test_rotary_entrypoint_preserves_values_and_storage(
    dtype, tokens, heads, kv_heads, head_size, rotary_dim, neox, packed
):
    from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

    rotary = RotaryEmbeddingFL(head_size, rotary_dim, 2048, 10000, neox, dtype).cuda()
    generator = torch.Generator().manual_seed(817)
    source = torch.randn(
        tokens, (heads + 2 * kv_heads) * head_size, generator=generator
    ).to(dtype)
    positions = torch.arange(tokens).roll(3)
    outputs = []
    for _ in range(2):
        owner = source.cuda()
        q = owner[:, : heads * head_size]
        k = owner[:, heads * head_size : (heads + kv_heads) * head_size]
        if not packed:
            q, k = q.contiguous(), k.contiguous()
        q_before, k_before = q.cpu(), k.cpu()
        expected = [
            _reference(
                value.view(tokens, h, head_size),
                rotary.cos_sin_cache.cpu(),
                positions,
                rotary_dim,
                neox,
            )
            .reshape_as(value)
            .to(dtype)
            for value, h in ((q_before, heads), (k_before, kv_heads))
        ]
        actual = rotary(positions.cuda(), q, k)
        torch.cuda.synchronize()
        for result, original, reference in zip(actual, (q, k), expected):
            # The upstream wrapper concatenates partial rotation with its tail.
            # Reshape may normalize the stride of an unused singleton axis.
            if rotary_dim == head_size:
                assert result.data_ptr() == original.data_ptr()
                for size, actual_stride, original_stride in zip(
                    original.shape, result.stride(), original.stride()
                ):
                    assert size == 1 or actual_stride == original_stride
            torch.testing.assert_close(result.cpu(), reference, atol=0.004, rtol=0.01)
            torch.testing.assert_close(original.cpu(), reference, atol=0.004, rtol=0.01)
        torch.testing.assert_close(
            owner[:, (heads + kv_heads) * head_size :].cpu(),
            source[:, (heads + kv_heads) * head_size :],
            atol=0,
            rtol=0,
        )
        outputs.append(tuple(t.cpu() for t in actual))
    for first, second in zip(*outputs):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rotary_dim", [64, 128])
@torch.inference_mode()
def test_rotary_graph_replay_uses_new_positions_and_qkv(dtype, rotary_dim):
    from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

    tokens, heads, head_size = 17, 10, 128
    rotary = RotaryEmbeddingFL(head_size, rotary_dim, 2048, 10000, True, dtype).cuda()
    owner = torch.empty(tokens, 3 * heads * head_size, device="cuda", dtype=dtype)
    q, k = owner[:, :1280], owner[:, 1280:2560]
    positions = torch.arange(tokens, device="cuda")
    owner.fill_(1)
    rotary(positions, q, k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rotary(positions, q, k)
    for epoch in (1, 2):
        source = (
            torch.linspace(-epoch, epoch, owner.numel()).reshape(owner.shape).to(dtype)
        )
        pos = torch.arange(tokens) + epoch * 19
        expected = [
            _reference(
                value.view(tokens, heads, head_size),
                rotary.cos_sin_cache.cpu(),
                pos,
                rotary_dim,
                True,
            )
            .reshape_as(value)
            .to(dtype)
            for value in (source[:, :1280], source[:, 1280:2560])
        ]
        owner.copy_(source)
        positions.copy_(pos)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(q.cpu(), expected[0], atol=0.004, rtol=0.01)
        torch.testing.assert_close(k.cpu(), expected[1], atol=0.004, rtol=0.01)
        torch.testing.assert_close(
            owner[:, 2560:].cpu(), source[:, 2560:], atol=0, rtol=0
        )

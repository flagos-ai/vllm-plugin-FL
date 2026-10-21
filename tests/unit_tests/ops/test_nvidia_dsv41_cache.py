# Copyright (c) 2026 BAAI. All rights reserved.

"""Check the actual Hopper cache bytes, query padding, and graph updates."""

import pytest
import torch

from tests.utils.device_utils import get_backend, requires_cuda_fp8

pytestmark = pytest.mark.skipif(get_backend() != "nvidia", reason="NVIDIA DSV4.1 cache")


def _inputs(block=16):
    generator = torch.Generator().manual_seed(713)
    q = torch.randn(5, 3, 512, generator=generator).bfloat16().cuda()
    kv = torch.randn(5, 512, generator=generator).bfloat16().cuda()
    kv[1, :64] = 0
    slots = torch.tensor(
        [-1, 0, block + 3, 2 * block - 1], dtype=torch.int64, device="cuda"
    )
    positions = torch.tensor([2, 0, 3, 1, 4], dtype=torch.int64, device="cuda")
    rope = torch.empty(5, 64, device="cuda")
    rope[:, :32] = 0.75
    rope[:, 32:] = 0.5
    rope[0, :32], rope[0, 32:] = 1, 0
    page_stride = ((block * 584 + 575) // 576) * 576
    storage = torch.full((3, page_stride), 165, dtype=torch.uint8, device="cuda")
    return q, kv, storage, slots, positions, rope


def _rotate(tensor, positions, rope):
    result = tensor.float().clone()
    phase = rope.float()[positions]
    if tensor.ndim == 3:
        phase = phase.unsqueeze(1)
    even, odd = result[..., 448::2].clone(), result[..., 449::2].clone()
    result[..., 448::2] = even * phase[..., :32] - odd * phase[..., 32:]
    result[..., 449::2] = even * phase[..., 32:] + odd * phase[..., :32]
    return result


def _reference(q, kv, storage, slots, positions, rope, block, padded, norm, rotate):
    q, kv, storage, slots, positions, rope = [
        t.cpu() for t in (q, kv, storage, slots, positions, rope)
    ]
    query = q.float()
    if norm:
        query = query * torch.rsqrt(query.square().mean(-1, keepdim=True) + 1e-6)
    if rotate:
        query = _rotate(query, positions, rope)
    output = (
        torch.zeros(q.shape[0], padded, 512, dtype=torch.bfloat16)
        if padded
        else torch.empty(0, dtype=torch.bfloat16)
    )
    if padded:
        output[:, : q.shape[1]] = query.bfloat16()
    expected = storage.clone()
    rotated_kv = _rotate(kv, positions, rope).bfloat16()
    for token, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        page, offset = divmod(slot, block)
        groups = kv[token, :448].float().reshape(7, 64)
        exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
        encoded = (
            (groups * torch.exp2(-exponent)[:, None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .flatten()
        )
        expected[page, offset * 576 : offset * 576 + 448] = encoded
        expected[page, offset * 576 + 448 : (offset + 1) * 576] = rotated_kv[
            token, 448:
        ].view(torch.uint8)
        scale_start = block * 576 + offset * 8
        expected[page, scale_start : scale_start + 7] = (
            (exponent + 127).clamp(0, 255).byte()
        )
        expected[page, scale_start + 7] = 0
    return output, expected


@pytest.mark.gpu
@requires_cuda_fp8
@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("padded", [0, 3, 64])
def test_qkv_cache_matches_torch_bytes_and_query(norm, rotate, padded):
    from vllm_fl.ops.nvidia_dsv41_ops import register_nvidia_dsv41_ops

    assert register_nvidia_dsv41_ops()
    qnorm_rope_kv_rope_quant_insert = (
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    )

    block = 16
    q, kv, storage, slots, positions, rope = _inputs(block)
    original_q, original_kv = q.clone(), kv.clone()
    expected_q, expected_cache = _reference(
        q, kv, storage, slots, positions, rope, block, padded, norm, rotate
    )
    actual = qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        storage[:, : block * 584],
        slots,
        positions,
        rope,
        padded,
        1e-6,
        block,
        norm,
        False,
        rotate,
        False,
    )
    torch.testing.assert_close(actual.cpu(), expected_q, rtol=0.01, atol=0.008)
    torch.testing.assert_close(storage.cpu(), expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(q, original_q, rtol=0, atol=0)
    torch.testing.assert_close(kv, original_kv, rtol=0, atol=0)


@pytest.mark.gpu
@requires_cuda_fp8
def test_kv_only_matches_qkv_insert_and_preserves_page_padding():
    from vllm_fl.ops.nvidia_dsv41_cache import kv_rope_insert

    block = 64
    q, kv, storage, slots, positions, rope = _inputs(block)
    _, expected = _reference(
        q, kv, storage, slots, positions, rope, block, 0, False, True
    )
    kv_rope_insert(
        kv, storage[:, : block * 584].view(3, block, 584), slots, positions, rope, block
    )
    torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)


@pytest.mark.gpu
@requires_cuda_fp8
def test_qkv_cache_cuda_graph_updates_slots_and_positions():
    from vllm_fl.ops.nvidia_dsv41_ops import register_nvidia_dsv41_ops

    assert register_nvidia_dsv41_ops()
    qnorm_rope_kv_rope_quant_insert = (
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    )

    block = 16
    q, kv, storage, slots, positions, rope = _inputs(block)

    def run():
        return qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            storage[:, : block * 584],
            slots,
            positions,
            rope,
            64,
            1e-6,
            block,
            False,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    storage.fill_(165)
    slots.copy_(
        torch.tensor([2 * block, -1, block + 5, 1], dtype=torch.int64, device="cuda")
    )
    positions.copy_(torch.tensor([4, 3, 2, 1, 0], dtype=torch.int64, device="cuda"))
    expected_q, expected_cache = _reference(
        q, kv, storage, slots, positions, rope, block, 64, False, True
    )
    graph.replay()
    torch.testing.assert_close(actual.cpu(), expected_q, rtol=0.01, atol=0.008)
    torch.testing.assert_close(storage.cpu(), expected_cache, rtol=0, atol=0)


def test_qkv_cache_rejects_sm100_formats():
    from vllm_fl.ops.nvidia_dsv41_cache import qnorm_rope_kv_rope_quant_insert

    for kwargs in ({"kv_mxfp8": True}, {"is_q_interleaved": True}):
        with pytest.raises(NotImplementedError, match="584-byte"):
            qnorm_rope_kv_rope_quant_insert(
                None, None, None, None, None, None, 64, 1e-6, 16, **kwargs
            )

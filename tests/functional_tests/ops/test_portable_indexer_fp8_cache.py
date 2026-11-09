# SPDX-License-Identifier: Apache-2.0
"""Validate FP8 cache writes independently of portable TopK selection."""

import flaggems_vllm
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.gpu


def _reference_cache(k, slots, shape, marker):
    """CPU reference for the HY4 E4M3 cache layout and UE8M0 scales."""
    expected = torch.full(shape, marker, dtype=torch.uint8)
    block_size = shape[1]
    head_dim = k.shape[-1]
    flat = expected.view(shape[0], -1)
    values = flat[:, : block_size * head_dim]
    scales = flat[:, block_size * head_dim :].view(torch.float32)
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        x = k[row].cpu().float()
        scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax().clamp_min(1e-4) / 448)))
        quantized = (x / scale).to(torch.float8_e4m3fn).view(torch.uint8)
        block, offset = divmod(slot, block_size)
        values[block, offset * head_dim : (offset + 1) * head_dim] = quantized
        scales[block, offset] = scale
    return expected


@pytest.mark.parametrize("rows", [1, 7, 32, 33])
def test_fp8_cache_with_refreshed_graph_inputs(rows):
    if not current_platform.is_cuda() or not current_platform.has_device_capability(89):
        pytest.skip("FP8 E4M3 cache insertion requires NVIDIA SM89 or newer")

    torch.manual_seed(53)
    # Nonconsecutive slots span cache blocks; an extra token must not be stored.
    slots = torch.cat(
        [
            (torch.arange(rows, device="cuda") * 3 + 7) % 64,
            torch.tensor([-1], device="cuda"),
        ]
    )
    k = torch.randn(rows + 1, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.full((4, 16, 132), 0x5A, device="cuda", dtype=torch.uint8)

    def write_cache():
        flaggems_vllm.indexer_k_quant_and_cache(k, cache, slots, 128, "ue8m0")

    expected = _reference_cache(k, slots, cache.shape, 0x5A)
    before = k.clone()
    for _ in range(3):
        write_cache()
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(k, before, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write_cache()
    for shift in (11, 19):
        k.normal_()
        slots[:-1].copy_((slots[:-1] + shift) % 64)
        before = k.clone()
        cache.fill_(0xA5)
        expected = _reference_cache(k, slots, cache.shape, 0xA5)
        graph.replay()
        torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(k, before, rtol=0, atol=0)

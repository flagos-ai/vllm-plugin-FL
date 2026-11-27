# Copyright (c) 2026 BAAI. All rights reserved.
"""MetaX cache-writer regressions for the vLLM 0.32 hybrid cache layout."""

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


@pytest.mark.parametrize("head_size", [32, 256])
def test_cache_writer_preserves_gapped_pages_and_ignores_padding(head_size):
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils import (
        reshape_and_cache_flash,
    )

    # Qwen3.5's hybrid Mamba/attention allocator uses 528-token pages. K/V
    # alternate within each physical page, so unbinding them creates a gap
    # between blocks. Include a final-page tail and graph padding (-1 slot).
    block_size, num_heads = 528, 2
    slots = [0, 529, 1583, -1, 531]
    elements = len(slots) * num_heads * head_size
    key = (
        torch.arange(elements, dtype=torch.float32)
        .reshape(len(slots), num_heads, head_size)
        .to(torch.bfloat16)
    )
    value = (
        (-torch.arange(elements, dtype=torch.float32) - 3)
        .reshape(len(slots), num_heads, head_size)
        .to(torch.bfloat16)
    )
    shape = (3, 2, block_size, num_heads, head_size)
    expected = torch.full(shape, -7, dtype=torch.bfloat16)
    for token, slot in enumerate(slots):
        if slot >= 0:
            expected[slot // block_size, 0, slot % block_size] = key[token]
            expected[slot // block_size, 1, slot % block_size] = value[token]

    cache = torch.full(shape, -7, dtype=torch.bfloat16, device="cuda").contiguous()
    key_cache, value_cache = cache.unbind(1)
    assert not key_cache.is_contiguous()
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    reshape_and_cache_flash(
        key.cuda().contiguous(),
        value.cuda().contiguous(),
        key_cache,
        value_cache,
        torch.tensor(slots, dtype=torch.int64, device="cuda"),
        "auto",
        scale,
        scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)

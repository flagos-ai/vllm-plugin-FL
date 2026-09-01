# Copyright (c) 2026 BAAI. All rights reserved.

"""CUDA correctness tests for the DeepSeek-V4 INT8 indexer kernels."""

import pytest
import torch

from vllm_fl.ops.deepseek_v4_int8_indexer import (
    int8_mqa_logits,
    int8_paged_mqa_logits,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_int8_mqa_logits_matches_torch():
    torch.manual_seed(2)
    num_queries, num_keys, num_heads, head_dim = 1, 97, 64, 128
    q = torch.randint(
        -127,
        128,
        (num_queries, num_heads, head_dim),
        dtype=torch.int8,
    )
    k = torch.randint(-127, 128, (num_keys, head_dim), dtype=torch.int8)
    k_scale = torch.rand(num_keys, dtype=torch.float32) * 0.02
    weights = torch.randn(num_queries, num_heads, dtype=torch.float32)
    cu_ks = torch.tensor([0], dtype=torch.int32)
    cu_ke = torch.tensor([num_keys], dtype=torch.int32)

    actual = int8_mqa_logits(
        q.cuda(),
        k.cuda(),
        k_scale.cuda(),
        weights.cuda(),
        cu_ks.cuda(),
        cu_ke.cuda(),
    ).cpu()
    dots = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    expected = (dots * k_scale[None, None, :]).relu()
    expected = (expected * weights[:, :, None]).sum(dim=1)

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_int8_paged_mqa_logits_matches_torch():
    torch.manual_seed(3)
    batch, next_n, num_heads, head_dim = 1, 1, 64, 128
    block_size, num_blocks, context_len = 64, 2, 100
    q = torch.randint(
        -127,
        128,
        (batch, next_n, num_heads, head_dim),
        dtype=torch.int8,
    )
    k = torch.randint(
        -127,
        128,
        (num_blocks, block_size, head_dim),
        dtype=torch.int8,
    )
    k_scale = torch.rand(num_blocks, block_size, dtype=torch.float32) * 0.02
    weights = torch.randn(batch * next_n, num_heads, dtype=torch.float32)

    # The compressor stores one packed page as all INT8 K bytes followed by
    # all fp32 scales. The logical tensor shape only reserves 132 bytes/token;
    # its final dimension must not be interpreted as an interleaved layout.
    cache = torch.empty(
        num_blocks,
        block_size,
        head_dim + torch.tensor([], dtype=torch.float32).element_size(),
        dtype=torch.uint8,
    )
    flat_cache = cache.view(-1)
    for block in range(num_blocks):
        page_base = block * cache.stride(0)
        k_bytes = k[block].contiguous().view(torch.uint8).reshape(-1)
        scale_bytes = (
            k_scale[block].contiguous().view(torch.uint8).reshape(-1)
        )
        flat_cache[page_base : page_base + k_bytes.numel()].copy_(k_bytes)
        scale_start = page_base + k_bytes.numel()
        flat_cache[scale_start : scale_start + scale_bytes.numel()].copy_(
            scale_bytes
        )

    context_lens = torch.tensor([[context_len]], dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    actual = int8_paged_mqa_logits(
        q.cuda(),
        cache.cuda(),
        weights.cuda(),
        context_lens.cuda(),
        block_table.cuda(),
        num_blocks * block_size,
    ).cpu()

    flat_k = k.reshape(-1, head_dim)[:context_len]
    flat_scale = k_scale.reshape(-1)[:context_len]
    dots = torch.einsum("hd,nd->hn", q[0, 0].float(), flat_k.float())
    expected = (dots * flat_scale[None, :]).relu()
    expected = (expected * weights[0, :, None]).sum(dim=0)

    torch.testing.assert_close(
        actual[0, :context_len], expected, atol=2e-3, rtol=2e-3
    )
    assert torch.isfinite(actual[0, :context_len]).all()

# Copyright (c) 2026 BAAI. All rights reserved.
"""Real-device read-path regression; deliberately do not mock native FA."""

import pytest
import torch


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize("d", [64, 128])
def test_native_paged_attention_accepts_standardized_strides(n, d):
    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        pytest.skip("Requires Enflame native FlashAttention and a GCU")
    from flash_attn.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(42)
    cache = torch.randn(2, 2, 16, 2 * d, dtype=torch.bfloat16, device="gcu")
    k, v = cache.transpose(1, 2).split(d, dim=-1)
    assert not k.is_contiguous()
    q = torch.randn(n, 4, d, dtype=torch.bfloat16, device="gcu")
    kwargs = dict(
        q=q,
        max_seqlen_q=n,
        cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32, device="gcu"),
        max_seqlen_k=20,
        seqused_k=torch.tensor([20], dtype=torch.int32, device="gcu"),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32, device="gcu"),
        causal=True,
    )
    actual = flash_attn_varlen_func(k=k, v=v, **kwargs)
    expected = flash_attn_varlen_func(k=k.contiguous(), v=v.contiguous(), **kwargs)
    torch.gcu.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

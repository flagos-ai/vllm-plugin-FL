# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from vllm.config import VllmConfig, set_current_vllm_config

from vllm_fl.dispatch.backends.vendor.ascend.impl.mm_encoder_attention import (
    AscendMMEncoderAttention,
)


@pytest.mark.gpu
@pytest.mark.parametrize("head_size", [72, 128])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_ascend_vision_attention_matches_packed_sdpa(head_size, kv_heads):
    # Exercise the real 0.28 constructor, packed lengths and NPU kernel.
    with set_current_vllm_config(VllmConfig()):
        attention = AscendMMEncoderAttention(
            num_heads=2, head_size=head_size, num_kv_heads=kv_heads, scale=0.125
        )
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(1, 8, 2, head_size, generator=generator).to(torch.bfloat16)
    k = torch.randn(1, 8, kv_heads, head_size, generator=generator).to(torch.bfloat16)
    v = torch.randn(1, 8, kv_heads, head_size, generator=generator).to(torch.bfloat16)
    expected = []
    for start, end in [(0, 3), (3, 8)]:
        expected.append(
            F.scaled_dot_product_attention(
                q[:, start:end].transpose(1, 2).float(),
                k[:, start:end]
                .repeat_interleave(2 // kv_heads, dim=2)
                .transpose(1, 2)
                .float(),
                v[:, start:end]
                .repeat_interleave(2 // kv_heads, dim=2)
                .transpose(1, 2)
                .float(),
                scale=0.125,
            ).transpose(1, 2)
        )
    expected = torch.cat(expected, dim=1).to(torch.bfloat16)
    actual = attention.forward_oot(
        q.npu(),
        k.npu(),
        v.npu(),
        cu_seqlens=torch.tensor([0, 3, 8], dtype=torch.int32, device="npu"),
        max_seqlen=5,
        sequence_lengths=None,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.02)

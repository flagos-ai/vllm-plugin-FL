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


@pytest.mark.gpu
def test_ascend_vision_attention_matches_packed_sdpa_for_eight_images():
    # Match Qwen3.6 vision attention's head shape and cover the eight-image
    # packed batch used by the adaptation gate.
    num_heads = 16
    head_size = 72
    segment_lengths = [16, 24, 32, 40, 48, 56, 64, 72]
    cumulative_lengths = [0]
    for length in segment_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + length)

    with set_current_vllm_config(VllmConfig()):
        attention = AscendMMEncoderAttention(
            num_heads=num_heads,
            head_size=head_size,
            num_kv_heads=num_heads,
            scale=head_size**-0.5,
        )

    generator = torch.Generator().manual_seed(29)
    num_tokens = cumulative_lengths[-1]
    q = torch.randn(1, num_tokens, num_heads, head_size, generator=generator).to(
        torch.bfloat16
    )
    k = torch.randn(1, num_tokens, num_heads, head_size, generator=generator).to(
        torch.bfloat16
    )
    v = torch.randn(1, num_tokens, num_heads, head_size, generator=generator).to(
        torch.bfloat16
    )

    expected = []
    for start, end in zip(cumulative_lengths[:-1], cumulative_lengths[1:]):
        expected.append(
            F.scaled_dot_product_attention(
                q[:, start:end].transpose(1, 2).float(),
                k[:, start:end].transpose(1, 2).float(),
                v[:, start:end].transpose(1, 2).float(),
                scale=head_size**-0.5,
            ).transpose(1, 2)
        )
    expected = torch.cat(expected, dim=1).to(torch.bfloat16)

    actual = attention.forward_oot(
        q.npu(),
        k.npu(),
        v.npu(),
        cu_seqlens=torch.tensor(cumulative_lengths, dtype=torch.int32, device="npu"),
        max_seqlen=max(segment_lengths),
        sequence_lengths=None,
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.02)

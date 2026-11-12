# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.models.bge_m3 import attention as bge


def test_fp32_adapter_isolates_packed_documents(monkeypatch):
    calls = []

    def reference(
        q,
        k,
        v,
        *,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        softmax_scale,
    ):
        assert not causal
        assert cu_seqlens_q is cu_seqlens_k
        assert max_seqlen_q == max_seqlen_k == 5
        calls.append(q.shape[0])
        offsets = cu_seqlens_q.tolist()
        for start, end in zip(offsets, offsets[1:]):
            qi, ki, vi = [x[start:end].transpose(0, 1) for x in (q, k, v)]
            out[start:end] = (
                (qi @ ki.transpose(-1, -2) * softmax_scale).softmax(-1) @ vi
            ).transpose(0, 1)
        return out

    monkeypatch.setattr(bge, "flash_attn_varlen_func", reference)
    torch.manual_seed(123)
    q, k, v = torch.randn(8, 3, 2, 64).unbind(1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    mask = labels[:, None] != labels[None, :]
    scores = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2)) / 8
    expected = (
        scores.masked_fill(mask, -torch.inf).softmax(-1) @ v.transpose(0, 1)
    ).transpose(0, 1)
    impl = SimpleNamespace(
        sliding_window=(-1, -1), alibi_slopes=None, logits_soft_cap=0, scale=0.125
    )
    metadata = SimpleNamespace(
        causal=False,
        query_start_loc=torch.tensor([0, 3, 8], dtype=torch.int32),
        max_query_len=5,
    )
    output = torch.empty_like(q)

    actual = bge.BgeM3AttentionImpl._forward_encoder_attention(
        impl, q, k, v, output, metadata, None
    )

    assert actual is output
    assert calls == [8]
    torch.testing.assert_close(actual, expected)
    metadata.causal = True
    with pytest.raises(ValueError, match="bidirectional"):
        bge.BgeM3AttentionImpl._forward_encoder_attention(
            impl, q, k, v, output, metadata, None
        )

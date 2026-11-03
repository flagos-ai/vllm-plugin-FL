# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("flattened", [False, True])
def test_native_vision_attention_preserves_windows_and_scale(flattened, monkeypatch):
    pytest.importorskip("torch_npu")
    from vllm_fl.dispatch.backends.vendor.ascend.impl.mm_encoder_attention import (
        AscendMMEncoderAttention,
    )

    layer = SimpleNamespace(
        num_heads=2, num_kv_heads=1, num_queries_per_kv=2, head_size=4, scale=0.25
    )
    layer.reshape_qkv_to_3d = partial(AscendMMEncoderAttention.reshape_qkv_to_3d, layer)
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(1, 263, 2, 4, generator=generator)
    k = torch.randn(1, 263, 1, 4, generator=generator)
    v = torch.randn(1, 263, 1, 4, generator=generator)
    expected = []
    for start, end in [(0, 257), (257, 263)]:
        expected.append(
            F.scaled_dot_product_attention(
                q[:, start:end].transpose(1, 2),
                k[:, start:end].repeat_interleave(2, dim=2).transpose(1, 2),
                v[:, start:end].repeat_interleave(2, dim=2).transpose(1, 2),
                scale=layer.scale,
            ).transpose(1, 2)
        )
    expected = torch.cat(expected, dim=1)
    if flattened:
        q, k, v = [x.flatten(2) for x in (q, k, v)]
        expected = expected.flatten(2)

    def cpu_fia(**kwargs):
        assert kwargs["actual_seq_lengths"] == [257, 263]
        assert kwargs["actual_seq_lengths_kv"] == [257, 263]
        assert kwargs["num_key_value_heads"] == layer.num_heads
        outputs = []
        start = 0
        for end in kwargs["actual_seq_lengths"]:
            qx, kx, vx = [
                kwargs[name][start:end].transpose(0, 1).unsqueeze(0)
                for name in ("query", "key", "value")
            ]
            outputs.append(
                F.scaled_dot_product_attention(qx, kx, vx, scale=kwargs["scale"])
                .squeeze(0)
                .transpose(0, 1)
            )
            start = end
        return torch.cat(outputs), None

    monkeypatch.setattr("torch_npu.npu_fused_infer_attention_score", cpu_fia)
    result = AscendMMEncoderAttention.forward_oot(
        layer, q, k, v, cu_seqlens=torch.tensor([0, 257, 263])
    )
    torch.testing.assert_close(result, expected)

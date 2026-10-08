# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

import math
import sys
from types import SimpleNamespace

import pytest
import torch

import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.mm_encoder_attention as mm_impl


def test_large_mm_kernel_receives_contiguous_inputs_and_host_lengths(monkeypatch):
    monkeypatch.setattr(mm_impl, "_NATIVE_MM_MIN_SEQUENCE_LENGTH", 1)
    query = torch.arange(2 * 10 * 4 * 72, dtype=torch.float16).view(2, 10, 4, 72)
    query = query[:, ::2]
    key, value = query + 1, query + 2
    expected = query.contiguous()
    calls = []
    scale = 0.25

    def flash(query_flat, key_flat, value_flat, cu_q, cu_k, max_q, max_k, **kwargs):
        assert all(
            tensor.is_contiguous() for tensor in (query_flat, key_flat, value_flat)
        )
        torch.testing.assert_close(query_flat, expected.view(10, 4, 72))
        assert kwargs["cu_seqlens_qo_cpu"].device.type == "cpu"
        assert kwargs["cu_seqlens_qo_cpu"].dtype == torch.int32
        torch.testing.assert_close(cu_q, torch.tensor([0, 5, 10], dtype=torch.int32))
        torch.testing.assert_close(cu_k, cu_q)
        assert max_q == max_k == 5
        assert kwargs["softmax_scale"] == pytest.approx(scale * math.sqrt(72))
        assert kwargs["dropout_p"] == 0.0
        assert not kwargs["causal"]
        assert kwargs["is_varlen"] and kwargs["is_prefill"]
        calls.append(True)
        return query_flat.clone(), None

    monkeypatch.setitem(
        sys.modules, "xtorch_ops", SimpleNamespace(flash_attn_varlen_func=flash)
    )
    actual = mm_impl.try_native_large_mm_attention(query, key, value, scale)

    assert calls == [True]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 128, 4, 72), torch.bfloat16),
        ((1, 8192, 4, 72), torch.float32),
        ((1, 8192, 4, 80), torch.bfloat16),
    ],
)
def test_small_or_unsupported_mm_inputs_keep_sdpa(shape, dtype):
    query = torch.empty(shape, dtype=dtype)

    assert mm_impl.try_native_large_mm_attention(query, query, query, 0.25) is None


def test_gqa_shapes_keep_sdpa():
    query = torch.empty((1, 8192, 4, 72), dtype=torch.bfloat16)
    key = torch.empty((1, 8192, 2, 72), dtype=torch.bfloat16)

    assert mm_impl.try_native_large_mm_attention(query, key, key, 0.25) is None


@pytest.mark.parametrize("head_dim", [72, 96])
@pytest.mark.parametrize("scale", [None, 0.25])
def test_native_mm_scaling_matches_cpu_reference(device, monkeypatch, head_dim, scale):
    from vllm.platforms import current_platform

    if (
        device.type != "cuda"
        or getattr(current_platform, "vendor_name", None) != "kunlunxin"
    ):
        pytest.skip("native Kunlunxin MM attention requires Kunlunxin hardware")

    monkeypatch.setattr(mm_impl, "_NATIVE_MM_MIN_SEQUENCE_LENGTH", 1)
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    generator = torch.Generator().manual_seed(23)
    cpu_inputs = [
        torch.randn((2, 37, 4, head_dim), generator=generator).to(torch.bfloat16)
        for _ in range(3)
    ]
    query, key, value = [tensor.to(device) for tensor in cpu_inputs]
    output = mm_impl.try_native_large_mm_attention(query, key, value, scale)
    q_cpu, k_cpu, v_cpu = [tensor.float().permute(0, 2, 1, 3) for tensor in cpu_inputs]
    expected = (q_cpu @ k_cpu.transpose(-1, -2) * scale).softmax(-1) @ v_cpu
    expected = expected.permute(0, 2, 1, 3)

    torch.testing.assert_close(output.cpu().float(), expected, rtol=0.025, atol=0.003)

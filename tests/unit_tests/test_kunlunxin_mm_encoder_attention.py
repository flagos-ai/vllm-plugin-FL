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


@pytest.mark.parametrize("length", [8192, 16384])
@pytest.mark.parametrize("head_dim", [64, 72, 96, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_long_mm_attention_matches_cpu_reference(device, length, head_dim, dtype):
    from vllm.platforms import current_platform

    if (
        device.type != "cuda"
        or getattr(current_platform, "vendor_name", None) != "kunlunxin"
    ):
        pytest.skip("long MM attention requires Kunlunxin hardware")
    generator = torch.Generator().manual_seed(718)
    cpu_inputs = [
        torch.randn((1, length, 4, head_dim), generator=generator).to(dtype)
        for _ in range(3)
    ]
    scale = head_dim**-0.5
    output = mm_impl.try_native_large_mm_attention(
        *(tensor.to(device) for tensor in cpu_inputs), scale
    )
    torch.cuda.synchronize()
    actual = output.cpu().float()
    q, k, v = [tensor.float().permute(0, 2, 1, 3) for tensor in cpu_inputs]
    # Compare every output element with a bounded independent CPU score matrix.
    for start in range(0, length, 256):
        scores = q[:, :, start : start + 256] @ k.transpose(-1, -2) * scale
        expected = (scores.softmax(-1) @ v).permute(0, 2, 1, 3)
        torch.testing.assert_close(
            actual[:, start : start + 256], expected, rtol=0.025, atol=0.003
        )


@pytest.mark.parametrize("entry", ["forward", "forward_cuda", "forward_native"])
def test_public_explicit_sdpa_never_calls_vendor(monkeypatch, entry):
    import torch.nn.functional as functional

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.attention.utils import patch_mm_encoder_attention

    if getattr(current_platform, "vendor_name", None) != "kunlunxin":
        pytest.skip("Kunlunxin provider routing")
    import vllm.model_executor.layers.attention.mm_encoder_attention as mm_module

    patch_mm_encoder_attention()
    monkeypatch.setattr(
        mm_module,
        "get_vit_attn_backend",
        lambda **kwargs: AttentionBackendEnum.TORCH_SDPA,
    )
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("explicit SDPA/reference entered vendor attention")

    def sdpa(q, k, v, **kwargs):
        calls.append(q.shape)
        return q.clone()

    monkeypatch.setattr(mm_impl, "try_native_large_mm_attention", forbidden)
    monkeypatch.setattr(functional, "scaled_dot_product_attention", sdpa)
    with set_current_vllm_config(VllmConfig()):
        attention = mm_module.MMEncoderAttention(4, 72)
        query = torch.zeros((1, 8192, 4, 72), dtype=torch.bfloat16)
        output = getattr(attention, entry)(query, query, query)
    assert calls and sum(shape[2] for shape in calls) == 8192
    torch.testing.assert_close(output, query)


@pytest.mark.parametrize("flattened", [False, True])
def test_public_cross_attention_preserves_sdpa(device, monkeypatch, flattened):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.attention.utils import patch_mm_encoder_attention

    if (
        device.type != "cuda"
        or getattr(current_platform, "vendor_name", None) != "kunlunxin"
    ):
        pytest.skip("Kunlunxin hardware provider routing")
    import vllm.model_executor.layers.attention.mm_encoder_attention as mm_module

    patch_mm_encoder_attention()
    monkeypatch.setattr(
        mm_module,
        "get_vit_attn_backend",
        lambda **kwargs: AttentionBackendEnum.FLASH_ATTN,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("cross-attention entered self-attention-only vendor provider")

    monkeypatch.setattr(mm_impl, "try_native_large_mm_attention", forbidden)
    generator = torch.Generator().manual_seed(317)
    cpu_inputs = [
        torch.randn((1, length, 4, 72), generator=generator).to(torch.bfloat16)
        for length in (37, 53, 53)
    ]
    q, k, v = [tensor.float().permute(0, 2, 1, 3) for tensor in cpu_inputs]
    expected = ((q @ k.transpose(-1, -2) / math.sqrt(72)).softmax(-1) @ v).permute(
        0, 2, 1, 3
    )
    inputs = [tensor.to(device) for tensor in cpu_inputs]
    if flattened:
        inputs = [tensor.flatten(2) for tensor in inputs]
        expected = expected.flatten(2)
    with set_current_vllm_config(VllmConfig()):
        attention = mm_module.MMEncoderAttention(4, 72)
        output = attention(*inputs)
    torch.testing.assert_close(output.cpu().float(), expected, rtol=0.025, atol=0.003)


@pytest.mark.parametrize("reference", [False, True])
def test_public_long_sdpa_numerics_without_vendor(device, monkeypatch, reference):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.attention.utils import patch_mm_encoder_attention

    if (
        device.type != "cuda"
        or getattr(current_platform, "vendor_name", None) != "kunlunxin"
    ):
        pytest.skip("Kunlunxin hardware SDPA routing")
    import xtorch_ops

    import vllm.model_executor.layers.attention.mm_encoder_attention as mm_module

    patch_mm_encoder_attention()
    monkeypatch.setattr(
        mm_module,
        "get_vit_attn_backend",
        lambda **kwargs: AttentionBackendEnum.TORCH_SDPA,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("explicit SDPA/reference invoked xtorch attention")

    monkeypatch.setattr(xtorch_ops, "flash_attn_varlen_func", forbidden)
    generator = torch.Generator().manual_seed(941)
    cpu = [
        torch.randn((1, 8192, 4, 72), generator=generator).to(torch.bfloat16)
        for _ in range(3)
    ]
    with set_current_vllm_config(VllmConfig()):
        attention = mm_module.MMEncoderAttention(4, 72)
        inputs = [tensor.to(device) for tensor in cpu]
        output = (
            (attention.forward_native(*inputs) if reference else attention(*inputs))
            .cpu()
            .float()
        )
    q, k, v = [tensor.float().permute(0, 2, 1, 3) for tensor in cpu]
    for start in range(0, 8192, 256):
        scores = q[:, :, start : start + 256] @ k.transpose(-1, -2) / math.sqrt(72)
        expected = (scores.softmax(-1) @ v).permute(0, 2, 1, 3)
        torch.testing.assert_close(
            output[:, start : start + 256], expected, rtol=0.025, atol=0.003
        )

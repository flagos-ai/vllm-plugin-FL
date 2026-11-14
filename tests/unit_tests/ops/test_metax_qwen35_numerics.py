# Copyright (c) 2026 BAAI. All rights reserved.
"""Qwen3.5-4B's actual Gemma norm and MRoPE paths on a MetaX GPU."""

from contextlib import contextmanager

import pytest
import torch


def _require_metax():
    if not torch.cuda.is_available():
        pytest.skip("Requires a MetaX GPU")
    from tests.utils.device_utils import get_backend

    if get_backend() != "metax":
        pytest.skip("MetaX-only Qwen3.5 operator numerics")


@contextmanager
def _eager_config():
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationMode

    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with (
        set_current_vllm_config(config),
        config.kernel_config.ir_op_priority.set_priority(),
    ):
        yield config


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(7, 2560), (7, 16, 256), (7, 32, 128)])
@pytest.mark.parametrize("has_residual", [False, True])
@torch.inference_mode()
def test_qwen35_gemma_norm_actual_ir_dispatch(shape, has_residual):
    _require_metax()
    from vllm import ir
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    device, dtype = torch.device("cuda"), torch.bfloat16
    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn_like(x) if has_residual else None
    original_x = x.clone()
    original_residual = residual.clone() if has_residual else None
    with _eager_config() as config:
        norm = GemmaRMSNorm(shape[-1], eps=1e-6).to(device=device, dtype=dtype)
        norm.weight.copy_(torch.randn_like(norm.weight) * 0.2)
        # Qwen3.5 uses GemmaRMSNorm: multiplication is FP32 with weight + 1,
        # before rounding to BF16. A same-dtype RMS test cannot check this.
        weight = norm.weight.float() + 1.0
        work = original_x.float()
        if has_residual:
            work = work + original_residual.float()
        expected_residual = work.to(dtype)
        expected = (
            work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-6) * weight
        ).to(dtype)
        op = ir.ops.fused_add_rms_norm if has_residual else ir.ops.rms_norm
        args = (x, residual, weight, 1e-6) if has_residual else (x, weight, 1e-6)
        selected = op.dispatch(*args).provider
        print(
            f"QWEN35-GEMMA {shape=} {has_residual=} selected={selected} "
            f"priority={getattr(config.kernel_config.ir_op_priority, op.name)} "
            f"forward={norm._forward_method.__qualname__} weight_dtype={weight.dtype}"
        )
        # Default MetaX priority may include vllm_c, whose argument predicate
        # must reject this FP32-weight/BF16-input combination.
        assert selected == "native"
        actual = norm(x, residual)
    output = actual[0] if has_residual else actual
    torch.cuda.synchronize()
    error = (output.float() - expected.float()).abs().max().item()
    print(f"QWEN35-GEMMA {shape=} {has_residual=} max_abs_error={error:.6g}")
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.02)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    if has_residual:
        torch.testing.assert_close(actual[1], expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)


def _mrope_reference(value, positions):
    """Independent [11, 11, 10] THW frequency selection and NeoX rotation."""
    rotary_dim, base = 64, 1e7
    inv_freq = base ** (-torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    # The native constructor computes a CPU FP32 cache, then rounds to BF16.
    angles = positions.cpu().float()[..., None] * inv_freq
    cos, sin = angles.cos().to(value.dtype), angles.sin().to(value.dtype)
    channels = torch.arange(rotary_dim // 2)
    axes = torch.where((channels % 3 == 1) & (channels < 11 * 3), 1, 0)
    axes = torch.where((channels % 3 == 2) & (channels < 10 * 3), 2, axes)
    token = torch.arange(positions.shape[-1])[:, None]
    cos = cos[axes[None], token, channels[None]].to(value.device).float()
    sin = sin[axes[None], token, channels[None]].to(value.device).float()
    first, second = value[..., :32].float(), value[..., 32:64].float()
    result = value.clone()
    result[..., :32] = first * cos[:, None] - second * sin[:, None]
    result[..., 32:64] = second * cos[:, None] + first * sin[:, None]
    return result


@pytest.mark.gpu
@pytest.mark.parametrize("different_axes", [False, True])
@torch.inference_mode()
def test_qwen35_native_mrope_partial_256_matches_oracle(different_axes):
    _require_metax()
    from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

    device, dtype = torch.device("cuda"), torch.bfloat16
    text_positions = torch.tensor([0, 1, 3, 7, 16, 21, 30], dtype=torch.long)
    positions = text_positions.repeat(3, 1)
    if different_axes:
        positions[1] = torch.tensor([0, 4, 2, 12, 11, 28, 37])
        positions[2] = torch.tensor([1, 5, 8, 13, 17, 31, 41])
    owner = torch.randn(7, 20, 256, device=device, dtype=dtype)
    original = owner.clone()
    query, key = owner[:, :16], owner[:, 16:]
    expected_query = _mrope_reference(query, positions)
    expected_key = _mrope_reference(key, positions)
    with _eager_config():
        rope = MRotaryEmbedding(
            head_size=256,
            rotary_dim=64,
            max_position_embeddings=128,
            base=1e7,
            is_neox_style=True,
            dtype=dtype,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
            positions_factor=1,
        ).to(device)
        print(
            f"QWEN35-MROPE {different_axes=} "
            f"forward={rope._forward_method.__qualname__} "
            f"query_stride={query.stride()} key_stride={key.stride()}"
        )
        output_query, output_key = rope(
            positions.to(device), query.flatten(1), key.flatten(1)
        )
    torch.cuda.synchronize()
    for output, expected in (
        (output_query.view_as(query), expected_query),
        (output_key.view_as(key), expected_key),
    ):
        print(
            f"QWEN35-MROPE {different_axes=} "
            f"max_abs_error={(output.float() - expected.float()).abs().max().item():.6g}"
        )
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.02)
        torch.testing.assert_close(output[..., 64:], expected[..., 64:], rtol=0, atol=0)
    torch.testing.assert_close(owner, original, rtol=0, atol=0)

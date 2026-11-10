# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon GPU correctness checks for the DeepSeek-V4 main sparse attention.

Run with::

    pytest -q -s \
      tests/unit_tests/dispatch/test_hygon_main_sparse_attention_correctness.py

The checks cover the LightOp Q/KV cache writer used by FL and the upstream
vLLM 0.24 ROCm sparse prefill/decode kernels selected by the Hygon backend.
References are expressed in PyTorch and do not import vllm_hcu.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def gpu_modules():
    if not torch.cuda.is_available():
        pytest.skip("Hygon/HIP GPU is required")

    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        dequantize_and_gather_k_cache_triton,
    )
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse

    print(
        "Sparse attention runtime: "
        f"device={torch.cuda.get_device_name(0)} "
        f"platform={type(current_platform).__name__} "
        f"is_fp8_fnuz={current_platform.is_fp8_fnuz()} "
        f"fp8_dtype={current_platform.fp8_dtype()}"
    )
    return rocm_aiter_mla_sparse, dequantize_and_gather_k_cache_triton


def _randn(*shape: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _error_stats(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(
        f"{name}: shape={tuple(actual.shape)} dtype={actual.dtype} "
        f"nan={bool(torch.isnan(actual_f).any().item())} "
        f"inf={bool(torch.isinf(actual_f).any().item())} "
        f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g}"
    )
    return max_abs, mean_abs


def _report_and_assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    _error_stats(name, actual, expected)
    assert torch.isfinite(actual.float()).all(), f"{name} produced NaN/Inf"
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _cos_sin_cache(num_positions: int, rope_dim: int = 64) -> torch.Tensor:
    angles = torch.randn(
        num_positions,
        rope_dim // 2,
        device="cuda",
        dtype=torch.float32,
    )
    # This matches RotaryEmbedding.cos_sin_cache in the real model.  LightOp's
    # fused Q/KV writer requires this cache to remain FP32.
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def _rope_reference(
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int = 64,
) -> torch.Tensor:
    result = value.float().clone()
    nope_dim = result.shape[-1] - rope_dim
    half = rope_dim // 2
    cos_sin = cos_sin_cache[positions.long()].float()
    cos, sin = cos_sin[..., :half], cos_sin[..., half:]
    while cos.ndim < result.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotary = result[..., nope_dim:]
    even = rotary[..., 0::2].clone()
    odd = rotary[..., 1::2].clone()
    result[..., nope_dim::2] = even * cos - odd * sin
    result[..., nope_dim + 1 :: 2] = odd * cos + even * sin
    return result.to(torch.bfloat16)


def _sparse_attention_reference(
    q: torch.Tensor,
    keys_per_query: list[torch.Tensor],
    scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.empty_like(q, dtype=torch.bfloat16)
    for row, keys in enumerate(keys_per_query):
        q_row = q[row].float()
        keys_f = keys.float()
        scores = (q_row @ keys_f.transpose(0, 1)) * scale
        if attn_sink is None:
            probs = torch.softmax(scores, dim=-1)
        else:
            sink = attn_sink[: q.shape[1]].float().unsqueeze(-1)
            probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[
                :, : keys.shape[0]
            ]
        output[row] = (probs @ keys_f).to(torch.bfloat16)
    return output


def _make_ocp_fp8_ds_mla_cache(
    rows: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode rows in the physical 576B data + 8B scale page layout.

    Scales are fixed at one (UE8M0 byte 127), so the PyTorch reference is
    simply the OCP-FP8-rounded NoPE part concatenated with the BF16 RoPE part.
    """
    assert rows.shape[-1] == 512 and rows.dtype == torch.bfloat16
    num_rows = rows.shape[0]
    num_blocks = (num_rows + block_size - 1) // block_size
    cache = torch.zeros(
        num_blocks,
        block_size,
        584,
        device=rows.device,
        dtype=torch.uint8,
    )
    nope_fp8 = rows[:, :448].to(torch.float8_e4m3fn)
    dequantized = torch.cat(
        (nope_fp8.to(torch.bfloat16), rows[:, 448:]),
        dim=-1,
    )

    for row in range(num_rows):
        block_idx, position = divmod(row, block_size)
        block = cache[block_idx].view(-1)
        data_offset = position * 576
        block[data_offset : data_offset + 448].copy_(nope_fp8[row].view(torch.uint8))
        block[data_offset + 448 : data_offset + 576].view(torch.bfloat16).copy_(
            rows[row, 448:]
        )
        scale_offset = block_size * 576 + position * 8
        block[scale_offset : scale_offset + 7] = 127
    return cache, dequantized


def test_lightop_qnorm_rope_cache_insert_matches_024_layout(gpu_modules):
    """Validate the LightOp 8-argument ABI against vLLM 0.24 cache readers."""
    _, dequantize_cache = gpu_modules
    try:
        from lightop import op as lightop_op
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"LightOp is required by the active Hygon patch: {exc}")

    kernel = getattr(
        lightop_op,
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        None,
    )
    if kernel is None:
        pytest.fail("Installed LightOp lacks the DeepSeek-V4 Q/KV insert kernel")

    torch.manual_seed(201)
    num_tokens, num_heads, head_dim = 9, 16, 512
    # DeepSeek-V4 SWA pages use the CLI block size (256 in the deployment).
    block_size = 256
    eps = 1e-6
    q_input = _randn(num_tokens, num_heads, head_dim) * 0.25
    kv_input = _randn(num_tokens, head_dim) * 0.25
    q_actual = q_input.clone()
    cache = torch.zeros(
        1,
        block_size,
        584,
        device="cuda",
        dtype=torch.uint8,
    )
    slots = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    cos_sin = _cos_sin_cache(32)

    kernel(
        q_actual,
        kv_input,
        cache.view(cache.shape[0], -1),
        slots,
        positions,
        cos_sin,
        eps,
        block_size,
    )

    q_norm = q_input.float() * torch.rsqrt(
        q_input.float().square().mean(dim=-1, keepdim=True) + eps
    )
    q_expected = _rope_reference(q_norm, positions, cos_sin)
    kv_expected = _rope_reference(kv_input, positions, cos_sin)
    _report_and_assert_close(
        "lightop_qnorm_rope/q",
        q_actual,
        q_expected,
        atol=0.025,
        rtol=0.02,
    )

    seq_lens = torch.tensor([num_tokens], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    gathered_ocp = torch.empty(
        1,
        num_tokens,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dequantize_cache(
        gathered_ocp,
        cache,
        seq_lens,
        None,
        block_table,
        block_size,
        0,
        use_fnuz=False,
    )
    _report_and_assert_close(
        "lightop_cache_insert/read_as_ocp",
        gathered_ocp[0],
        kv_expected,
        atol=0.08,
        rtol=0.16,
    )

    # Diagnostic only: if this error is much smaller than OCP, LightOp wrote
    # FNUZ bytes while PlatformFL/vLLM 0.24 decodes them as OCP.
    try:
        gathered_fnuz = torch.empty_like(gathered_ocp)
        dequantize_cache(
            gathered_fnuz,
            cache,
            seq_lens,
            None,
            block_table,
            block_size,
            0,
            use_fnuz=True,
        )
        _error_stats(
            "lightop_cache_insert/read_as_fnuz_diagnostic",
            gathered_fnuz[0],
            kv_expected,
        )
    except Exception as exc:
        print(f"FNUZ diagnostic unavailable: {type(exc).__name__}: {exc}")


def test_main_sparse_prefill_matches_torch(gpu_modules):
    sparse_ops, _ = gpu_modules
    torch.manual_seed(202)
    num_queries, num_keys, num_heads, head_dim = 6, 160, 16, 512
    q = _randn(num_queries, num_heads, head_dim) * 0.125
    kv = _randn(num_keys, 1, head_dim) * 0.125
    width = 96
    indices = torch.empty(num_queries, width, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([96, 81, 65, 49, 33, 17], device="cuda", dtype=torch.int32)
    for row in range(num_queries):
        permutation = torch.randperm(num_keys, device="cuda")[:width]
        indices[row] = permutation.to(torch.int32)
        indices[row, int(lengths[row].item()) :] = -1
    scale = head_dim**-0.5
    attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.1
    actual = torch.empty_like(q)

    sparse_ops.rocm_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=lengths,
        scale=scale,
        head_dim=head_dim,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=attn_sink,
        output=actual,
    )
    keys_per_query = [
        kv[
            indices[row, : int(lengths[row].item())].long(),
            0,
        ]
        for row in range(num_queries)
    ]
    expected = _sparse_attention_reference(q, keys_per_query, scale, attn_sink)
    _report_and_assert_close(
        "main_sparse_attention/prefill",
        actual,
        expected,
        atol=0.035,
        rtol=0.04,
    )


def test_main_sparse_decode_matches_torch(gpu_modules):
    sparse_ops, _ = gpu_modules
    torch.manual_seed(203)
    batch_size, num_heads, head_dim = 3, 16, 512
    main_rows = _randn(192, head_dim) * 0.125
    extra_rows = _randn(96, head_dim) * 0.125
    main_cache, main_dequant = _make_ocp_fp8_ds_mla_cache(main_rows, 256)
    extra_cache, extra_dequant = _make_ocp_fp8_ds_mla_cache(extra_rows, 64)

    q = _randn(batch_size, num_heads, head_dim) * 0.125
    main_width, extra_width = 80, 48
    main_lens = torch.tensor([80, 61, 43], device="cuda", dtype=torch.int32)
    extra_lens = torch.tensor([48, 35, 21], device="cuda", dtype=torch.int32)
    main_indices = torch.full(
        (batch_size, 1, main_width),
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    extra_indices = torch.full(
        (batch_size, 1, extra_width),
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    for row in range(batch_size):
        main_len = int(main_lens[row].item())
        extra_len = int(extra_lens[row].item())
        main_indices[row, 0, :main_len] = torch.randperm(
            main_rows.shape[0], device="cuda"
        )[:main_len].to(torch.int32)
        extra_indices[row, 0, :extra_len] = torch.randperm(
            extra_rows.shape[0], device="cuda"
        )[:extra_len].to(torch.int32)

    scale = head_dim**-0.5
    attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.1
    actual = torch.empty_like(q)
    sparse_ops.rocm_sparse_attn_decode(
        q=q,
        kv_cache=extra_cache,
        swa_k_cache=main_cache,
        swa_only=False,
        topk_indices=extra_indices,
        topk_lens=extra_lens,
        swa_indices=main_indices,
        swa_lens=main_lens,
        swa_ragged_indices=None,
        swa_ragged_indptr=None,
        topk_ragged_indices=None,
        topk_ragged_indptr=None,
        attn_sink=attn_sink,
        scale=scale,
        head_dim=head_dim,
        nope_head_dim=448,
        rope_head_dim=64,
        output=actual,
    )

    keys_per_query = []
    for row in range(batch_size):
        main_selected = main_dequant[
            main_indices[row, 0, : int(main_lens[row].item())].long()
        ]
        extra_selected = extra_dequant[
            extra_indices[row, 0, : int(extra_lens[row].item())].long()
        ]
        keys_per_query.append(torch.cat((main_selected, extra_selected), dim=0))
    expected = _sparse_attention_reference(q, keys_per_query, scale, attn_sink)
    _report_and_assert_close(
        "main_sparse_attention/decode",
        actual,
        expected,
        atol=0.04,
        rtol=0.05,
    )

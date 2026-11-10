# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon GPU correctness checks for the FL DeepSeek-V4 MHC path.

Run with::

    pytest -q -s tests/unit_tests/dispatch/test_hygon_mhc_correctness.py

The patched HIP entry points are compared with vLLM's PyTorch definitions for
MHC pre, post, fused post+pre, and the validated TileLang HC head reduction.
"""

from __future__ import annotations

import math

import pytest
import torch


HC_MULT = 4
HIDDEN_SIZE = 7168
HC_DIM = HC_MULT * HIDDEN_SIZE
MIX_HC = (2 + HC_MULT) * HC_MULT
RMS_EPS = 1e-6
HC_EPS = 1e-6
HC_POST_MULT = 2.0
SINKHORN_REPEAT = 3


@pytest.fixture(scope="module")
def mhc_modules():
    if not torch.cuda.is_available():
        pytest.skip("Hygon/HIP GPU is required")

    import vllm.model_executor.kernels.mhc as mhc_kernels
    import vllm.model_executor.layers.mhc as mhc_layers
    from vllm_fl.dispatch.backends.vendor.hygon.impl import mhc as mhc_impl

    print(
        "MHC runtime: "
        f"HAS_TILELANG_MHC={mhc_layers.HAS_TILELANG_MHC} "
        f"FL_backend={mhc_impl.get_mhc_backend_name()} "
        f"device={torch.cuda.get_device_name(0)}"
    )
    if not mhc_layers.HAS_TILELANG_MHC:
        pytest.fail(
            "The active Hygon service profile used TileLang MHC, but this test "
            "process selected the native path"
        )
    for op_cls in (
        mhc_layers.MHCPreOp,
        mhc_layers.MHCPostOp,
        mhc_layers.MHCFusedPostPreOp,
    ):
        if not getattr(op_cls.forward_hip, "_vllm_fl_hygon", False):
            pytest.fail(f"FL did not patch {op_cls.__name__}.forward_hip")
    return mhc_layers, mhc_kernels


def _randn(*shape: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _report_and_assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    print(
        f"{name}: shape={tuple(actual.shape)} dtype={actual.dtype} "
        f"nan={bool(torch.isnan(actual_f).any().item())} "
        f"inf={bool(torch.isinf(actual_f).any().item())} "
        f"max_abs={diff.max().item():.8g} "
        f"mean_abs={diff.mean().item():.8g}"
    )
    assert torch.isfinite(actual_f).all(), f"{name} produced NaN/Inf"
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _mhc_parameters(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    fn = _randn(MIX_HC, HC_DIM, dtype=torch.float32) / math.sqrt(HC_DIM)
    hc_scale = torch.tensor([0.5, 0.5, 0.5], device="cuda", dtype=torch.float32)
    hc_base = _randn(MIX_HC, dtype=torch.float32) * 0.05
    return fn, hc_scale, hc_base


def _compare_pre_outputs(
    name: str,
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    _report_and_assert_close(
        f"{name}/post_mix",
        actual[0],
        expected[0],
        atol=0.004,
        rtol=0.01,
    )
    _report_and_assert_close(
        f"{name}/comb_mix",
        actual[1],
        expected[1],
        atol=0.004,
        rtol=0.01,
    )
    _report_and_assert_close(
        f"{name}/layer_input",
        actual[2],
        expected[2],
        atol=0.035,
        rtol=0.035,
    )


@pytest.mark.parametrize("num_tokens", [1, 32, 128])
def test_mhc_pre_hygon_matches_torch(mhc_modules, num_tokens):
    mhc_layers, mhc_kernels = mhc_modules
    fn, hc_scale, hc_base = _mhc_parameters(300 + num_tokens)
    residual = _randn(num_tokens, HC_MULT, HIDDEN_SIZE) * 0.25

    # Call the patched production method without constructing CustomOp, which
    # would require a global VllmConfig in this standalone operator test.
    actual = mhc_layers.MHCPreOp.forward_hip(
        None,
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
        1,
        None,
        0.0,
    )
    expected = mhc_kernels.mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
    )
    _compare_pre_outputs(f"mhc_pre/tokens={num_tokens}", actual, expected)


def test_mhc_pre_hygon_optional_norm_matches_torch(mhc_modules):
    mhc_layers, mhc_kernels = mhc_modules
    fn, hc_scale, hc_base = _mhc_parameters(350)
    residual = _randn(3, HC_MULT, HIDDEN_SIZE) * 0.25
    norm_weight = _randn(HIDDEN_SIZE) * 0.1 + 1.0

    actual = mhc_layers.MHCPreOp.forward_hip(
        None,
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
        1,
        norm_weight,
        RMS_EPS,
    )
    expected = list(
        mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            RMS_EPS,
            HC_EPS,
            HC_EPS,
            HC_POST_MULT,
            SINKHORN_REPEAT,
        )
    )
    layer_input_fp32 = expected[2].float()
    expected[2] = (
        layer_input_fp32
        * torch.rsqrt(layer_input_fp32.square().mean(dim=-1, keepdim=True) + RMS_EPS)
        * norm_weight.float()
    ).to(torch.bfloat16)
    _compare_pre_outputs("mhc_pre/optional_norm", actual, tuple(expected))


@pytest.mark.parametrize("num_tokens", [1, 64])
def test_mhc_post_hygon_matches_torch(mhc_modules, num_tokens):
    mhc_layers, mhc_kernels = mhc_modules
    fn, hc_scale, hc_base = _mhc_parameters(400 + num_tokens)
    residual = _randn(num_tokens, HC_MULT, HIDDEN_SIZE) * 0.25
    post_mix, comb_mix, _ = mhc_kernels.mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
    )
    x = _randn(num_tokens, HIDDEN_SIZE) * 0.25

    actual = mhc_layers.MHCPostOp.forward_hip(
        None,
        x,
        residual,
        post_mix,
        comb_mix,
    )
    expected = mhc_kernels.mhc_post_torch(
        x,
        residual,
        post_mix,
        comb_mix,
    )
    _report_and_assert_close(
        f"mhc_post/tokens={num_tokens}",
        actual,
        expected,
        atol=0.025,
        rtol=0.025,
    )


@pytest.mark.parametrize("num_tokens", [1, 32])
def test_mhc_fused_post_pre_hygon_matches_torch(mhc_modules, num_tokens):
    mhc_layers, mhc_kernels = mhc_modules
    old_fn, old_scale, old_base = _mhc_parameters(500 + num_tokens)
    new_fn, new_scale, new_base = _mhc_parameters(600 + num_tokens)
    residual = _randn(num_tokens, HC_MULT, HIDDEN_SIZE) * 0.25
    post_mix, comb_mix, _ = mhc_kernels.mhc_pre_torch(
        residual,
        old_fn,
        old_scale,
        old_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
    )
    x = _randn(num_tokens, HIDDEN_SIZE) * 0.25

    actual = mhc_layers.MHCFusedPostPreOp.forward_hip(
        None,
        x,
        residual,
        post_mix,
        comb_mix,
        new_fn,
        new_scale,
        new_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
        1,
        1,
        None,
        0.0,
    )
    residual_expected = mhc_kernels.mhc_post_torch(
        x,
        residual,
        post_mix,
        comb_mix,
    )
    pre_expected = mhc_kernels.mhc_pre_torch(
        residual_expected,
        new_fn,
        new_scale,
        new_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
    )

    _report_and_assert_close(
        f"mhc_fused_post_pre/tokens={num_tokens}/residual",
        actual[0],
        residual_expected,
        atol=0.025,
        rtol=0.025,
    )
    _compare_pre_outputs(
        f"mhc_fused_post_pre/tokens={num_tokens}",
        actual[1:],
        pre_expected,
    )


@pytest.mark.parametrize("num_tokens", [1, 32])
def test_hc_head_tilelang_matches_torch(mhc_modules, num_tokens):
    mhc_layers, _ = mhc_modules
    torch.manual_seed(700 + num_tokens)
    hidden_states = _randn(num_tokens, HC_MULT, HIDDEN_SIZE) * 0.25
    hc_fn = _randn(HC_MULT, HC_DIM, dtype=torch.float32) / math.sqrt(HC_DIM)
    hc_scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    hc_base = _randn(HC_MULT, dtype=torch.float32) * 0.05

    actual = torch.ops.vllm.hc_head_fused_kernel_tilelang(
        hidden_states.reshape(-1, HC_MULT, HIDDEN_SIZE),
        hc_fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
    )
    flattened = hidden_states.flatten(-2).float()
    normalized = flattened * torch.rsqrt(
        flattened.square().mean(dim=-1, keepdim=True) + RMS_EPS
    )
    gates = torch.sigmoid(normalized @ hc_fn.transpose(0, 1) * hc_scale + hc_base)
    gates = gates + HC_EPS
    expected = (
        (gates.unsqueeze(-1) * hidden_states.float()).sum(dim=-2).to(torch.bfloat16)
    )
    _report_and_assert_close(
        f"hc_head/tokens={num_tokens}",
        actual,
        expected,
        atol=0.035,
        rtol=0.035,
    )

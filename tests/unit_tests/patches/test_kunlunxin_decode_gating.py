# Copyright (c) 2026 BAAI. All rights reserved.

"""The Kunlunxin decode gating must keep beta in the activation dtype.

The recurrent kernel takes beta in the activation dtype while g follows the ssm
cache, which is float32 on some configs and the activation dtype on others. A
hard-coded bfloat16 cast makes fp16 decode silently wrong, and the sigmoid
itself has to stay in float32 because it saturates in both half precisions.
"""

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_forward_core import (
    _kunlunxin_decode_gating,
)

DTYPES = [torch.float16, torch.bfloat16]


def _inputs(dtype, num_tokens=4, num_heads=2):
    return (
        torch.randn(num_tokens, num_heads, dtype=dtype),
        torch.randn(num_tokens, num_heads, dtype=dtype),
        torch.randn(num_heads),
        torch.randn(num_heads),
    )


@pytest.mark.parametrize("cache_fp32", [False, True], ids=["cache-half", "cache-fp32"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gating_shapes_and_dtypes(dtype, cache_fp32):
    a, b, a_log, dt_bias = _inputs(dtype)
    state_dtype = torch.float32 if cache_fp32 else dtype

    g, beta = _kunlunxin_decode_gating(a, b, a_log, dt_bias, state_dtype)

    assert g.shape == beta.shape == (1, a.shape[0], a.shape[1])
    assert beta.dtype == dtype
    assert g.dtype == (torch.float32 if cache_fp32 else dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_beta_is_sigmoid_b_fp32(dtype):
    """The downcast must not move the value away from sigmoid(b)."""
    a, b, a_log, dt_bias = _inputs(dtype)

    _, beta = _kunlunxin_decode_gating(a, b, a_log, dt_bias, dtype)

    atol = 1e-3 if dtype == torch.float16 else 1e-2
    assert torch.allclose(beta.squeeze(0).float(), torch.sigmoid(b.float()), atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_g_matches_fp32_reference(dtype):
    """g is accumulated in float32 even when the ssm cache is narrower."""
    a, b, a_log, dt_bias = _inputs(dtype)

    g, _ = _kunlunxin_decode_gating(a, b, a_log, dt_bias, torch.float32)

    reference = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
        a.float() + dt_bias.float()
    )
    assert torch.allclose(g.squeeze(0), reference, rtol=1e-5, atol=1e-6)

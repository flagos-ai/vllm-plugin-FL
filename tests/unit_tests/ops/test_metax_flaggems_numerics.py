# Copyright (c) 2026 BAAI. All rights reserved.
"""Real FlagOS-provider numerics on MACA, independent of model execution."""

from types import SimpleNamespace

import pytest
import torch


def _require_metax():
    if not torch.cuda.is_available():
        pytest.skip("Requires a MetaX GPU")
    from tests.utils.device_utils import get_backend

    if get_backend() != "metax":
        pytest.skip("MetaX-only FlagGems provider numerics")


def _rope_reference(value, cos, sin, positions, interleaved):
    c, s = cos[positions].float(), sin[positions].float()
    if interleaved:
        c, s = c.repeat_interleave(2, -1), s.repeat_interleave(2, -1)
        rotated = torch.stack((-value[..., 1::2], value[..., ::2]), -1).flatten(-2)
    else:
        c, s = torch.cat((c, c), -1), torch.cat((s, s), -1)
        left, right = value.chunk(2, -1)
        rotated = torch.cat((-right, left), -1)
    return (value.float() * c[:, None] + rotated.float() * s[:, None]).to(value.dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("rotary_dim", [64, 128])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@torch.inference_mode()
def test_flaggems_bridge_rotary_packed_partial_views(rotary_dim, interleaved, inplace):
    _require_metax()
    from vllm_fl.dispatch.backends.flaggems.impl.rotary import rotary_embedding_flaggems

    device = torch.device("cuda")
    dtype = torch.bfloat16
    owner = torch.randn(7, 10, 128, dtype=dtype, device=device)
    original = owner.clone()
    query, key = owner[:, :8, :rotary_dim], owner[:, 8:, :rotary_dim]
    angles = torch.randn(32, rotary_dim // 2, device=device)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    positions = torch.tensor([0, 1, 3, 7, 16, 21, 30], dtype=torch.long, device=device)
    q_expected = _rope_reference(query, cos, sin, positions, interleaved)
    k_expected = _rope_reference(key, cos, sin, positions, interleaved)
    q_out, k_out = rotary_embedding_flaggems(
        object(),
        query,
        key,
        cos,
        sin,
        positions,
        rotary_interleaved=interleaved,
        inplace=inplace,
    )
    torch.cuda.synchronize()
    q_error = (q_out.float() - q_expected.float()).abs().max().item()
    k_error = (k_out.float() - k_expected.float()).abs().max().item()
    print(
        f"FG-ROPE {rotary_dim=} {interleaved=} {inplace=} max_abs_q={q_error:.6g} max_abs_k={k_error:.6g}"
    )
    torch.testing.assert_close(q_out, q_expected, rtol=0.02, atol=0.04)
    torch.testing.assert_close(k_out, k_expected, rtol=0.02, atol=0.04)
    if inplace:
        assert q_out.data_ptr() == query.data_ptr()
        assert k_out.data_ptr() == key.data_ptr()
        assert q_out.stride() == query.stride()
        assert k_out.stride() == key.stride()
        torch.testing.assert_close(
            owner[..., rotary_dim:], original[..., rotary_dim:], rtol=0, atol=0
        )
    else:
        torch.testing.assert_close(owner, original, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(7, 1024), (7, 2560), (7, 4, 128)])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_flaggems_rms_and_residual_against_float32(shape, has_residual, strided):
    _require_metax()
    from vllm_fl.dispatch.backends.flaggems.impl.normalization import rms_norm_flaggems

    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden = shape[-1]
    padded_shape = (*shape[:-1], hidden + 32) if strided else shape
    owner = torch.randn(padded_shape, dtype=dtype, device=device)
    x = owner[..., :hidden]
    original = owner.clone()
    residual = torch.randn_like(x) if has_residual else None
    weight = torch.randn(hidden, dtype=dtype, device=device)
    eps = 1e-6
    work = x.float()
    if residual is not None:
        work = work + residual.float()
    expected_residual = work.to(dtype)
    expected = (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + eps)).to(
        dtype
    ) * weight
    result = rms_norm_flaggems(
        SimpleNamespace(weight=weight, variance_epsilon=eps), x, residual
    )
    actual = result[0] if has_residual else result
    torch.cuda.synchronize()
    error = (actual.float() - expected.float()).abs().max().item()
    print(f"FG-RMS {shape=} {has_residual=} {strided=} max_abs_error={error:.6g}")
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.05)
    if has_residual:
        # The FL RMS interface returns both tensors; it does not promise the
        # native _C kernel's aliases. A higher-rank provider may compact views.
        torch.testing.assert_close(result[1], expected_residual, rtol=0, atol=0)
    else:
        torch.testing.assert_close(x, original[..., :hidden], rtol=0, atol=0)
    torch.testing.assert_close(
        owner[..., hidden:], original[..., hidden:], rtol=0, atol=0
    )

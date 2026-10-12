# Copyright 2026 FlagOS Contributors
"""Compare operator call subsets against the NVIDIA FlagGems baseline.

These comparisons cover the plugin's call subset, not equivalence of every
optional argument or vendor implementation. Ascend's retained RoPE path has
its own contract tests in test_ascend_rotary.py.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]


@pytest.fixture(scope="module")
def providers(backend):
    if backend != "nvidia":
        pytest.skip("comparison targets the NVIDIA FlagGems migration baseline")

    import flaggems_vllm
    from flag_gems.modules.activation import gems_silu_and_mul
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    return SimpleNamespace(
        silu=(gems_silu_and_mul, flaggems_vllm.silu_and_mul),
        rotary=(gems_rope_forward, flaggems_vllm.apply_rotary_pos_emb),
    )


@pytest.mark.parametrize("op", ["silu", "rotary"])
def test_current_calls_bind_to_both_signatures(providers, op):
    value = object()
    calls = {
        "silu": ((value, value), {}),
        "rotary": (
            (value, value, value, value),
            {"position_ids": value, "rotary_interleaved": False, "inplace": True},
        ),
    }
    args, kwargs = calls[op]
    for fn in getattr(providers, op):
        inspect.signature(fn).bind(*args, **kwargs)


def _assert_output(actual, expected, device, dtype):
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert actual.device == device
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 256), (17, 256), (2, 3, 256)])
def test_silu_matches_original_provider(providers, device, dtype, shape):
    generator = torch.Generator().manual_seed(13)
    packed = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    before = packed.clone()
    x, y = packed.chunk(2, dim=-1)
    expected = (F.silu(x.cpu().float()) * y.cpu().float()).to(dtype)
    old, new = (fn(x, y) for fn in providers.silu)
    _assert_output(old, expected, device, dtype)
    _assert_output(new, expected, device, dtype)
    _assert_output(new, old.cpu(), device, dtype)
    torch.testing.assert_close(packed, before, rtol=0, atol=0)


def _rotary_reference(value, cos, sin, positions, interleaved):
    value = value.cpu().float()
    cos, sin = (
        cos.cpu()[positions].float()[:, None],
        sin.cpu()[positions].float()[:, None],
    )
    if interleaved:
        first, second = value[..., ::2], value[..., 1::2]
        return torch.stack(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        ).flatten(-2)
    first, second = value.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("packed_views", [False, True])
def test_rotary_matches_original_values_and_mutation(
    providers, device, dtype, interleaved, inplace, packed_views
):
    generator = torch.Generator().manual_seed(17)
    original = torch.randn((7, 6, 160), generator=generator).to(dtype)
    angles = torch.randn((16, 40), generator=generator)
    cos, sin = (
        angles.cos().to(device=device, dtype=dtype),
        angles.sin().to(device=device, dtype=dtype),
    )
    positions_cpu = torch.tensor([3, 0, 7, 4, 9, 1, 5])
    positions = positions_cpu.to(device)
    outputs = []
    for fn in providers.rotary:
        packed = original.to(device).clone()
        query, key = packed[:, :4, :80], packed[:, 4:, :80]
        if not packed_views:
            query, key = query.contiguous(), key.contiguous()
        before = [query.clone(), key.clone()]
        actual = fn(
            query,
            key,
            cos,
            sin,
            position_ids=positions,
            rotary_interleaved=interleaved,
            inplace=inplace,
        )
        assert isinstance(actual, tuple) and len(actual) == 2
        for output, value, saved in zip(actual, (query, key), before):
            expected = _rotary_reference(
                saved, cos, sin, positions_cpu, interleaved
            ).to(dtype)
            _assert_output(output, expected, device, dtype)
            if inplace:
                # The generic export may return a new view; check storage
                # aliasing and mutation, rather than Python object identity.
                assert output.data_ptr() == value.data_ptr()
                assert output.stride() == value.stride()
                _assert_output(value, expected, device, dtype)
            else:
                assert output.data_ptr() != value.data_ptr()
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
        if packed_views and inplace:
            torch.testing.assert_close(
                packed[:, :, 80:].cpu(), original[:, :, 80:], rtol=0, atol=0
            )
        else:
            torch.testing.assert_close(packed.cpu(), original, rtol=0, atol=0)
        outputs.append(actual)
    for old, new in zip(*outputs):
        _assert_output(new, old.cpu(), device, dtype)

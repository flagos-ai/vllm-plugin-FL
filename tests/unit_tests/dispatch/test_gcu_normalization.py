"""CPU rounding contracts; real device precision is checked separately."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_PATH = Path(__file__).parents[3] / 'vllm_fl/dispatch/backends/vendor/gcu/impl/normalization.py'
_SPEC = importlib.util.spec_from_file_location('_gcu_norm_contract', _PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('add', [False, True])
@pytest.mark.parametrize('weight', [False, True])
@pytest.mark.parametrize('partial', [False, True])
def test_rounding_contract(dtype, add, weight, partial):
    generator = torch.Generator(device='cpu').manual_seed(37)
    x = torch.randn(3, 16, generator=generator, device='cpu').to(dtype)
    r = torch.randn(3, 16, generator=generator, device='cpu').to(dtype) if add else None
    w = torch.randn(16, generator=generator, device='cpu').to(dtype)
    obj = SimpleNamespace(weight=w, variance_epsilon=1e-6,
                          variance_size_override=8 if partial else None,
                          pass_weight=weight, pass_weight_add=weight)
    original_x = x.clone()
    original_r = r.clone() if add else None
    actual = _MOD.rms_norm_gcu(obj, x, r)
    summed = (x.float() + r.float()).to(dtype) if add else x
    full = summed.float()
    expected = full * torch.rsqrt(full[..., :8 if partial else 16].square().mean(-1, keepdim=True) + 1e-6)
    if weight:
        expected = expected * w.float()
    result = actual[0] if add else actual
    torch.testing.assert_close(result, expected.to(dtype), rtol=0, atol=0)
    # FP64 is a numerical oracle, not the specified FP32 rounding contract.
    high = summed.double()
    high = high * torch.rsqrt(high[..., :8 if partial else 16].square().mean(-1, keepdim=True) + 1e-6)
    if weight:
        high = high * w.double()
    torch.testing.assert_close(result.double(), high, rtol=2 * torch.finfo(dtype).eps, atol=torch.finfo(dtype).eps)
    assert result.dtype == dtype
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    if add:
        torch.testing.assert_close(actual[1], summed, rtol=0, atol=0)
        torch.testing.assert_close(r, original_r, rtol=0, atol=0)


def test_bf16_weight_multiply_does_not_round_twice():
    generator = torch.Generator(device='cpu').manual_seed(37)
    x = torch.randn(4, 32, generator=generator, device='cpu').bfloat16()
    w = torch.randn(32, generator=generator, device='cpu').bfloat16()
    obj = SimpleNamespace(weight=w, variance_epsilon=1e-6)
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    once = (normalized * w.float()).bfloat16()
    twice = normalized.bfloat16() * w
    assert not torch.equal(once, twice), 'fixture must expose the rounding regression'
    torch.testing.assert_close(_MOD.rms_norm_gcu(obj, x), once, rtol=0, atol=0)

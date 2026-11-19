"""Preserve Gemma weight/residual semantics across PCP tensor layouts."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.cp.fused_norm import prefill_gemma_norm


def reference(x, weight, residual, eps):
    if residual is not None:
        x = x + residual
    summed = x
    xf = x.float()
    out = (
        xf
        * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        * (weight.float() + 1)
    ).to(x.dtype)
    return out if residual is None else (out, summed)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(0, 2048), (31, 127), (257, 2048), (33, 16, 256)])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_prefill_norm_values_and_input_ownership(shape, has_residual, strided):
    torch.manual_seed(42)
    if strided:
        storage = torch.randn(
            (*shape[:-1], shape[-1] * 4), device="cuda", dtype=torch.bfloat16
        )
        x = storage[..., ::4]
    else:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if has_residual else None
    weight = torch.randn(shape[-1] * 2, device="cuda", dtype=x.dtype)[::2]
    module = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    before = x.clone()
    before_res = residual.clone() if has_residual else None
    expected = reference(x, weight, residual, module.variance_epsilon)
    actual = prefill_gemma_norm(module, x, residual)
    if has_residual:
        assert torch.equal(actual[1], expected[1])
        assert torch.equal(residual, before_res)
        actual, expected = actual[0], expected[0]
    # The native mean reduction and separate BF16 residual rounding remain.
    assert torch.equal(actual, expected)
    assert actual.is_contiguous()
    assert torch.equal(x, before)


def test_non_cuda_norm_uses_original_module():
    calls = []

    class Norm:
        def __call__(self, *args):
            calls.append(args)
            return "native"

    x = torch.ones(2, 3)
    assert prefill_gemma_norm(Norm(), x) == "native"
    assert prefill_gemma_norm(Norm(), x, x) == "native"
    assert len(calls[0]) == 1 and len(calls[1]) == 2


def test_non_prefill_layer_dispatches_to_native(monkeypatch):
    from vllm_fl.cp import prstyle

    sentinel = object()
    monkeypatch.setattr(prstyle, "get_forward_context", lambda: None)
    monkeypatch.setattr(prstyle, "_context_metadata", lambda _: None)
    monkeypatch.setattr(prstyle, "_NATIVE_LAYER_FORWARD", lambda *a, **k: sentinel)
    monkeypatch.setattr(
        prstyle, "prefill_gemma_norm", lambda *a: pytest.fail("decode called fusion")
    )
    assert prstyle._pr_layer_forward(None, None, None) is sentinel

# Copyright (c) 2026 BAAI. All rights reserved.
"""Real PTPU numerical checks for the historical provider math namespace.

These tests launch the pinned provider's kernels and compare CPU references.
The b5 provider's public swiglu wrapper rejects non-CUDA tensors; its unchanged
kernel is checked directly, while the serving path uses silu_and_mul.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")
pytestmark = [pytest.mark.gpu, pytest.mark.functional]
_vendor_pow = None


@triton.jit
def _integer_pow_kernel(input_ptr, output_ptr, n: tl.constexpr, exponent: tl.constexpr):
    offsets = tl.arange(0, 32)
    value = tl.load(input_ptr + offsets, offsets < n, other=0).to(tl.float32)
    result = _vendor_pow(value, exponent)
    tl.store(output_ptr + offsets, result, offsets < n)


@pytest.fixture(scope="module")
def sunrise_provider():
    from vllm.platforms import current_platform

    if current_platform.device_type != "ptpu":
        pytest.skip("Requires the Sunrise/PTPU compiler stack")
    from vllm_fl.patches.sunrise_compiler import install_sunrise_libdevice_compat

    math = install_sunrise_libdevice_compat()
    globals()["_vendor_pow"] = math.pow
    torch.ptpu.set_device(0)
    provider = importlib.import_module("flaggems_vllm")
    swiglu = importlib.import_module("flaggems_vllm.ops.swiglu")
    return SimpleNamespace(provider=provider, swiglu=swiglu, device="ptpu:0")


def _inputs(dtype):
    values = [-6, -2.5, -1, -0.25, -0.0, 0.0, 0.25, 1, 2.5, 6]
    x = torch.tensor(values, dtype=dtype, device="cpu").repeat(3, 1)
    y = torch.tensor([2, -1, 0.5, 0, -2, 1, -0.25, 3, -1.5, 0.75], dtype=dtype)
    return x, y.repeat(3, 1)


def _assert_close(actual, expected, dtype):
    tolerance = {
        torch.float32: (3e-5, 3e-6),
        torch.float16: (3e-3, 3e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }
    rtol, atol = tolerance[dtype]
    torch.testing.assert_close(
        actual.detach().cpu().float(), expected.to(dtype).float(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("exponent", [2, 3])
def test_vendor_pow_signed_integer_inputs(sunrise_provider, exponent):
    x, _ = _inputs(torch.float32)
    x = x[0].contiguous()
    actual = torch.empty_like(x, device=sunrise_provider.device)
    _integer_pow_kernel[(1,)](
        x.to(sunrise_provider.device), actual, x.numel(), exponent
    )
    actual = actual.cpu()
    expected = x.pow(exponent)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    # An exp/log approximation fails negative bases; signed zero also matters.
    zero = x == 0
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_provider_gelu_and_mul(sunrise_provider, dtype, approximate):
    x, y = _inputs(dtype)
    actual = sunrise_provider.provider.gelu_and_mul(
        x.to(sunrise_provider.device), y.to(sunrise_provider.device), approximate
    )
    expected = F.gelu(x.float(), approximate=approximate) * y.float()
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_provider_geglu_forward_and_backward(sunrise_provider, dtype):
    x, y = _inputs(dtype)
    packed = torch.cat((x, y), dim=-1)
    upstream = torch.linspace(-1, 1, x.numel(), dtype=dtype).reshape_as(x)
    packed_device = packed.to(sunrise_provider.device)
    actual = sunrise_provider.provider.geglu(packed_device)
    actual_grad = sunrise_provider.provider.dgeglu(
        upstream.to(sunrise_provider.device), packed_device
    )
    reference_input = packed.float().requires_grad_(True)
    ref_x, ref_y = reference_input.chunk(2, dim=-1)
    reference = F.gelu(ref_x, approximate="tanh") * ref_y
    reference.backward(upstream.float())
    _assert_close(actual, reference, dtype)
    _assert_close(actual_grad, reference_input.grad, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_provider_swiglu_and_serving_silu(sunrise_provider, dtype):
    x, y = _inputs(dtype)
    packed = torch.cat((x, y), dim=-1).to(sunrise_provider.device)
    output = torch.empty_like(x, device=sunrise_provider.device)
    m, h = x.shape
    # Execute the actual public provider kernel without altering its CUDA-only
    # Python wrapper or claiming that wrapper accepts PTPU.
    sunrise_provider.swiglu.swiglu_kernel[(triton.cdiv(m, 64), triton.cdiv(h, 64))](
        packed,
        output,
        m,
        h,
        packed.stride(0),
        packed.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_H=64,
    )
    expected = F.silu(x.float()) * y.float()
    _assert_close(output, expected, dtype)
    serving = sunrise_provider.provider.silu_and_mul(
        x.to(sunrise_provider.device), y.to(sunrise_provider.device)
    )
    _assert_close(serving, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_provider_tanh_forward_and_backward(sunrise_provider, dtype):
    x, upstream = _inputs(dtype)
    actual = sunrise_provider.provider.tanh(x.to(sunrise_provider.device))
    actual_grad = sunrise_provider.provider.tanh_backward(
        upstream.to(sunrise_provider.device), actual
    )
    expected = torch.tanh(x.float())
    # The backward kernel consumes the already-rounded forward output.
    expected_grad = upstream.float() * (1 - expected.to(dtype).float().square())
    _assert_close(actual, expected, dtype)
    _assert_close(actual_grad, expected_grad, dtype)

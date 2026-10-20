# Copyright (c) 2026 BAAI. All rights reserved.

"""Clamped SwiGLU's EMPTY CUDA ABI preserves all native scalar parameters."""

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA EMPTY activation regression"
)

_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
_PARAMS = [(10.0, 1.0, 0.0), (0.7, 1.0, 0.0), (2.3, 1.702, 1.0)]


@pytest.fixture
def gems_policy(monkeypatch):
    import flag_gems

    from vllm.platforms import current_platform

    from vllm_fl.dispatch.config.utils import load_platform_config
    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    assert isinstance(current_platform, NvidiaPlatformFL)

    def fail_registration(error):
        raise RuntimeError(f"FlagGems registration failed: {error}")

    monkeypatch.setattr(flag_gems.runtime.error, "register_error", fail_registration)
    return flag_gems, load_platform_config("nvidia")["flagos_blacklist"]


def _reference(cpu_input, limit, alpha, beta):
    gate, up = cpu_input.float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(cpu_input.dtype)


def _check(actual, expected):
    tolerance = {
        torch.float16: (0.002, 0.0002),
        torch.bfloat16: (0.012, 0.001),
        torch.float32: (2e-5, 2e-6),
    }
    rtol, atol = tolerance[actual.dtype]
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("params", _PARAMS)
@pytest.mark.parametrize("strided", [False, True])
def test_clamped_swiglu_cuda_abi_matches_cpu_reference(
    gems_policy, dtype, params, strided
):
    flag_gems, blacklist = gems_policy
    cpu_input = torch.linspace(-14, 14, 2 * 3 * 128).reshape(2, 3, 128).to(dtype)
    x = cpu_input.cuda()
    if strided:
        x = x[..., ::2]
        cpu_input = cpu_input[..., ::2]
    width = x.shape[-1] // 2
    storage = torch.full((*x.shape[:-1], width * 2), -99.0, dtype=dtype, device="cuda")
    out = storage[..., ::2] if strided else torch.empty_like(x[..., :width])
    with flag_gems.use_gems(exclude=blacklist):
        torch.ops._C.silu_and_mul_with_clamp(out, x, *params)
    _check(out, _reference(cpu_input, *params))
    if strided:
        assert torch.all(storage[..., 1::2] == -99.0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("params", _PARAMS)
def test_clamped_swiglu_cuda_graph_replays_new_input(gems_policy, dtype, params):
    flag_gems, blacklist = gems_policy
    cpu_input = torch.linspace(-12, 12, 17 * 64).reshape(17, 64).to(dtype)
    x = cpu_input.cuda()
    out = torch.empty((17, 32), dtype=dtype, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with flag_gems.use_gems(exclude=blacklist):
        with torch.cuda.stream(stream):
            torch.ops._C.silu_and_mul_with_clamp(out, x, *params)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.ops._C.silu_and_mul_with_clamp(out, x, *params)
        graph.replay()
    _check(out, _reference(cpu_input, *params))
    cpu_input = -cpu_input.flip(-1)
    x.copy_(cpu_input)
    graph.replay()
    _check(out, _reference(cpu_input, *params))


@pytest.mark.gpu
def test_upstream_clamped_activation_uses_registered_empty_abi(gems_policy):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    flag_gems, blacklist = gems_policy
    x_cpu = torch.tensor([[-12.0, 14.0, 13.0, -14.0]], dtype=torch.bfloat16)
    with set_current_vllm_config(VllmConfig()), flag_gems.use_gems(exclude=blacklist):
        layer = SiluAndMulWithClamp(10.0)
        result = layer.forward_cuda(x_cpu.cuda())
    _check(result, _reference(x_cpu, 10.0, 1.0, 0.0))


def test_activation_registration_preserves_native_wheel(monkeypatch):
    from vllm_fl.ops import nvidia_activation_ops

    monkeypatch.setattr(nvidia_activation_ops, "has_native_vllm_kernels", lambda: True)
    assert nvidia_activation_ops.register_nvidia_activation_ops() is False

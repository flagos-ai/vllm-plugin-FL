# Copyright (c) 2026 BAAI. All rights reserved.

"""Exercise real upstream rotary dispatch and EMPTY CUDA math/graph behavior."""

import inspect

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "nvidia", reason="NVIDIA rotary")


@pytest.fixture
def rotary_provider():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_fl.ops import nvidia_rotary_ops as provider

    if provider.has_native_vllm_kernels():
        pytest.skip("Native wheel supplies its own rotary operation")
    previous = op_registry_oot.pop("ApplyRotaryEmb", None)
    config = VllmConfig()
    config.compilation_config.custom_ops = ["all"]
    try:
        with set_current_vllm_config(config):
            yield provider
    finally:
        op_registry_oot.pop("ApplyRotaryEmb", None)
        if previous is not None:
            op_registry_oot["ApplyRotaryEmb"] = previous


def _rotary(provider, neox, fp32):
    assert provider.register_nvidia_rotary_ops()
    layer = provider.ApplyRotaryEmb(
        enforce_enable=True,
        is_neox_style=neox,
        enable_fp32_compute=fp32,
    )
    assert type(layer) is provider.NvidiaApplyRotaryEmb
    assert layer._forward_method.__func__ is provider.NvidiaApplyRotaryEmb.forward_cuda
    return layer


def test_upstream_interface_and_registration_idempotence(rotary_provider):
    provider = rotary_provider
    layer = _rotary(provider, True, False)
    assert provider.register_nvidia_rotary_ops()
    assert provider.op_registry_oot["ApplyRotaryEmb"] is type(layer)
    assert type(layer).__init__ is provider.ApplyRotaryEmb.__init__
    assert type(layer).forward_native is provider.ApplyRotaryEmb.forward_native
    assert type(layer).name == provider.ApplyRotaryEmb.name
    assert inspect.signature(type(layer).forward_cuda) == inspect.signature(
        provider.ApplyRotaryEmb.forward_cuda
    )


def test_registration_preserves_user_provider(rotary_provider):
    provider = rotary_provider

    class UserRotary(provider.ApplyRotaryEmb):
        pass

    provider.op_registry_oot["ApplyRotaryEmb"] = UserRotary
    assert not provider.register_nvidia_rotary_ops()
    assert provider.op_registry_oot["ApplyRotaryEmb"] is UserRotary


def test_registration_preserves_native_wheel(rotary_provider, monkeypatch):
    monkeypatch.setattr(rotary_provider, "has_native_vllm_kernels", lambda: True)
    assert not rotary_provider.register_nvidia_rotary_ops()
    assert "ApplyRotaryEmb" not in rotary_provider.op_registry_oot


def test_registration_is_cuda_only(rotary_provider, monkeypatch):
    monkeypatch.setattr(rotary_provider.current_platform, "is_cuda", lambda: False)
    assert not rotary_provider.register_nvidia_rotary_ops()
    assert "ApplyRotaryEmb" not in rotary_provider.op_registry_oot


@pytest.mark.parametrize("enabled", [False, True])
def test_rotary_preserves_upstream_compilation_directives(rotary_provider, enabled):
    from vllm.config import get_cached_compilation_config

    provider = rotary_provider
    config = get_cached_compilation_config()
    config.custom_ops = (
        ["none", "+apply_rotary_emb"] if enabled else ["all", "-apply_rotary_emb"]
    )
    assert provider.register_nvidia_rotary_ops()
    layer = provider.ApplyRotaryEmb()
    assert type(layer) is provider.NvidiaApplyRotaryEmb
    expected_method = (
        provider.NvidiaApplyRotaryEmb.forward_cuda
        if enabled
        else provider.ApplyRotaryEmb.forward_native
    )
    assert layer._forward_method.__func__ is expected_method


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("fp32", [False, True])
def test_rotary_cuda_matches_upstream_cpu(rotary_provider, dtype, batched, neox, fp32):
    provider = rotary_provider
    layer = _rotary(provider, neox, fp32)
    torch.manual_seed(19)
    if batched:
        x = torch.randn(2, 7, 3, 8, device="cuda", dtype=dtype)
    else:
        x = torch.randn(14, 3, 8, device="cuda", dtype=dtype)[::2]
        assert not x.is_contiguous()
    angles = torch.randn(7, 4, device="cuda", dtype=torch.float32)
    cos, sin = angles.cos(), angles.sin()
    original = x.cpu()
    expected = provider.ApplyRotaryEmb.forward_static(
        original, cos.cpu(), sin.cpu(), neox, fp32
    )
    output = layer(x, cos, sin)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    torch.testing.assert_close(output.cpu(), expected)
    torch.testing.assert_close(x.cpu(), original, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("fp32", [False, True])
def test_rotary_gems_graph_replay_reads_new_inputs(rotary_provider, neox, fp32):
    import flag_gems

    from vllm_fl.utils import get_flag_gems_whitelist_blacklist

    provider = rotary_provider
    layer = _rotary(provider, neox, fp32)
    x = torch.randn(2, 7, 3, 8, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(7, 4, device="cuda")
    cos, sin = angles.cos(), angles.sin()
    new_x = torch.randn_like(x)
    new_angles = torch.randn_like(angles)
    new_cos, new_sin = new_angles.cos(), new_angles.sin()
    expected = provider.ApplyRotaryEmb.forward_static(
        new_x.cpu(), new_cos.cpu(), new_sin.cpu(), neox, fp32
    )
    _, blacklist = get_flag_gems_whitelist_blacklist()
    with flag_gems.use_gems(exclude=blacklist):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            layer(x, cos, sin)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(x, cos, sin)
        x.copy_(new_x)
        cos.copy_(new_cos)
        sin.copy_(new_sin)
        graph.replay()
    torch.testing.assert_close(output.cpu(), expected)


def test_worker_registers_empty_rotary_after_fl_runtime(rotary_provider, monkeypatch):
    from vllm_fl.worker import nvidia_worker

    events = []

    def native_init(worker):
        worker.rank = 0
        events.append("native")

    def fl_runtime(rank):
        assert rank == 0
        assert "ApplyRotaryEmb" not in rotary_provider.op_registry_oot
        events.append("fl")

    monkeypatch.setattr(nvidia_worker.NativeGPUWorker, "__init__", native_init)
    monkeypatch.setattr(nvidia_worker, "_initialize_fl_runtime", fl_runtime)
    nvidia_worker.NvidiaWorkerFL()
    assert events == ["native", "fl"]
    assert (
        rotary_provider.op_registry_oot["ApplyRotaryEmb"]
        is rotary_provider.NvidiaApplyRotaryEmb
    )

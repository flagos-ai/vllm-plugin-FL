"""Exercise the integrated clamp guard without launching GPU kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.ops.fused_moe import fused_moe_utils as moe


@pytest.mark.parametrize("error", [RuntimeError, torch.OutOfMemoryError])
def test_clamped_activation_propagates_execution_failure(monkeypatch, error):
    import flaggems_vllm

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_fl.ops.fused_moe.activation import apply_moe_activation

    fail = Mock(side_effect=error("clamp launch failed"))
    monkeypatch.setattr(flaggems_vllm, "silu_and_mul_with_clamp_out", fail)
    # Meta tensors exercise launch dispatch without allocating accelerator memory.
    with pytest.raises(error, match="clamp launch failed"):
        apply_moe_activation(
            MoEActivation.SILU,
            torch.empty(2, 3, device="meta"),
            torch.ones(2, 6, device="meta"),
            clamp_limit=7.0,
        )
    fail.assert_called_once()


def test_clamped_moe_resolves_platform_at_apply_time(monkeypatch):
    expert = object.__new__(moe.TritonExpertsFL)
    expert.quant_config = SimpleNamespace(gemm1_clamp_limit=7.0)
    runtime = SimpleNamespace(is_cuda=lambda: True)
    resolver = Mock(side_effect=lambda: runtime)
    monkeypatch.setattr(moe, "_get_current_platform", resolver)
    monkeypatch.setattr(moe, "has_native_triton_moe", lambda: True)
    sentinel = object()
    calls = []

    def native_apply(self, *args):
        calls.append((self, args))
        return sentinel

    monkeypatch.setattr(moe.TritonExperts, "apply", native_apply)
    args = tuple(object() for _ in range(15))
    assert expert.apply(*args) is sentinel
    assert calls == [(expert, args)]
    resolver.assert_called_once_with()

    # A later platform must be resolved again, not cached at import time.
    class PlatformChanged(Exception):
        pass

    def changed_platform():
        raise PlatformChanged

    runtime = SimpleNamespace(is_cuda=changed_platform)
    with pytest.raises(PlatformChanged):
        expert.apply(*args)
    assert resolver.call_count == 2
    assert len(calls) == 1


@pytest.mark.parametrize("rocm", [True, False])
def test_priority_uses_runtime_platform(monkeypatch, rocm):
    runtime = SimpleNamespace(
        is_rocm=lambda: rocm,
        is_cuda=lambda: False,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
    )
    monkeypatch.setattr(moe, "_get_current_platform", lambda: runtime)
    config = SimpleNamespace(moe_parallel_config=SimpleNamespace(dp_size=1))
    expected = [
        moe.UnquantizedMoeBackend.TRITON,
        moe.UnquantizedMoeBackend.BATCHED_TRITON,
    ]
    if rocm:
        expected.insert(0, moe.UnquantizedMoeBackend.AITER)
    assert moe._get_priority_backends(config) == expected


@pytest.mark.parametrize("direct_output", [False, True])
def test_silu_output_path_is_executed_once(monkeypatch, direct_output):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_fl.ops.fused_moe import activation as module

    inputs = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 10
    output = torch.full((2, 3), float("nan"))
    expected = torch.nn.functional.silu(inputs[:, :3]) * inputs[:, 3:]
    default = Mock(return_value=expected)

    def write_output(input_tensor, output_tensor):
        output_tensor.copy_(expected)

    direct = Mock(side_effect=write_output)
    monkeypatch.setattr(module, "_silu_and_mul", default)
    result = module.apply_moe_activation(
        MoEActivation.SILU,
        output,
        inputs,
        silu_out=direct if direct_output else None,
    )
    assert result is output
    torch.testing.assert_close(output, expected)
    if direct_output:
        direct.assert_called_once_with(inputs, output)
        default.assert_not_called()
    else:
        default.assert_called_once_with(None, inputs)
        direct.assert_not_called()


def test_silu_output_failure_is_not_retried(monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_fl.ops.fused_moe import activation as module

    default = Mock()
    direct = Mock(side_effect=RuntimeError("direct output failed"))
    monkeypatch.setattr(module, "_silu_and_mul", default)
    with pytest.raises(RuntimeError, match="direct output failed"):
        module.apply_moe_activation(
            MoEActivation.SILU,
            torch.empty(2, 3),
            torch.ones(2, 6),
            silu_out=direct,
        )
    direct.assert_called_once()
    default.assert_not_called()


def test_clamped_cpu_activation_preserves_clamp_with_output_callback(monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_fl.ops.fused_moe import activation as module

    inputs = torch.ones(2, 6)
    output = torch.empty(2, 3)
    clamped = Mock(return_value=output)
    direct = Mock()
    default = Mock()
    monkeypatch.setattr(module, "upstream_apply_moe_activation", clamped)
    monkeypatch.setattr(module, "_silu_and_mul", default)
    result = module.apply_moe_activation(
        MoEActivation.SILU, output, inputs, clamp_limit=7.0, silu_out=direct
    )
    assert result is output
    clamped.assert_called_once_with(MoEActivation.SILU, output, inputs, clamp_limit=7.0)
    direct.assert_not_called()
    default.assert_not_called()

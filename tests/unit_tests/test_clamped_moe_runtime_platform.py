"""Exercise the integrated clamp guard without launching GPU kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.ops.fused_moe import fused_moe_utils as moe


@pytest.mark.parametrize("error", [RuntimeError, torch.OutOfMemoryError])
def test_clamped_activation_propagates_execution_failure(monkeypatch, error):
    import importlib

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_fl.ops.fused_moe.activation import apply_moe_activation

    kernel = importlib.import_module("flag_gems.fused.silu_and_mul_with_clamp")

    def fail(*args):
        raise error("clamp launch failed")

    monkeypatch.setattr(kernel, "silu_and_mul_with_clamp_out", fail)
    with pytest.raises(error, match="clamp launch failed"):
        apply_moe_activation(
            MoEActivation.SILU, torch.empty(2, 3), torch.ones(2, 6), 7.0
        )


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

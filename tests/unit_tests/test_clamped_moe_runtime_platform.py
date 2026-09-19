"""Exercise the integrated clamp guard without launching GPU kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.ops.fused_moe import fused_moe_utils as moe


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

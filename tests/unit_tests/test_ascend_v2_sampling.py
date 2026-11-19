# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for binding the provider's Ascend V2 sampling APIs."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


class _RejectOneWrite(ModuleType):
    blocked_name = None

    def __setattr__(self, name, value):
        if name == self.blocked_name:
            object.__setattr__(self, "blocked_name", None)
            raise RuntimeError(f"write rejected: {name}")
        super().__setattr__(name, value)


@pytest.fixture
def sampling(monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/ascend/patches/ascend_v2_sampling.py"
    )
    spec = importlib.util.spec_from_file_location("ascend_v2_sampling_under_test", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    upstream = Mock()
    modules = SimpleNamespace(
        penalties=ModuleType("vllm.v1.worker.gpu.sample.penalties"),
        gumbel=_RejectOneWrite("vllm.v1.worker.gpu.sample.gumbel"),
        states=ModuleType("vllm.v1.worker.gpu.sample.states"),
        platforms=ModuleType("vllm.platforms"),
        provider=ModuleType("flaggems_vllm"),
        rng=ModuleType("flaggems_vllm.ops.v2_gumbel_rng"),
    )
    modules.platforms.current_platform = SimpleNamespace(device_type="npu")
    modules.penalties.apply_penalties = upstream
    modules.penalties._penalties_kernel = upstream
    modules.gumbel.apply_temperature = upstream
    modules.gumbel.gumbel_noised_argmax = upstream
    modules.gumbel.gumbel_sample = upstream
    modules.gumbel._murmur3_mix = upstream
    modules.states.apply_temperature = upstream
    modules.provider.v2_apply_penalties = Mock(name="v2_apply_penalties")
    modules.provider.v2_apply_temperature = Mock(name="v2_apply_temperature")
    modules.provider.v2_gumbel_rng = modules.rng
    for name in ("murmur3_hash32", "murmur3_uniform32", "murmur3_uniform64"):
        setattr(modules.gumbel, name, upstream)
        setattr(modules.rng, name, Mock(name=name))
    for module in (
        modules.penalties,
        modules.gumbel,
        modules.states,
        modules.platforms,
        modules.provider,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(
        install=adapter.install_ascend_v2_sampling_shims,
        modules=modules,
        upstream=upstream,
    )


def _bindings(modules):
    return [
        modules.penalties.apply_penalties,
        modules.gumbel.apply_temperature,
        modules.states.apply_temperature,
        modules.gumbel.murmur3_hash32,
        modules.gumbel.murmur3_uniform32,
        modules.gumbel.murmur3_uniform64,
    ]


def test_uses_provider_callables_and_preserves_upstream_kernel_and_sampler(sampling):
    m = sampling.modules
    assert sampling.install()
    assert sampling.install()
    assert _bindings(m) == [
        m.provider.v2_apply_penalties,
        m.provider.v2_apply_temperature,
        m.provider.v2_apply_temperature,
        m.rng.murmur3_hash32,
        m.rng.murmur3_uniform32,
        m.rng.murmur3_uniform64,
    ]
    assert m.penalties._penalties_kernel is sampling.upstream
    assert m.gumbel.gumbel_noised_argmax is sampling.upstream
    assert m.gumbel.gumbel_sample is sampling.upstream
    assert m.gumbel._murmur3_mix is sampling.upstream


def test_passes_the_original_logits_and_state_without_copying(sampling):
    m = sampling.modules
    assert sampling.install()
    logits = torch.zeros((2, 1044))[:, :1031]
    mapping = torch.tensor([1, 0], dtype=torch.int32)
    tokens = torch.tensor([31, 1024], dtype=torch.int32)
    position = torch.tensor([0, 0], dtype=torch.int32)
    repetition = torch.tensor([1.2, 1.7])
    frequency = torch.tensor([0.4, -0.25])
    presence = torch.tensor([-0.3, 0.6])
    mask = torch.zeros((2, 35), dtype=torch.int32)[:, :33]
    counts = torch.zeros((2, 1038), dtype=torch.int32)[:, :1031]
    inputs = (
        logits,
        mapping,
        tokens,
        position,
        repetition,
        frequency,
        presence,
        mask,
        counts,
    )
    m.penalties.apply_penalties(*inputs)
    assert all(
        a is b
        for a, b in zip(
            m.provider.v2_apply_penalties.call_args.args, inputs, strict=True
        )
    )

    scale = torch.tensor([0.0, 0.7])
    m.states.apply_temperature(logits, mapping, scale)
    assert all(
        a is b
        for a, b in zip(
            m.provider.v2_apply_temperature.call_args.args,
            (logits, mapping, scale),
            strict=True,
        )
    )


def test_other_platform_does_not_load_provider(sampling, monkeypatch):
    m = sampling.modules
    m.platforms.current_platform.device_type = "cuda"
    monkeypatch.setitem(sys.modules, m.provider.__name__, None)
    assert not sampling.install()
    assert all(binding is sampling.upstream for binding in _bindings(m))


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("provider", "v2_apply_penalties"),
        ("provider", "v2_apply_temperature"),
        ("provider", "v2_gumbel_rng"),
        ("rng", "murmur3_uniform64"),
    ],
)
def test_incomplete_provider_leaves_upstream_bindings_intact(
    sampling, monkeypatch, module, name
):
    m = sampling.modules
    monkeypatch.delattr(getattr(m, module), name)
    with pytest.raises(RuntimeError, match=name):
        sampling.install()
    assert all(binding is sampling.upstream for binding in _bindings(m))


def test_missing_upstream_binding_does_not_partially_install(sampling, monkeypatch):
    m = sampling.modules
    monkeypatch.delattr(m.states, "apply_temperature")
    with pytest.raises(AttributeError, match="apply_temperature"):
        sampling.install()
    assert m.penalties.apply_penalties is sampling.upstream
    assert m.gumbel.apply_temperature is sampling.upstream
    assert m.gumbel.murmur3_uniform64 is sampling.upstream


def test_install_rolls_back_already_replaced_bindings(sampling):
    m = sampling.modules
    m.gumbel.blocked_name = "murmur3_uniform64"
    with pytest.raises(RuntimeError, match="write rejected"):
        sampling.install()
    assert all(binding is sampling.upstream for binding in _bindings(m))

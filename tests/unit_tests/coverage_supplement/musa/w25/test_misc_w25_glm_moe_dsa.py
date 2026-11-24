# SPDX-License-Identifier: Apache-2.0
"""Coverage for the Ascend GLM-5/5.2 DSA wrapper in ``vllm_fl.models``.

The real ``vllm_fl/models/glm_moe_dsa.py`` source is executed; the upstream
vLLM model/config classes and the Ascend dense-fallback helpers are supplied as
explicit stubs so the wrapper's own branching can be asserted in isolation.
No weights or accelerators are involved.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

SOURCE_REL = "vllm_fl/models/glm_moe_dsa.py"


def _source_path() -> Path:
    candidates = []
    override = os.environ.get("VLLM_FL_SOURCE_ROOT")
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [Path("/tmp/flsrc"), Path.cwd(), Path(__file__).resolve().parents[2]]
    )
    for root in candidates:
        target = root / SOURCE_REL
        if target.is_file():
            return target
    raise FileNotFoundError("could not locate " + SOURCE_REL)


def _leaf(monkeypatch, name, **attrs):
    # A fully qualified name in ``sys.modules`` satisfies ``from ... import``
    # without importing (or touching) the real parent packages.
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load(monkeypatch, base_cls, prepare, use_cm):
    _leaf(monkeypatch, "vllm.config", VllmConfig=object)
    _leaf(
        monkeypatch,
        "vllm.model_executor.models.deepseek_v2",
        GlmMoeDsaForCausalLM=base_cls,
    )
    _leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.ascend_glm_dsa",
        prepare_glm_dsa_dense_fallback=prepare,
        use_glm_dsa_dense_hf_config=use_cm,
    )

    spec = importlib.util.spec_from_file_location("_w25_glm_moe_dsa", _source_path())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_glm_init_prepares_fallback_inside_dense_config_scope(monkeypatch):
    events = []

    class Base:
        def __init__(self, *, vllm_config, prefix=""):
            events.append(("base_init", vllm_config, prefix, _in_scope["value"]))

    _in_scope = {"value": False}
    model_config = object()
    vllm_config = types.SimpleNamespace(model_config=model_config)

    def prepare(config):
        events.append(("prepare", config))
        return True

    @contextlib.contextmanager
    def use_cm(config):
        events.append(("enter", config))
        _in_scope["value"] = True
        try:
            yield "dense"
        finally:
            _in_scope["value"] = False
            events.append(("exit", config))

    module = _load(monkeypatch, Base, prepare, use_cm)
    model = module.GlmMoeDsaForCausalLM(vllm_config=vllm_config, prefix="block")

    assert isinstance(model, Base)
    assert events[0] == ("prepare", model_config)
    assert events[1] == ("enter", model_config)
    assert events[2] == ("base_init", vllm_config, "block", True)
    assert events[3] == ("exit", model_config)
    assert _in_scope["value"] is False


def test_glm_init_restores_scope_when_base_raises(monkeypatch):
    events = []

    class Base:
        def __init__(self, *, vllm_config, prefix=""):
            raise RuntimeError("base failed")

    def prepare(config):
        return True

    @contextlib.contextmanager
    def use_cm(config):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    module = _load(monkeypatch, Base, prepare, use_cm)
    with pytest.raises(RuntimeError, match="base failed"):
        module.GlmMoeDsaForCausalLM(
            vllm_config=types.SimpleNamespace(model_config=object())
        )
    assert events == ["enter", "exit"]


def test_glm_init_defaults_prefix_and_uses_keyword_call(monkeypatch):
    captured = {}

    class Base:
        def __init__(self, *, vllm_config, prefix=""):
            captured["vllm_config"] = vllm_config
            captured["prefix"] = prefix

    def prepare(config):
        return False

    @contextlib.contextmanager
    def use_cm(config):
        yield None

    module = _load(monkeypatch, Base, prepare, use_cm)
    sentinel = object()
    module.GlmMoeDsaForCausalLM(
        vllm_config=types.SimpleNamespace(model_config=sentinel)
    )
    assert captured["prefix"] == ""
    assert captured["vllm_config"].model_config is sentinel


def test_glm_load_weights_drops_top_level_rot(monkeypatch):
    class Base:
        def load_weights(self, weights):
            self.received = list(weights)
            return {"ok"}

    def prepare(config):
        raise AssertionError("load_weights must not touch model config")

    def use_cm(config):  # pragma: no cover - never called here
        raise AssertionError("load_weights must not touch the HF config")

    module = _load(monkeypatch, Base, prepare, use_cm)
    model = object.__new__(module.GlmMoeDsaForCausalLM)

    import torch

    weights = [
        ("rot.weight", torch.zeros(1)),
        ("model.layers.0.self_attn.q_proj.weight", torch.ones(1)),
        ("lm_head.weight", torch.ones(1)),
    ]
    result = model.load_weights(iter(weights))

    assert result == {"ok"}
    assert [name for name, _ in model.received] == [
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ]
    assert all(tuple(tensor.shape) == (1,) for _, tensor in model.received)

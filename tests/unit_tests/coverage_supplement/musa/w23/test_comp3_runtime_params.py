# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual branches for compilation graph_params / graph_runtime.

Loaded from the real source file (importlib) with only the accelerator
dependencies stubbed, so no capture or device is touched.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _repo_root() -> Path:
    import os

    candidates = [Path("/tmp/flsrc")]
    if os.environ.get("VLLM_FL_ROOT"):
        candidates.insert(0, Path(os.environ["VLLM_FL_ROOT"]))
    for base in [*candidates, Path.cwd(), *Path(__file__).resolve().parents]:
        if not base.is_dir():
            continue
        if (base / "vllm_fl/compilation/graph_params.py").is_file():
            return base
    raise RuntimeError("cannot locate vllm_fl source tree")


ROOT = _repo_root()


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclasses resolve annotations via
    # ``sys.modules[cls.__module__].__dict__``; register before executing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BatchDescriptor:
    """Hashable stand-in usable as a GraphParams dict key."""

    def __init__(self, num_tokens):
        self.num_tokens = num_tokens


@pytest.fixture
def graph_params_module():
    return _load("w23_graph_params", "vllm_fl/compilation/graph_params.py")


# --------------------------------------------------------------------------- #
# graph_params
# --------------------------------------------------------------------------- #
def test_prepare_capture_rejects_nonempty_handles(graph_params_module):
    params = graph_params_module.GraphParams()
    descriptor = BatchDescriptor(4)
    params.prepare_capture(descriptor)
    params.handles[descriptor].append(object())
    with pytest.raises(RuntimeError, match="already contain captured state"):
        params.prepare_capture(descriptor)


def test_prepare_capture_rejects_nonempty_events(graph_params_module):
    params = graph_params_module.GraphParams()
    descriptor = BatchDescriptor(4)
    params.prepare_capture(descriptor)
    params.events[descriptor].append(object())
    with pytest.raises(RuntimeError, match="already contain captured state"):
        params.prepare_capture(descriptor)


def test_weak_ref_workspace_requires_preparation(graph_params_module):
    graph_params_module.clear_graph_params()
    descriptor = BatchDescriptor(2)
    with pytest.raises(RuntimeError, match="not prepared"):
        graph_params_module.weak_ref_workspace(descriptor, lambda value: value)


# --------------------------------------------------------------------------- #
# graph_runtime
# --------------------------------------------------------------------------- #
def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_runtime(monkeypatch):
    platform = SimpleNamespace(
        torch_device_fn=SimpleNamespace(
            current_stream=MagicMock(),
            Stream=MagicMock(return_value="stream"),
        )
    )
    vllm = _module("vllm")
    vllm_platforms = _module("vllm.platforms", current_platform=platform)
    vllm.platforms = vllm_platforms
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", vllm_platforms)
    return _load("w23_graph_runtime", "vllm_fl/compilation/graph_runtime.py")


def test_bind_attention_groups_skips_plain_backend_and_duplicate_impls(monkeypatch):
    module = _load_runtime(monkeypatch)

    class AttentionBackend:
        @staticmethod
        def get_impl_cls():
            return "base"

    backend_module = _module(
        "vllm.v1.attention.backend", AttentionBackend=AttentionBackend
    )
    for name in ("vllm.v1", "vllm.v1.attention"):
        monkeypatch.setitem(sys.modules, name, _module(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backend", backend_module)

    impl = SimpleNamespace(update_graph_params=MagicMock())
    WithFactory = type(
        "WithFactory", (), {"get_impl_cls": staticmethod(lambda: impl)}
    )
    # A second backend whose factory returns the same impl -> deduplicated.
    OtherFactory = type(
        "OtherFactory", (), {"get_impl_cls": staticmethod(lambda: impl)}
    )
    # A backend that never overrides get_impl_cls -> factory is None -> skip.
    NoFactory = type("NoFactory", (), {})

    controller = module.GraphRuntimeController(vllm_config=object())
    groups = [
        [
            SimpleNamespace(backend=NoFactory),
            SimpleNamespace(backend=WithFactory),
        ],
        # Repeated backend is skipped by the seen_backends set.
        [SimpleNamespace(backend=WithFactory)],
        # Different backend, same impl instance -> not appended twice.
        [SimpleNamespace(backend=OtherFactory)],
    ]
    controller.bind_attention_groups(groups)

    assert controller.attention_impls == (impl,)

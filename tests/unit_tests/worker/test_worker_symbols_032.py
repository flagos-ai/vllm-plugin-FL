# Copyright (c) 2026 BAAI. All rights reserved.

"""The shared worker compatibility shim must preserve installed APIs."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("native_helpers", [False, True])
def test_worker_helpers_are_missing_only_and_do_not_import_v2_attention(
    monkeypatch, native_helpers
):
    names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.rotary_embedding",
        "vllm.model_executor.models",
        "vllm.model_executor.models.interfaces",
        "vllm.v1",
        "vllm.v1.outputs",
        "vllm.v1.worker",
        "vllm.v1.worker.utils",
    ]
    modules = {}
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)

    targets = [
        ("vllm.v1.worker.utils", "is_residual_scattered_for_sp"),
        ("vllm.model_executor.layers.rotary_embedding", "XDRotaryEmbedding"),
        ("vllm.model_executor.models.interfaces", "SupportsXDRoPE"),
        ("vllm.model_executor.models.interfaces", "supports_xdrope"),
        ("vllm.v1.outputs", "RoutedExpertsLists"),
        ("vllm.v1.outputs", "RoutedExpertsTensors"),
    ]
    sentinels = {target: object() for target in targets}
    if native_helpers:
        for (module_name, name), value in sentinels.items():
            setattr(modules[module_name], name, value)

    path = Path(__file__).resolve().parents[3] / "vllm_fl/patches/worker_symbols_032.py"
    spec = importlib.util.spec_from_file_location("_fl_worker_symbols_test", path)
    shim = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, shim)
    spec.loader.exec_module(shim)
    # No vllm.v1.worker.gpu package is installed in this isolated runtime.
    # Importing its attn_utils for the removed allocator shim would fail here.
    assert shim.apply_worker_symbols_032()
    installed = {}
    for module_name, name in targets:
        value = getattr(modules[module_name], name)
        installed[module_name, name] = value
        if native_helpers:
            assert value is sentinels[module_name, name]
        else:
            assert callable(value)
    assert shim.apply_worker_symbols_032()
    for (module_name, name), value in installed.items():
        assert getattr(modules[module_name], name) is value

    # Default 0.32 has no enable_sp field, so the fallback remains disabled.
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(pass_config=SimpleNamespace())
    )
    assert not shim._is_residual_scattered_for_sp(config, 1)
    tensors = shim._RoutedExpertsTensors(torch.ones(2, 3), torch.arange(2))
    assert tensors.to_cpu_nonblocking() is tensors
    assert tensors.tolists().routing_data.shape == (2, 3)

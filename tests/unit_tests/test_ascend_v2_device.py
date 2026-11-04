# Copyright (c) 2026 BAAI. All rights reserved.

"""V2 NPU device shims stay local and preserve ordinary PyTorch APIs."""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "vllm_fl"


@pytest.fixture
def shim(monkeypatch):
    path = ROOT / "patches/ascend_v2_device.py"
    spec = importlib.util.spec_from_file_location("_test_ascend_v2_device", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_facade_routes_streams_events_and_graphs_without_mutating_torch(shim):
    cuda = torch.cuda
    stream, event, graph = object(), object(), type("NPUGraph", (), {})
    device = SimpleNamespace(
        Stream=Mock(return_value=stream),
        Event=Mock(return_value=event),
        current_stream=Mock(return_value=stream),
        set_stream=Mock(),
        NPUGraph=graph,
    )
    facade = shim._TorchFacade(device)
    assert facade.cuda.Stream("npu:0") is stream
    assert facade.cuda.Event(blocking=True) is event
    assert facade.cuda.current_stream() is stream
    facade.cuda.set_stream(stream)
    device.set_stream.assert_called_once_with(stream)
    assert facade.cuda.CUDAGraph is graph
    assert facade.Tensor is torch.Tensor
    assert torch.equal(facade.ones(3), torch.ones(3))
    assert torch.cuda is cuda
    assert sys.modules["torch"] is torch


@pytest.mark.parametrize("device_type", ["npu", "cuda"])
def test_installer_selects_non_uva_locally_and_preserves_pinned_memory(
    monkeypatch, shim, device_type
):
    platform = SimpleNamespace(
        device_type=device_type,
        torch_device_fn=SimpleNamespace(NPUGraph=object()),
        is_pin_memory_available=lambda: True,
    )
    root = ModuleType("vllm")
    root.__path__ = []
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = platform
    root.platforms = platforms
    monkeypatch.setitem(sys.modules, "vllm", root)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    targets = {}
    for name in shim._DEVICE_MODULES:
        target = ModuleType(name)
        target.torch = torch
        targets[name] = target
        monkeypatch.setitem(sys.modules, name, target)
    buffers = ModuleType("vllm.v1.worker.gpu.buffer_utils")
    uva_available = lambda: True
    buffers.is_uva_available = uva_available
    monkeypatch.setitem(sys.modules, buffers.__name__, buffers)

    assert shim.install_ascend_v2_device_shims() is (device_type == "npu")
    assert platform.is_pin_memory_available()
    if device_type == "npu":
        assert not buffers.is_uva_available()
        assert all(
            target.torch.cuda._device_api is platform.torch_device_fn
            for target in targets.values()
        )
        # Re-installation covers modules imported later without changing the
        # process-wide torch module or restoring CUDA UVA assumptions.
        assert shim.install_ascend_v2_device_shims()
        assert not buffers.is_uva_available()
    else:
        assert buffers.is_uva_available is uva_available
        assert all(target.torch is torch for target in targets.values())
    assert sys.modules["torch"] is torch


@pytest.mark.parametrize("requested,expected", [(None, "1"), ("0", "0"), ("1", "1")])
def test_ascend_defaults_to_v2_and_retains_explicit_v1(
    monkeypatch, requested, expected
):
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    register = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "register"
    )
    utils = ModuleType("vllm_fl.utils")
    utils.DeviceInfo = lambda: SimpleNamespace(vendor_name="ascend")
    monkeypatch.setitem(sys.modules, "vllm_fl.utils", utils)
    if requested is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", requested)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    namespace = {"os": os, "logger": Mock()}
    for name in (
        "_init_vendor_device",
        "_arm_cpu_platform",
        "_patch_custom_ops",
        "_patch_flash_attn_import",
        "_get_op_config",
    ):
        namespace[name] = lambda: None
    exec(
        compile(ast.Module(body=[register], type_ignores=[]), "register", "exec"),
        namespace,
    )
    assert namespace["register"]() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == expected


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("backend_fails", [False, True])
def test_kv_layout_rpc_binds_model_config_and_restores_context(hybrid, backend_fails):
    from contextlib import contextmanager

    source = ROOT / "worker/worker.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    )
    method = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_supported_kv_cache_layouts"
    )
    previous = object()
    active = [previous]
    config = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=hybrid))

    @contextmanager
    def set_config(value):
        prior = active[0]
        active[0] = value
        try:
            yield
        finally:
            active[0] = prior

    class Base:
        def get_supported_kv_cache_layouts(self):
            assert active[0] is config
            if backend_fails:
                raise RuntimeError("backend failed")
            return ["LBNHC" if active[0].model_config.is_hybrid else "LHBNC"]

    namespace = {"Base": Base, "set_current_vllm_config": set_config}
    derived = ast.ClassDef(
        name="Worker",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    worker = namespace["Worker"]()
    worker.vllm_config = config
    if backend_fails:
        with pytest.raises(RuntimeError, match="backend failed"):
            worker.get_supported_kv_cache_layouts()
    else:
        assert worker.get_supported_kv_cache_layouts() == [
            "LBNHC" if hybrid else "LHBNC"
        ]
    assert active[0] is previous

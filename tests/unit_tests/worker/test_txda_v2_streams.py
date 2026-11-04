# Copyright (c) 2026 BAAI. All rights reserved.
"""Native stream routing without changing torch.cuda or the V2 runner class."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def adapter(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3] / "vllm_fl/worker/txda_model_runner_v2.py"
    )
    spec = importlib.util.spec_from_file_location("txda_v2_streams_test", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    torch = ModuleType("torch")
    torch.cuda = object()
    torch.Tensor = type("Tensor", (), {})
    torch.txda = SimpleNamespace(
        **{name: Mock(name=name) for name in helper._STREAM_APIS}
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    platform = SimpleNamespace(device_type="txda")
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = platform
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    zeroer = ModuleType("vllm_fl.worker.txda_kv_zeroer")
    zeroer.patch_txda_v2_kv_zeroer = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, zeroer.__name__, zeroer)
    preprocess = ModuleType("vllm_fl.worker.txda_v2_preprocess")
    preprocess.patch_txda_v2_preprocess = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, preprocess.__name__, preprocess)
    modules = []
    runner_class = type("GPUModelRunner", (), {})
    for name in helper._DEVICE_MODULES:
        module = ModuleType(name)
        module.torch = torch
        module.GPUModelRunner = runner_class
        monkeypatch.setitem(sys.modules, name, module)
        modules.append(module)
    return SimpleNamespace(
        helper=helper,
        torch=torch,
        platform=platform,
        modules=modules,
        cls=runner_class,
        preprocess=preprocess.patch_txda_v2_preprocess,
        zeroer=zeroer.patch_txda_v2_kv_zeroer,
    )


@pytest.mark.parametrize(
    "api", ["Stream", "Event", "current_stream", "set_stream", "stream"]
)
def test_v2_modules_route_only_stream_calls_to_native_txda(adapter, api):
    original_cuda = adapter.torch.cuda
    assert adapter.helper.get_txda_v2_model_runner() is adapter.cls
    for module in adapter.modules:
        function = getattr(module.torch.cuda, api)
        assert function is getattr(adapter.torch.txda, api)
        assert function("device-or-stream", blocking=True) is function.return_value
        function.assert_called_with("device-or-stream", blocking=True)
        assert module.torch.Tensor is adapter.torch.Tensor
    assert adapter.torch.cuda is original_cuda
    adapter.preprocess.assert_called_once_with()
    adapter.zeroer.assert_called_once_with(adapter.modules[0])


def test_repeated_install_preserves_module_views_and_upstream_class(adapter):
    adapter.helper.get_txda_v2_model_runner()
    views = [module.torch for module in adapter.modules]
    assert adapter.helper.get_txda_v2_model_runner() is adapter.cls
    assert all(module.torch is view for module, view in zip(adapter.modules, views))


@pytest.mark.parametrize("device", ["cuda", "npu", "musa", "cpu"])
def test_other_platforms_cannot_install_txda_stream_views(adapter, device):
    adapter.platform.device_type = device
    with pytest.raises(RuntimeError, match="requires the TXDA platform"):
        adapter.helper.get_txda_v2_model_runner()
    assert all(module.torch is adapter.torch for module in adapter.modules)


def test_missing_native_api_fails_before_modifying_v2_modules(adapter):
    del adapter.torch.txda.Event
    with pytest.raises(RuntimeError, match="missing the V2 stream API Event"):
        adapter.helper.get_txda_v2_model_runner()
    assert all(module.torch is adapter.torch for module in adapter.modules)


def test_unhandled_cuda_api_fails_instead_of_calling_cpu_cuda_stub(adapter):
    adapter.helper.get_txda_v2_model_runner()
    with pytest.raises(RuntimeError, match="does not provide CUDA API CUDAGraph"):
        adapter.modules[0].torch.cuda.CUDAGraph()

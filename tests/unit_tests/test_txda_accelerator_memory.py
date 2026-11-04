# Copyright (c) 2026 BAAI. All rights reserved.
"""Device-wide memory reporting without loading the TXDA SDK."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from torch.accelerator._utils import _get_device_index

ROOT = Path(__file__).resolve().parents[2]


class _Device:
    def __init__(self, name):
        self.type, _, index = name.partition(":")
        self.index = int(index) if index else None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


@pytest.fixture
def runtime(monkeypatch):
    torch = ModuleType("torch")
    torch.device = _Device
    accelerator = ModuleType("torch.accelerator")
    accelerator.current_accelerator = Mock(return_value=_Device("txda"))
    accelerator.current_device_index = Mock(return_value=3)
    pool = Mock(return_value=(2079184, 2097152))

    def original(device_index=None, /):
        return pool(device_index)

    accelerator.get_memory_info = original
    accelerator.memory_stats = Mock(
        return_value={"reserved_bytes.all.current": 2097152}
    )
    txda = SimpleNamespace(
        is_available=Mock(return_value=True),
        mem_get_info=Mock(return_value=(60 * 1024**3, 64 * 1024**3)),
    )
    torch.accelerator = accelerator
    torch.txda = txda
    utils = ModuleType("torch.accelerator._utils")
    # Execute the installed PyTorch normalization function with a stub device
    # class so tests can exercise TXDA devices on a CPU-only PyTorch build.
    utils._get_device_index = FunctionType(
        _get_device_index.__code__,
        {"torch": torch},
        argdefs=_get_device_index.__defaults__,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.accelerator", accelerator)
    monkeypatch.setitem(sys.modules, "torch.accelerator._utils", utils)
    monkeypatch.setitem(sys.modules, "torch_txda", ModuleType("torch_txda"))
    path = ROOT / "vllm_fl/txda_compat.py"
    spec = importlib.util.spec_from_file_location("txda_memory_compat_test", path)
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    return SimpleNamespace(
        torch=torch,
        accelerator=accelerator,
        txda=txda,
        original=original,
        pool=pool,
        patch=compat.patch_accelerator_memory_info,
    )


@pytest.mark.parametrize("pool_bytes", [(0, 0), (2079184, 2097152)])
def test_reports_physical_free_and_total_instead_of_allocator_pool(runtime, pool_bytes):
    runtime.pool.return_value = pool_bytes
    stats = runtime.accelerator.memory_stats
    assert runtime.patch()

    assert runtime.accelerator.get_memory_info() == (60 * 1024**3, 64 * 1024**3)
    runtime.txda.mem_get_info.assert_called_once_with(3)
    runtime.pool.assert_not_called()
    assert runtime.accelerator.memory_stats is stats


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        (None, 3),
        (0, 0),
        (2, 2),
        ("txda:1", 1),
        ("txda", 3),
        (_Device("txda:2"), 2),
        (_Device("txda"), 3),
    ],
)
def test_normalizes_none_integer_string_and_device_arguments(runtime, device, expected):
    assert runtime.patch()
    result = runtime.accelerator.get_memory_info(device)
    assert result is runtime.txda.mem_get_info.return_value
    runtime.txda.mem_get_info.assert_called_once_with(expected)


def test_default_device_follows_current_device_at_call_time(runtime):
    assert runtime.patch()
    runtime.accelerator.current_device_index.return_value = 5
    runtime.accelerator.get_memory_info()
    runtime.txda.mem_get_info.assert_called_once_with(5)


@pytest.mark.parametrize("device", ["cuda:0", _Device("cpu"), _Device("xpu:1")])
def test_explicit_other_device_retains_pytorch_validation(runtime, device):
    assert runtime.patch()
    with pytest.raises(ValueError, match="doesn't match the current accelerator"):
        runtime.accelerator.get_memory_info(device)
    runtime.txda.mem_get_info.assert_not_called()
    runtime.pool.assert_not_called()


def test_retains_positional_only_signature(runtime):
    assert runtime.patch()
    with pytest.raises(TypeError, match="positional-only"):
        runtime.accelerator.get_memory_info(device_index=0)
    runtime.txda.mem_get_info.assert_not_called()


def test_repeated_patch_preserves_original_wrapper(runtime):
    assert runtime.patch()
    installed = runtime.accelerator.get_memory_info
    assert runtime.patch()
    assert runtime.accelerator.get_memory_info is installed
    assert installed.__wrapped__ is runtime.original
    installed(1)
    runtime.txda.mem_get_info.assert_called_once_with(1)


@pytest.mark.parametrize("backend", [None, "cuda", "musa", "xpu"])
def test_other_active_backends_are_unchanged(runtime, backend):
    runtime.accelerator.current_accelerator.return_value = (
        _Device(backend) if backend else None
    )
    assert not runtime.patch()
    assert runtime.accelerator.get_memory_info is runtime.original
    runtime.txda.mem_get_info.assert_not_called()


def test_calls_original_if_active_backend_changes_after_patch(runtime):
    assert runtime.patch()
    runtime.accelerator.current_accelerator.return_value = _Device("cuda")
    device = _Device("cuda:1")
    assert runtime.accelerator.get_memory_info(device) == runtime.pool.return_value
    runtime.pool.assert_called_once_with(device)
    runtime.txda.mem_get_info.assert_not_called()


@pytest.mark.parametrize("missing_sdk", [False, True])
def test_unavailable_txda_does_not_replace_memory_info(
    runtime, monkeypatch, missing_sdk
):
    if missing_sdk:
        monkeypatch.setitem(sys.modules, "torch_txda", None)
    else:
        runtime.txda.is_available.return_value = False
    assert not runtime.patch()
    assert runtime.accelerator.get_memory_info is runtime.original


@pytest.mark.parametrize("vendor", ["tsingmicro", "nvidia"])
def test_platform_entry_point_applies_only_txda_helper(monkeypatch, vendor):
    # Load the real early entry point without importing the plugin package,
    # which would trigger unrelated accelerator and dispatcher discovery.
    helper = ModuleType("vllm_fl.txda_compat")
    helper.patch_accelerator_memory_info = Mock()
    utils = ModuleType("vllm_fl.utils")
    utils.DeviceInfo = lambda: SimpleNamespace(vendor_name=vendor)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    path = ROOT / "vllm_fl/__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_vendor_device"
    )
    namespace = {"_install_metax_compat": Mock()}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace["_init_vendor_device"]()
    assert helper.patch_accelerator_memory_info.call_count == (vendor == "tsingmicro")

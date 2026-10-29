# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for actual test-device detection and hardware marker gating.

Device/runtime doubles avoid importing or initializing any accelerator. The
production utility module and the unchanged collection hook execute directly.
"""

import ast
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
NS = SimpleNamespace


@pytest.fixture
def load_utils():
    def load(
        *,
        backend="",
        cuda=False,
        npu=False,
        musa=False,
        count=4,
        npu_installed=False,
        musa_installed=True,
        import_error=None,
    ):
        cuda_api = NS(
            is_available=Mock(return_value=cuda), device_count=Mock(return_value=2)
        )
        npu_api = NS(
            is_available=Mock(return_value=npu), device_count=Mock(return_value=3)
        )
        musa_api = NS(
            is_available=Mock(return_value=musa), device_count=Mock(return_value=count)
        )
        imports = []
        torch = NS(cuda=cuda_api)

        class Device:
            def __init__(self, name):
                # torch_musa registers the PrivateUse1 name before torch.device
                # can accept it, including an explicit FL_BACKEND=musa.
                if name.startswith("musa"):
                    assert hasattr(torch, "musa")
                parts = name.split(":")
                self.type = parts[0]
                self.index = int(parts[1]) if len(parts) > 1 else None

        torch.device = Device
        environment = {"FL_BACKEND": backend} if backend else {}

        def import_module(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                return torch
            if name == "os":
                return NS(environ=environment)
            if name in ("torch_npu", "torch_musa"):
                imports.append(name)
                installed = npu_installed if name == "torch_npu" else musa_installed
                if not installed:
                    raise ImportError(name + " unavailable")
                if import_error is not None and name == "torch_musa":
                    raise import_error
                setattr(
                    torch,
                    "npu" if name == "torch_npu" else "musa",
                    npu_api if name == "torch_npu" else musa_api,
                )
                return NS()
            return builtins.__import__(name, globals, locals, fromlist, level)

        namespace = {"__builtins__": {**vars(builtins), "__import__": import_module}}
        path = ROOT / "tests/utils/device_utils.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        namespace["find_spec"] = Mock(return_value=None)
        return NS(**namespace), NS(
            torch=torch,
            imports=imports,
            cuda=cuda_api,
            npu=npu_api,
            musa=musa_api,
            environment=environment,
        )

    return load


def test_musa_autodetection_uses_native_device_and_count(load_utils):
    utils, state = load_utils(musa=True)
    assert utils.get_backend() == "musa"
    assert utils.is_accelerator_available()
    device = utils.get_device(2)
    assert isinstance(device, state.torch.device)
    assert (device.type, device.index) == ("musa", 2)
    assert utils.get_device_count() == 4
    state.musa.device_count.assert_called_once()
    state.cuda.device_count.assert_not_called()


def test_explicit_musa_overrides_other_hardware_and_registers_device(load_utils):
    utils, state = load_utils(backend="MuSa", cuda=True, npu=True, npu_installed=True)
    assert utils.get_backend() == "musa"
    device = utils.get_device(1)
    assert (device.type, device.index) == ("musa", 1)
    assert utils.get_device_count() == 4
    assert state.imports == ["torch_musa", "torch_musa"]
    state.cuda.is_available.assert_not_called()
    state.npu.is_available.assert_not_called()
    state.musa.is_available.assert_not_called()


@pytest.mark.parametrize(
    "cuda,npu,npu_installed,expected,expected_count",
    [(True, True, True, "nvidia", 2), (False, True, True, "ascend", 3)],
)
def test_existing_hardware_priority_precedes_musa(
    load_utils, cuda, npu, npu_installed, expected, expected_count
):
    utils, state = load_utils(
        cuda=cuda, npu=npu, npu_installed=npu_installed, musa=True
    )
    assert utils.get_backend() == expected
    assert utils.get_device_count() == expected_count
    assert "torch_musa" not in state.imports
    device = utils.get_device(1)
    assert (device.type, device.index) == ("cuda" if cuda else "npu", 1)


@pytest.mark.parametrize(
    "npu_installed,musa_installed",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_no_available_accelerator_keeps_cpu_fallback(
    load_utils, npu_installed, musa_installed
):
    utils, _ = load_utils(npu_installed=npu_installed, musa_installed=musa_installed)
    assert utils.get_backend() == "cpu"
    assert not utils.is_accelerator_available()
    assert utils.get_device().type == "cpu"
    assert utils.get_device_count() == 0


@pytest.mark.parametrize(
    "backend,expected_device,expected_count",
    [
        ("nvidia", "cuda", 2),
        ("tianshu", "cuda", 2),
        ("ascend", "npu", 3),
        ("cpu", "cpu", 0),
        ("custom", "cpu", 0),
    ],
)
def test_existing_explicit_backends_preserve_selection(
    load_utils, backend, expected_device, expected_count
):
    utils, state = load_utils(
        backend=backend, cuda=True, npu=True, npu_installed=True, musa=True
    )
    assert utils.get_backend() == backend
    assert utils.get_device().type == expected_device
    assert utils.get_device_count() == expected_count
    assert "torch_musa" not in state.imports
    state.cuda.is_available.assert_not_called()


def test_missing_musa_module_count_is_zero_and_device_does_not_fall_back(load_utils):
    utils, _ = load_utils(backend="musa", musa_installed=False)
    assert utils.get_device_count() == 0
    with pytest.raises(ImportError, match="torch_musa unavailable"):
        utils.get_device()


def test_musa_initialization_errors_are_not_hidden_as_cpu(load_utils):
    error = RuntimeError("native initialization sentinel")
    utils, _ = load_utils(import_error=error)
    with pytest.raises(RuntimeError) as caught:
        utils.get_backend()
    assert caught.value is error


@pytest.mark.parametrize(
    "available,count,expected_skips",
    [(True, 4, [0, 0, 0]), (True, 1, [0, 1, 0]), (False, 0, [1, 2, 0])],
)
def test_unchanged_collection_hook_gates_actual_musa_count(
    load_utils, available, count, expected_skips
):
    utils, _ = load_utils(musa=available, count=count)
    path = ROOT / "tests/conftest.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hook = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "pytest_collection_modifyitems"
    )
    namespace = {
        "pytest": pytest,
        "is_accelerator_available": utils.is_accelerator_available,
        "get_device_count": utils.get_device_count,
    }
    exec(
        compile(ast.Module(body=[hook], type_ignores=[]), str(path), "exec"), namespace
    )
    items = [
        NS(keywords=keywords, add_marker=Mock())
        for keywords in [{"gpu"}, {"gpu", "multi_gpu"}, set()]
    ]
    namespace[hook.name](None, items)
    assert [item.add_marker.call_count for item in items] == expected_skips


def test_backend_environment_is_still_read_at_import(load_utils):
    utils, state = load_utils(backend="nvidia", musa=True)
    state.environment["FL_BACKEND"] = "musa"
    assert utils.get_backend() == "nvidia"


@pytest.fixture
def device_utils(monkeypatch):
    def load(
        *, explicit="", vendor="", installed=False, gcu=False, cuda=False, npu=False
    ):
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(
            is_available=Mock(return_value=cuda),
            device_count=Mock(return_value=2),
        )
        torch.gcu = SimpleNamespace(
            is_available=Mock(return_value=gcu),
            device_count=Mock(return_value=8),
        )
        torch.npu = SimpleNamespace(is_available=Mock(return_value=npu))
        torch.device = Mock(side_effect=lambda name: name)
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torch_musa", None)
        monkeypatch.setitem(
            sys.modules, "torch_gcu", ModuleType("torch_gcu") if installed else None
        )
        monkeypatch.setitem(
            sys.modules, "torch_npu", ModuleType("torch_npu") if npu else None
        )
        monkeypatch.setenv("FL_BACKEND", explicit)
        monkeypatch.setenv("GEMS_VENDOR", vendor)
        path = Path(__file__).parents[1] / "utils" / "device_utils.py"
        spec = importlib.util.spec_from_file_location("_device_utils_regression", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(
            module, "find_spec", Mock(return_value=spec if installed else None)
        )
        return module, torch

    return load


@pytest.mark.parametrize(
    "installed,gcu,cuda,npu,expected",
    [
        (True, True, True, False, "enflame"),
        (True, True, False, False, "enflame"),
        (True, False, True, False, "nvidia"),
        (False, False, True, False, "nvidia"),
        (True, False, False, True, "ascend"),
        (False, False, False, True, "ascend"),
        (True, False, False, False, "cpu"),
        (False, False, False, False, "cpu"),
    ],
)
def test_auto_backend_uses_native_gcu_before_cuda_alias(
    device_utils, installed, gcu, cuda, npu, expected
):
    utils, torch = device_utils(installed=installed, gcu=gcu, cuda=cuda, npu=npu)
    assert utils.get_backend() == expected
    if gcu:
        torch.cuda.is_available.assert_not_called()


@pytest.mark.parametrize("explicit", ["nvidia", "tianshu", "ascend", "cpu"])
def test_explicit_non_gcu_backend_does_not_probe_gcu(device_utils, explicit):
    utils, torch = device_utils(explicit=explicit, installed=True, gcu=True, cuda=True)
    assert utils.get_backend() == explicit
    utils.find_spec.assert_not_called()
    torch.gcu.is_available.assert_not_called()
    torch.cuda.is_available.assert_not_called()


@pytest.mark.parametrize("explicit", ["enflame", "gcu"])
@pytest.mark.parametrize(
    "installed,available", [(False, False), (True, False), (True, True)]
)
def test_explicit_gcu_requires_native_availability(
    device_utils, explicit, installed, available
):
    utils, torch = device_utils(
        explicit=explicit, installed=installed, gcu=available, cuda=True
    )
    assert utils.get_backend() == ("enflame" if available else "cpu")
    torch.cuda.is_available.assert_not_called()


@pytest.mark.parametrize("explicit", ["", "enflame"])
@pytest.mark.parametrize("failure", ["import", "availability"])
def test_installed_gcu_errors_are_not_hidden_by_cuda_alias(
    device_utils, monkeypatch, explicit, failure
):
    utils, torch = device_utils(explicit=explicit, installed=True, gcu=True, cuda=True)
    if failure == "import":
        monkeypatch.setitem(sys.modules, "torch_gcu", None)
        error = ModuleNotFoundError
    else:
        torch.gcu.is_available.side_effect = RuntimeError("GCU driver failure")
        error = RuntimeError
    with pytest.raises(error):
        utils.get_backend()
    torch.cuda.is_available.assert_not_called()


def test_gcu_device_and_count_do_not_use_cuda_alias(device_utils):
    utils, torch = device_utils(installed=True, gcu=True, cuda=True)
    assert utils.get_device(3) == "gcu:3"
    assert utils.get_device_count() == 8
    assert utils.is_accelerator_available()
    torch.cuda.device_count.assert_not_called()


@pytest.mark.parametrize("gcu", [False, True])
def test_nvidia_only_selection_rejects_gcu_alias(device_utils, gcu):
    utils, _ = device_utils(installed=gcu, gcu=gcu, cuda=True)

    @utils.skip_if_not_nvidia
    def nvidia_only():
        pass

    (mark,) = nvidia_only.pytestmark
    assert mark.name == "skipif"
    assert mark.args == (gcu,)


@pytest.mark.parametrize("vendor", ["hygon", "HYGON"])
def test_hygon_vendor_keeps_cuda_api_as_a_distinct_backend(device_utils, vendor):
    utils, _ = device_utils(vendor=vendor, cuda=True)
    assert utils.get_backend() == "hygon"
    assert utils.is_accelerator_available()
    assert utils.get_device(3) == "cuda:3"
    assert utils.get_device_count() == 2

    @utils.skip_if_not_nvidia
    def nvidia_only():
        pass

    (mark,) = nvidia_only.pytestmark
    assert mark.name == "skipif"
    assert mark.args == (True,)


def test_hygon_vendor_does_not_override_native_gcu(device_utils):
    utils, torch = device_utils(vendor="hygon", installed=True, gcu=True, cuda=True)
    assert utils.get_backend() == "enflame"
    torch.cuda.is_available.assert_not_called()


def test_hygon_vendor_requires_available_hardware(device_utils):
    utils, _ = device_utils(vendor="hygon")
    assert utils.get_backend() == "cpu"
    assert not utils.is_accelerator_available()

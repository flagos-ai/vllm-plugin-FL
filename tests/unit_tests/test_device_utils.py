# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for actual test-device detection and hardware marker gating.

Device/runtime doubles avoid importing or initializing any accelerator. The
production utility module and the unchanged collection hook execute directly.
"""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


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

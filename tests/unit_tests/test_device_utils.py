# Copyright 2026 FlagOS Contributors
"""Exercise backend selection without importing accelerator runtimes."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


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

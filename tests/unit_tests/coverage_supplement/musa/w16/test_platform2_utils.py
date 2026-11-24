# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual CPU contracts for vllm_fl.utils (branches not covered by w4a)."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_fl.utils as utils


def _install(monkeypatch, full, **attrs):
    parts = full.split(".")
    for i in range(1, len(parts)):
        pname = ".".join(parts[:i])
        if pname in sys.modules:
            continue
        pkg = ModuleType(pname)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, pname, pkg)
        if i > 1:
            parent = sys.modules[".".join(parts[: i - 1])]
            monkeypatch.setattr(parent, parts[i - 1], pkg, raising=False)
    module = ModuleType(full)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, full, module)
    parent_name, _, child = full.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child, module, raising=False)
    return module


def test_device_info_init_wires_backend(monkeypatch):
    detected = SimpleNamespace(
        vendor_name="mthreads", name="musa", dispatch_key="PrivateUse1"
    )
    backend = SimpleNamespace(set_torch_backend_device_fn=Mock())
    _install(
        monkeypatch,
        "flag_gems.runtime",
        backend=backend,
        device=detected,
    )
    info = utils.DeviceInfo()
    assert info.device is detected
    assert info._backend is backend
    assert "mthreads" in info.supported_device
    backend.set_torch_backend_device_fn.assert_called_once_with("mthreads")
    assert info.vendor_name == "mthreads"


def test_device_control_env_var_blank_vendor_uses_placeholder(monkeypatch):
    monkeypatch.delenv("FLAGOS_DEVICE_CONTROL_ENV_VAR", raising=False)
    assert utils.get_device_control_env_var("") == (
        utils._DEVICE_CONTROL_ENV_VAR_PLACEHOLDER
    )


@pytest.mark.parametrize("prefer", ["flagos", "FLAGOS", " FlagOS "])
def test_use_flaggems_accepts_flagos_case_insensitively(monkeypatch, prefer):
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "True")
    monkeypatch.setenv("VLLM_FL_PREFER", prefer)
    monkeypatch.delenv("USE_FLAGGEMS", raising=False)
    assert utils.use_flaggems(default=True) is True

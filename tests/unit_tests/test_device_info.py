# Copyright (c) 2025 BAAI. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.utils import DeviceInfo


@pytest.mark.parametrize("detector_module", ["device", "device_finder"])
@pytest.mark.parametrize(
    "vendor,device_name,dispatch_key",
    [
        ("ascend", "npu", "PrivateUse1"),
        ("hygon", "cuda", "CUDA"),
        ("nvidia", "cuda", "CUDA"),
    ],
)
def test_device_info_uses_public_runtime_device(
    monkeypatch, detector_module, vendor, device_name, dispatch_key
):
    """Both FlagGems layouts expose the same public runtime.device API."""
    for name in ("device", "device_finder"):
        monkeypatch.setitem(sys.modules, f"flag_gems.runtime.backend.{name}", None)

    gems = ModuleType("flag_gems")
    runtime = ModuleType("flag_gems.runtime")
    backend = ModuleType("flag_gems.runtime.backend")
    for package in (gems, runtime, backend):
        package.__path__ = []
    detected = SimpleNamespace(
        vendor_name=vendor, name=device_name, dispatch_key=dispatch_key
    )
    torch_device = object()
    torch_backend = object()
    backend.set_torch_backend_device_fn = Mock()
    backend.gen_torch_device_object = Mock(return_value=torch_device)
    backend.get_torch_backend_device_fn = Mock(return_value=torch_backend)
    runtime.device = detected
    runtime.backend = backend
    gems.runtime = runtime

    # Older releases put the detector in backend.device; newer ones moved it.
    private = ModuleType(f"flag_gems.runtime.backend.{detector_module}")
    private.DeviceDetector = lambda: detected
    setattr(backend, detector_module, private)
    for module in (gems, runtime, backend, private):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    info = DeviceInfo()

    assert info.device is detected
    assert info.vendor_name == vendor
    assert info.device_type == device_name
    assert info.dispatch_key == dispatch_key
    backend.set_torch_backend_device_fn.assert_called_once_with(vendor)
    assert info.torch_device_fn is torch_device
    assert info.torch_backend_device is torch_backend

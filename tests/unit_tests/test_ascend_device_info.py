import builtins
import sys
from types import SimpleNamespace

import torch

from vllm_fl.utils import DeviceInfo


def test_ascend_device_detection_does_not_require_flaggems(monkeypatch):
    fake_npu = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    real_import = builtins.__import__

    def reject_flaggems(name, *args, **kwargs):
        if name == "flag_gems" or name.startswith("flag_gems."):
            raise AssertionError("Ascend detection must not import FlagGems")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_flaggems)

    device = DeviceInfo()

    assert device.vendor_name == "ascend"
    assert device.device_type == "npu"
    assert device.dispatch_key == "PrivateUse1"
    assert device.torch_device_fn is fake_npu


def test_ascend_device_detection_bootstraps_torch_npu(monkeypatch):
    fake_npu = SimpleNamespace(is_available=lambda: True)
    monkeypatch.delattr(torch, "npu", raising=False)

    real_import = builtins.__import__

    def bootstrap_torch_npu(name, *args, **kwargs):
        if name == "torch_npu":
            monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
            return SimpleNamespace()
        if name == "flag_gems" or name.startswith("flag_gems."):
            raise AssertionError("Ascend detection must not import FlagGems")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    monkeypatch.setattr(builtins, "__import__", bootstrap_torch_npu)

    device = DeviceInfo()

    assert device.vendor_name == "ascend"
    assert device.device_type == "npu"
    assert device.torch_device_fn is fake_npu

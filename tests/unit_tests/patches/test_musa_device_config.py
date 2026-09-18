import pytest
import torch


def test_musa_device_config_is_index_free(monkeypatch):
    pytest.importorskip("torch_musa")

    import vllm.config.device as device_module

    from vllm_fl.dispatch.backends.vendor.musa.patch import (
        patch_device_config_for_musa,
    )

    class DeviceConfig:
        def __init__(self, device):
            self.device = device
            self.__post_init__()

        def __post_init__(self):
            self.device_type = self.device

    monkeypatch.setattr(device_module, "DeviceConfig", DeviceConfig)
    patch_device_config_for_musa()

    config = DeviceConfig("musa")
    assert config.device == torch.device("musa")
    assert config.device.index is None
    assert config.device_type == "musa"

    other = DeviceConfig("cpu")
    assert (other.device, other.device_type) == ("cpu", "cpu")

from flag_gems.runtime import backend
from flag_gems.runtime.backend.device import DeviceDetector


class DeviceInfo:
    def __init__(self):
        self.device = DeviceDetector()
        self.supported_device = ["nvidia", "ascend"]
        backend.set_torch_backend_device_fn(self.device.vendor_name)

    @property
    def dispatch_key(self):
        return self.device.dispatch_key

    @property
    def vendor_name(self):
        return self.device.vendor_name

    @property
    def device_type(self):
        return self.device.name

    @property
    def torch_device_fn(self):
        # torch_device_fn is like 'torch.cuda' object
        return backend.gen_torch_device_object()

    @property
    def torch_backend_device(self):
        # torch_backend_device is like 'torch.backend.cuda' object
        return backend.get_torch_backend_device_fn()

    def get_supported_device(self):
        if self.vendor_name in self.supported_device:
            raise NotImplementedError(f"{self.vendor_name} is not support now!")
        return True


if __name__ == "__main__":
    device = DeviceInfo()

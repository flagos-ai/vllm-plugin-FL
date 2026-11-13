# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Import shim making the distributed area collectable on a CPU-only host.

The dev venv on this Windows host has no ``triton``, so the installed
``flag_gems`` package (imported by ``vllm_fl.utils``, which
``vllm_fl.distributed`` pulls in) cannot import, and ``vllm.v1.utils``
imports ``uvloop``. The A100 CI image has every real package; there each
step below is probe-guarded and therefore a no-op:

* ``uvloop`` — only the event-loop installer is stubbed.
* ``flag_gems`` — the same inert placeholder the dispatch/worker areas use
  (``_FULL_CONFIG`` plus the ``runtime.backend`` attribute trio).

Importing this module is idempotent.
"""

from __future__ import annotations

import sys
import types


def _importable(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _install_placeholder_uvloop() -> None:
    if _importable("uvloop"):
        return
    module = types.ModuleType("uvloop")
    module.install = lambda *a, **k: None
    sys.modules["uvloop"] = module


def _install_placeholder_flag_gems() -> None:
    if "flag_gems" in sys.modules:
        return
    try:
        __import__("flag_gems")
        return
    except Exception:
        for name in [n for n in sys.modules if n.startswith("flag_gems")]:
            sys.modules.pop(name, None)

    flag_gems = types.ModuleType("flag_gems")
    flag_gems._FULL_CONFIG = ()
    runtime = types.ModuleType("flag_gems.runtime")
    backend = types.ModuleType("flag_gems.runtime.backend")
    backend.set_torch_backend_device_fn = lambda vendor: None
    backend.get_torch_backend_device_fn = lambda: None

    def _gen_torch_device_object():
        import torch

        return torch.cuda

    backend.gen_torch_device_object = _gen_torch_device_object
    device = types.ModuleType("flag_gems.runtime.backend.device")

    class DeviceDetector:
        def __init__(self, *args, **kwargs):
            self.vendor_name = "nvidia"
            self.device_type = "cuda"
            self.name = "cuda"
            self.dispatch_key = "CUDA"

    device.DeviceDetector = DeviceDetector
    flag_gems.runtime = runtime
    runtime.backend = backend
    backend.device = device
    for name, module in (
        ("flag_gems", flag_gems),
        ("flag_gems.runtime", runtime),
        ("flag_gems.runtime.backend", backend),
        ("flag_gems.runtime.backend.device", device),
    ):
        sys.modules[name] = module


def ensure_importable_env() -> None:
    """Install the placeholders; idempotent and a no-op on the CI image."""
    _install_placeholder_uvloop()
    _install_placeholder_flag_gems()


ensure_importable_env()

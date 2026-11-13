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

"""CPU-collection shim making the worker-area GPU file collectable on any host.

The A100 CI image (real FlagGems + triton + full vLLM) imports everything
natively, so every probe below first tries the real package and only falls
back to an inert placeholder when the import fails. Nothing is stubbed on a
host that can import the real module.

This mirrors the worker-area unit shim (``coverage_tests/unit/worker/
_imports.py``), trimmed to what this file's target needs:

* ``uvloop`` / ``xgrammar`` / ``flag_gems`` — the ``vllm_fl.worker`` import
  chain reaches all three; the placeholders expose only the import-time
  surface and are never exercised by the single-GPU tests.
* ``vllm_fl.compilation.graph`` — selects its graph class from
  ``current_platform.device_type`` at import time and raises
  ``NotImplementedError`` for anything CUDA-less. On a non-CUDA authoring
  host the module is imported once inside a CUDA-like platform window; the
  real platform is restored immediately afterwards, exactly the pattern
  ``coverage_tests/unit/worker/_imports.py`` uses. Modules imported inside
  the window hold a direct ``from vllm.platforms import current_platform``
  reference, so the real object is rebound onto every already-imported
  ``vllm_fl`` module to keep sibling suites unpolluted.

Importing this module is idempotent and never alters behavior on a host that
already runs a CUDA platform.
"""

from __future__ import annotations

import asyncio
import sys
import types


def _real_importable(name: str) -> bool:
    """Return True when the real package imports successfully."""
    if name in sys.modules:
        return True
    try:
        __import__(name)
        return True
    except Exception:
        # Undo any half-imported fragments so the placeholder can own the
        # name cleanly.
        for mod in [n for n in sys.modules if n == name or n.startswith(name + ".")]:
            sys.modules.pop(mod, None)
        return False


def _stub_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _ensure_uvloop() -> None:
    """Stub uvloop (no Windows wheel) with inert event-loop helpers."""
    if _real_importable("uvloop"):
        return
    module = _stub_module("uvloop", Loop=type("Loop", (), {}), install=lambda: None)
    module.new_event_loop = asyncio.new_event_loop


def _ensure_xgrammar() -> None:
    """Stub xgrammar with the class/function surface vllm reads at import."""
    if _real_importable("xgrammar"):
        return
    class_names = (
        "Grammar",
        "StructuralTag",
        "ConstStringFormat",
        "Format",
        "GrammarFormat",
        "JSONSchemaFormat",
        "OrFormat",
        "RegexFormat",
        "TagFormat",
        "TriggeredTagsFormat",
        "AnyTextFormat",
        "OptionalFormat",
        "PlusFormat",
        "SequenceFormat",
        "StarFormat",
        "TagsWithSeparatorFormat",
        "TokenTriggeredTagsFormat",
        "BuiltinToolParam",
        "FunctionToolParam",
    )
    classes = {name: type(name, (), {}) for name in class_names}
    xgrammar = _stub_module("xgrammar", **classes)
    xgrammar.normalize_tool_choice = lambda *args, **kwargs: None
    xgrammar.get_model_structural_tag = lambda *args, **kwargs: None
    structural_tag = _stub_module("xgrammar.structural_tag", **classes)
    xgrammar.structural_tag = structural_tag
    openai_tool = _stub_module(
        "xgrammar.openai_tool_call_schema",
        BuiltinToolParam=classes["BuiltinToolParam"],
        FunctionToolParam=classes["FunctionToolParam"],
    )
    xgrammar.openai_tool_call_schema = openai_tool


def _ensure_flag_gems() -> None:
    """Stub flag_gems (needs triton) with the vllm_fl.utils import surface."""
    if _real_importable("flag_gems"):
        return
    flag_gems = _stub_module("flag_gems", _FULL_CONFIG=())
    runtime = _stub_module("flag_gems.runtime")
    backend = _stub_module("flag_gems.runtime.backend")
    device = _stub_module("flag_gems.runtime.backend.device")

    class DeviceDetector:
        """Minimal detector matching the flag_gems <= 5.0.2 shape."""

        def __init__(self, *args, **kwargs):
            self.vendor_name = "nvidia"
            self.device_type = "cuda"
            self.name = "cuda"
            self.dispatch_key = "CUDA"

    device.DeviceDetector = DeviceDetector
    backend.set_torch_backend_device_fn = lambda vendor_name: None
    backend.gen_torch_device_object = lambda: None
    backend.get_torch_backend_device_fn = lambda: None
    flag_gems.runtime = runtime
    runtime.backend = backend
    backend.device = device


def _ensure_graph_import_window() -> None:
    """Import ``vllm_fl.compilation.graph`` under a CUDA-like platform.

    The module's ``Graph`` class reads ``current_platform.device_type`` at
    class-body time. On the CI image the real platform is already CUDA, so
    the window is skipped. Elsewhere the vLLM modules the graph module needs
    are pre-imported under the real platform first (the stand-in only exists
    to satisfy the device-type branch), then the platform is swapped for the
    duration of the import and restored in ``finally``.
    """
    import torch

    import vllm.platforms as platforms

    if "vllm_fl.compilation.graph" in sys.modules:
        return  # already imported by an earlier load; nothing to do

    real_platform = platforms.current_platform
    if real_platform.device_type == "cuda" and hasattr(
        real_platform, "torch_device_fn"
    ):
        return  # the NVIDIA CI image already supplies the runtime platform

    import vllm.config  # noqa: F401
    import vllm.utils.torch_utils  # noqa: F401  (pre-import under real platform)
    import vllm.v1.attention.backends.utils  # noqa: F401

    class _CudaLike:
        torch_device_fn = torch.cuda
        device_type = "cuda"
        vendor_name = "nvidia"
        dist_backend = "nccl"
        simple_compile_backend = "inductor"
        dispatch_key = "CUDA"

        @staticmethod
        def is_cuda():
            return True

        @staticmethod
        def is_rocm():
            return False

        @staticmethod
        def is_xpu():
            return False

        @staticmethod
        def is_cpu():
            return False

        @staticmethod
        def is_cuda_alike():
            return True

        @staticmethod
        def fp8_dtype():
            return torch.float8_e4m3fn

        @staticmethod
        def has_device_capability(capability):
            return False

    platforms.current_platform = _CudaLike()
    try:
        import vllm_fl.compilation.graph  # noqa: F401  (import side effects)
    finally:
        platforms.current_platform = real_platform
        # Modules imported inside the window did
        # ``from vllm.platforms import current_platform`` and hold a direct
        # reference to the stand-in object; repoint them at the real platform
        # now that it is restored.
        for name in list(sys.modules):
            if not name.startswith("vllm_fl"):
                continue
            module = sys.modules.get(name)
            if module is None or getattr(module, "current_platform", None) is None:
                continue
            if module.current_platform is real_platform:
                continue
            module.current_platform = real_platform


def install() -> None:
    """Install every needed placeholder (no-op for real installs)."""
    _ensure_uvloop()
    _ensure_xgrammar()
    _ensure_flag_gems()
    _ensure_graph_import_window()


install()

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

"""CPU-collection shim making the qwen3_8 area collectable on any host.

The A100 CI image (real FlagGems + triton + full vLLM) imports everything
natively, so every probe below first tries the real package and only falls
back to an inert placeholder when the import fails. Nothing is stubbed on a
host that can import the real module.

This mirrors the hy4/glm5 area shims and is needed here because importing the
model package pulls in ``vllm_fl.utils`` (which imports ``flag_gems``) and the
vLLM config/runner machinery. The ``flag_gems`` placeholder exposes exactly
the attributes ``vllm_fl.utils`` reads at import time; the detector reports
``nvidia``/``cuda``, matching the CUDA-shaped runtime these tests run under.

Importing this module is idempotent and never alters the behavior under test:
tensors are created explicitly on the CPU or on the session ``device`` fixture
and only sanctioned boundaries are monkeypatched.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import sys
import types

import torch


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
            self.name = "cuda"
            self.dispatch_key = "CUDA"

    device.DeviceDetector = DeviceDetector
    backend.set_torch_backend_device_fn = lambda vendor_name: None
    backend.gen_torch_device_object = lambda: None
    backend.get_torch_backend_device_fn = lambda: None
    flag_gems.runtime = runtime
    runtime.backend = backend
    backend.device = device


def _ensure_torchvision() -> None:
    """Stub torchvision (no Windows wheel) with the import-time symbol surface.

    Only names read while ``transformers`` imports its video/processor
    utilities are provided; none is invoked by these tests, which never build
    pixel tensors.
    """
    if _real_importable("torchvision"):
        return

    def pkg(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []
        module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        module.__version__ = "0.20.0"
        sys.modules[name] = module
        return module

    members = dict(
        NEAREST=0,
        NEAREST_EXACT=1,
        BILINEAR=2,
        BICUBIC=3,
        LANCZOS=4,
        BOX=5,
        HAMMING=6,
    )
    interpolation = type("InterpolationMode", (), members)

    torchvision = pkg("torchvision")
    transforms = pkg("torchvision.transforms")
    torchvision.transforms = transforms
    transforms.InterpolationMode = interpolation

    v2 = pkg("torchvision.transforms.v2")
    transforms.v2 = v2
    functional_v2 = types.ModuleType("torchvision.transforms.v2.functional")
    functional_v2.InterpolationMode = interpolation

    def _pad(image, padding, fill=0, padding_mode="constant"):
        """``tvF.pad`` using the (left, top, right, bottom) padding order."""
        left, top, right, bottom = padding
        return torch.nn.functional.pad(image, (left, right, top, bottom), value=fill)

    def _resize(image, new_size, interpolation=None, antialias=None):
        """Nearest-neighbor resize matching the ``torchvision`` signature."""
        try:
            height, width = new_size["height"], new_size["width"]
        except TypeError:
            height, width = new_size
        shape = tuple(image.shape)
        if len(shape) == 3:
            batch = image.unsqueeze(0)
        elif len(shape) == 4:
            batch = image
        else:
            batch = image.reshape(-1, *shape[-3:])
        out = torch.nn.functional.interpolate(
            batch, size=(height, width), mode="nearest"
        )
        return out.reshape(*shape[:-2], out.shape[-2], out.shape[-1])

    functional_v2.pad = _pad
    functional_v2.resize = _resize
    v2.functional = functional_v2
    sys.modules["torchvision.transforms.v2.functional"] = functional_v2

    functional = types.ModuleType("torchvision.transforms.functional")
    functional.InterpolationMode = interpolation
    functional.pil_to_tensor = lambda *args, **kwargs: None
    functional.to_pil_image = lambda *args, **kwargs: None
    sys.modules["torchvision.transforms.functional"] = functional

    io = types.ModuleType("torchvision.io")
    io.read_video = lambda *args, **kwargs: None
    io.read_video_timestamps = lambda *args, **kwargs: None
    torchvision.io = io
    sys.modules["torchvision.io"] = io


def install() -> None:
    """Install every needed placeholder (no-op for real installs)."""
    _ensure_uvloop()
    _ensure_xgrammar()
    _ensure_flag_gems()
    _ensure_torchvision()


install()

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

"""CPU-collection shim for hosts whose FlagGems/vLLM imports need triton.

The A100 CI image (real FlagGems + triton + full vLLM) imports everything
natively, so every probe below first tries the real package and only falls
back to an inert placeholder when the import fails. Nothing is stubbed on a
host that can import the real module:

* ``flag_gems`` — needs triton, for which no Windows wheel exists. The
  placeholder exposes exactly the attributes ``vllm_fl.utils`` reads at
  import time (``_FULL_CONFIG``, ``DeviceDetector``, backend functions).
  The detector reports ``nvidia``/``cuda``, matching the CUDA-shaped
  runtime that the tests below configure explicitly.
* ``uvloop`` — imported by ``vllm.v1.utils``; only the event-loop
  constructors are needed and never invoked by these tests.
* ``xgrammar`` — imported by ``vllm.transformers_utils``; only class and
  function symbols are needed for import to succeed.

Importing this module is idempotent. The placeholders never alter the
behavior under test: vendor identity, device probing and environment
variables are driven by the tests through the sanctioned boundaries
(vllm.platforms.current_platform, device probes, env vars — CONTRACT R2).
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
    """Stub xgrammar with the class/function surface vllm transformers_utils reads."""
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
        # Another suite may have installed a placeholder with a different
        # attribute shape (e.g. the dispatch-area shim, whose DeviceDetector
        # lacks the ``name`` field vllm_fl.utils reads). Upgrade its detector
        # in place so both areas' tests can run in one process.
        detector = getattr(
            sys.modules.get("flag_gems.runtime.backend.device"), "DeviceDetector", None
        )
        if detector is not None and not hasattr(detector(), "name"):
            for mod in [
                n for n in sys.modules if n == "flag_gems" or n.startswith("flag_gems.")
            ]:
                sys.modules.pop(mod, None)
            # vllm_fl.utils (if already imported) holds a reference to the old
            # detector class; rebind it to the upgraded one below.
            _rebind_vllm_fl_utils = True
        else:
            return
    else:
        _rebind_vllm_fl_utils = False
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

    if _rebind_vllm_fl_utils and "vllm_fl.utils" in sys.modules:
        sys.modules["vllm_fl.utils"].DeviceDetector = DeviceDetector


def install() -> None:
    """Install every needed placeholder (no-op for real installs)."""
    _ensure_uvloop()
    _ensure_xgrammar()
    _ensure_flag_gems()


install()

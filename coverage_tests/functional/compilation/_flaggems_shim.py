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

"""CPU-collection shim for hosts without a usable FlagGems install.

The plugin's ``vllm_fl.utils`` imports ``flag_gems`` at module level, and
FlagGems imports ``triton`` — for which no Windows wheel exists. On the
A100 CI image (and any host with real FlagGems) this module changes
nothing: it first probes whether the real package imports and only then
falls back to installing an inert ``flag_gems`` placeholder that exposes
exactly the three attributes ``vllm_fl.utils`` reads at import time:

* ``flag_gems._FULL_CONFIG`` (read lazily inside
  ``vllm_fl.utils.get_flaggems_all_ops``),
* ``flag_gems.runtime.backend.device.DeviceDetector`` (only used when
  the local platform probe finds no device),
* ``flag_gems.runtime.backend`` (``set_torch_backend_device_fn`` /
  ``gen_torch_device_object`` / ``get_torch_backend_device_fn`` — only
  used on live CUDA runtimes).

None of the dispatch/io modules under test here touch any of those
attributes, so the placeholder cannot alter the behavior under test.
Importing this module is idempotent.
"""

from __future__ import annotations

import sys
import types


def _real_flag_gems_importable() -> bool:
    """Return True when the real flag_gems package imports successfully."""
    if "flag_gems" in sys.modules:
        return True
    try:
        __import__("flag_gems")
        return True
    except Exception:
        # Undo any half-imported fragments so the placeholder below can
        # own the name cleanly.
        for name in [n for n in sys.modules if n.startswith("flag_gems")]:
            sys.modules.pop(name, None)
        return False


def _install_placeholder_flag_gems() -> None:
    """Install an inert flag_gems package into sys.modules (no-op if real)."""
    if _real_flag_gems_importable():
        return

    flag_gems = types.ModuleType("flag_gems")
    flag_gems._FULL_CONFIG = ()

    runtime = types.ModuleType("flag_gems.runtime")
    backend = types.ModuleType("flag_gems.runtime.backend")

    def _set_torch_backend_device_fn(vendor_name: str) -> None:
        # Inert: only reached on live accelerator runtimes.
        return None

    def _gen_torch_device_object():
        import torch

        return torch.cuda

    def _get_torch_backend_device_fn():
        return None

    backend.set_torch_backend_device_fn = _set_torch_backend_device_fn
    backend.gen_torch_device_object = _gen_torch_device_object
    backend.get_torch_backend_device_fn = _get_torch_backend_device_fn

    device = types.ModuleType("flag_gems.runtime.backend.device")

    class DeviceDetector:
        """Minimal detector matching the flag_gems <= 5.0.2 attribute shape."""

        def __init__(self, *args, **kwargs):
            self.vendor_name = "nvidia"
            self.device_type = "cuda"
            self.name = "cuda"  # vllm_fl DeviceInfo.device_type reads .name
            self.dispatch_key = "CUDA"  # vllm_fl DeviceInfo.dispatch_key

    device.DeviceDetector = DeviceDetector

    flag_gems.runtime = runtime
    runtime.backend = backend
    backend.device = device

    sys.modules["flag_gems"] = flag_gems
    sys.modules["flag_gems.runtime"] = runtime
    sys.modules["flag_gems.runtime.backend"] = backend
    sys.modules["flag_gems.runtime.backend.device"] = device


def _install_placeholder_xgrammar() -> None:
    """Stub xgrammar (Linux-only) for hosts without a wheel.

    vllm.parser imports Grammar/StructuralTag at module import; the
    sparse-indexer chain pulls vllm.parser on Windows hosts.
    """
    try:
        __import__("xgrammar")
        return
    except ImportError:
        pass
    xg = types.ModuleType("xgrammar")
    xg.__path__ = []  # a package so submodule imports resolve

    class Grammar:
        pass

    class StructuralTag:
        pass

    class GrammarMatcher:
        def __init__(self, *args, **kwargs):
            pass

        def fill_next_token_bitmask(self, *args, **kwargs):
            return None

    xg.Grammar = Grammar
    xg.StructuralTag = StructuralTag
    xg.GrammarMatcher = GrammarMatcher
    xg.BatchingGrammarMatcher = GrammarMatcher
    xg.normalize_tool_choice = lambda *args, **kwargs: None
    xg.get_model_structural_tag = lambda *args, **kwargs: None
    st = types.ModuleType("xgrammar.structural_tag")

    class _TagFormatBase:
        def __init__(self, *args, **kwargs):
            pass

    class Element:
        def __init__(self, *args, **kwargs):
            pass

    class StructuralTagInfo:
        def __init__(self, *args, **kwargs):
            pass

    st.Element = Element
    st.StructuralTagInfo = StructuralTagInfo
    # Every *Format name any vllm parser revision imports; inert and
    # constructor-shaped so class checks fail loudly rather than silently.
    for _fmt in (
        "AnyTextFormat",
        "ConstStringFormat",
        "Format",
        "GrammarFormat",
        "JSONSchemaFormat",
        "OptionalFormat",
        "OrFormat",
        "PlusFormat",
        "RegexFormat",
        "SequenceFormat",
        "StarFormat",
        "TagFormat",
        "TagsWithSeparatorFormat",
        "TokenTriggeredTagsFormat",
        "TriggeredTagsFormat",
    ):
        setattr(st, _fmt, type(_fmt, (_TagFormatBase,), {}))
    xg.structural_tag = st
    ots = types.ModuleType("xgrammar.openai_tool_call_schema")

    class FunctionToolParam:
        def __init__(self, *args, **kwargs):
            pass

    ots.FunctionToolParam = FunctionToolParam

    class BuiltinToolParam:
        def __init__(self, *args, **kwargs):
            pass

    ots.BuiltinToolParam = BuiltinToolParam
    xg.openai_tool_call_schema = ots
    sys.modules["xgrammar"] = xg
    sys.modules["xgrammar.structural_tag"] = st
    sys.modules["xgrammar.openai_tool_call_schema"] = ots


def _install_placeholder_uvloop() -> None:
    """Stub uvloop (Linux-only) for hosts without a wheel.

    vllm.v1.utils imports uvloop at module import; the compilation test
    files pull vllm.v1 worker modules on Windows hosts. Real installs
    keep the real package.
    """
    try:
        __import__("uvloop")
        return
    except ImportError:
        pass
    uvloop = types.ModuleType("uvloop")

    class Loop:
        pass

    uvloop.Loop = Loop
    uvloop.install = lambda: None
    sys.modules["uvloop"] = uvloop


_install_placeholder_uvloop()
_install_placeholder_xgrammar()
_install_placeholder_flag_gems()

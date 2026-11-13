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

"""Import shim making the fused-MoE ops area collectable on a CPU-only host.

The dev venv on this Windows host has no ``triton`` (no Windows wheel), so
the installed ``flag_gems`` package cannot import and ``vllm``'s
structured-output chain cannot import ``xgrammar``. The A100 CI image has
every real package; there each step below is a probe-guarded no-op:

* ``uvloop`` — imported by ``vllm.v1.utils``; only the event-loop installer
  is stubbed.
* ``flag_gems`` — same inert placeholder the dispatch/worker areas use
  (``_FULL_CONFIG`` plus the ``runtime.backend`` attribute trio).
* ``xgrammar`` — vllm's parser modules import a fixed name set; the
  placeholder mirrors the worker-area recipe exactly.
* ``vllm...fused_moe.layer.FusedMoE`` / package ``FusedMoE`` — the factory
  is named ``FusedMoEFactory`` in the local 0.33 build; the alias points
  back at it so the plugin's imports resolve.
* ``FlashinferMoeBackend`` — removed from ``flashinfer_utils`` in 0.33; the
  placeholder enum exposes the two members the plugin compares against.

Import order matters: run ``ensure_importable_env()`` before importing
``vllm_fl.ops.fused_moe``. Importing this module is idempotent.
"""

from __future__ import annotations

import enum
import sys
import types


def _importable(name: str) -> bool:
    """Return True when ``name`` is imported or imports cleanly."""
    if name in sys.modules:
        return True
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _install_placeholder_uvloop() -> None:
    """Install an inert uvloop when absent (vllm.v1 imports it)."""
    if _importable("uvloop"):
        return
    module = types.ModuleType("uvloop")
    module.install = lambda *a, **k: None
    sys.modules["uvloop"] = module


def _install_placeholder_flag_gems() -> None:
    """Install the inert flag_gems placeholder (dispatch-area recipe)."""
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


_STRUCTURAL_NAMES = (
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
    "StructuralTagItems",
    "TagFormat",
    "TagsWithSeparatorFormat",
    "TokenTriggeredTagsFormat",
    "TriggeredTagsFormat",
    "TriggerSpaces",
    "buildStructuralTag",
)


def _install_xgrammar_compat() -> None:
    """Install an inert xgrammar with the name set vllm's parser reads.

    A previous suite may already have put a foreign xgrammar placeholder
    into ``sys.modules``; its symbol set is merged rather than replaced. A
    real install (has a ``__file__``) is never touched.
    """
    if "xgrammar" in sys.modules:
        existing = sys.modules["xgrammar"]
        if getattr(existing, "__file__", None) is not None:
            return
        _merge_xgrammar_symbols(existing)
        return
    if _importable("xgrammar"):
        return

    xgrammar = types.ModuleType("xgrammar")
    xgrammar.__path__ = []
    xgrammar.Grammar = object
    xgrammar.StructuralTag = object
    xgrammar.normalize_tool_choice = lambda *a, **k: None
    xgrammar.get_model_structural_tag = lambda *a, **k: None
    xgrammar.BatchGrammarMatcher = _CompatBatchGrammarMatcher
    _merge_xgrammar_symbols(xgrammar)
    sys.modules["xgrammar"] = xgrammar

    structural_tag = types.ModuleType("xgrammar.structural_tag")
    for attr in _STRUCTURAL_NAMES:
        setattr(structural_tag, attr, object)
    sys.modules["xgrammar.structural_tag"] = structural_tag
    xgrammar.structural_tag = structural_tag

    tool_call_schema = types.ModuleType("xgrammar.openai_tool_call_schema")
    tool_call_schema.BuiltinToolParam = object
    tool_call_schema.FunctionToolParam = object
    sys.modules["xgrammar.openai_tool_call_schema"] = tool_call_schema
    xgrammar.openai_tool_call_schema = tool_call_schema


class _CompatBatchGrammarMatcher:
    """Recorder-shaped stand-in; ops tests never call it."""

    def __init__(self, max_threads: int = 1):
        self.max_threads = max_threads


def _merge_xgrammar_symbols(module) -> None:
    for name, value in (
        ("Grammar", object),
        ("StructuralTag", object),
        ("normalize_tool_choice", lambda *a, **k: None),
        ("get_model_structural_tag", lambda *a, **k: None),
        ("BatchGrammarMatcher", _CompatBatchGrammarMatcher),
    ):
        if not hasattr(module, name):
            setattr(module, name, value)
    for name in _STRUCTURAL_NAMES:
        if not hasattr(module, name):
            setattr(module, name, object)


def _install_vllm_symbol_backfills() -> None:
    """Backfill vllm symbols absent from the local 0.33 build."""
    import vllm.model_executor.layers.fused_moe as fused_moe_pkg
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    if not hasattr(fused_moe_layer, "FusedMoE"):
        fused_moe_layer.FusedMoE = fused_moe_layer.FusedMoEFactory
    if not hasattr(fused_moe_pkg, "FusedMoE"):
        fused_moe_pkg.FusedMoE = fused_moe_layer.FusedMoEFactory

    import vllm.model_executor.layers.quantization.utils.flashinfer_utils as fiu

    if not hasattr(fiu, "FlashinferMoeBackend"):

        class FlashinferMoeBackend(str, enum.Enum):
            CUTLASS = "cutlass"
            TENSORRT_LLM = "trtllm"

        fiu.FlashinferMoeBackend = FlashinferMoeBackend


def ensure_importable_env() -> None:
    """Install every placeholder/backfill needed for the ops imports.

    Idempotent; on the A100 CI image each step is a no-op.
    """
    _install_placeholder_uvloop()
    _install_placeholder_flag_gems()
    _install_xgrammar_compat()
    _install_vllm_symbol_backfills()


ensure_importable_env()

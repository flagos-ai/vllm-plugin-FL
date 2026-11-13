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

"""Import shim making the W4 worker-area targets collectable on any host.

The dev venv on this Windows host (no triton, no flag_gems, no xgrammar,
no uvloop, vllm 0.33 instead of the CI image's 0.24-line) cannot import
``vllm_fl.worker.worker`` / ``vllm_fl.worker.scheduler_fl`` without help.
The A100 CI image has every real package, and there each step below is a
no-op guarded by a probe of the real symbol/module:

* ``uvloop`` — imported by ``vllm.v1.utils``; only the event-loop
  installer is stubbed.
* ``flag_gems`` — same inert placeholder the dispatch/compilation areas
  use (``_FULL_CONFIG`` plus the runtime.backend attribute trio).
* ``xgrammar`` — vllm's parser module imports a fixed name set from it;
  the placeholder mirrors ``kv_transfer/_flagcx_env`` plus the
  ``BatchGrammarMatcher`` the plugin's scheduler reads.
* ``vllm.v1.worker.utils.is_residual_scattered_for_sp`` — removed in
  vllm 0.33; the shim installs a raise-if-called placeholder (the SP
  path is never exercised by these CPU tests).
* ``vllm...fused_moe.layer.FusedMoE`` / package ``FusedMoE`` — the
  factory was renamed to ``FusedMoEFactory`` in 0.33; the alias points
  back at it so the plugin's imports resolve.
* ``FlashinferMoeBackend`` — removed from flashinfer_utils in 0.33; the
  placeholder enum exposes the two members the plugin's dead branch
  compares against.
* ``prepare_communication_buffer_for_model``, ``SparseWeightPatch``,
  ``XDRotaryEmbedding``, ``SupportsXDRoPE``/``supports_xdrope``,
  ``is_quantized_kv_cache``, ``RoutedExpertsLists``/
  ``RoutedExpertsTensors``, ``vllm.v1.spec_decode.medusa``,
  ``get_total_cp_world_size``, ``_reshape_attention_kv_cache`` and the
  moved ``LateInteractionRunner`` — all symbols the CI image's vllm
  provides at the paths the plugin imports; local-only backfills keep
  ``vllm_fl.worker.model_runner`` importable. Raise-if-called
  placeholders are used where a test would otherwise silently pass
  through a stubbed behavior.
* ``FusedMoEParallelConfig.make`` — local vllm 0.33 added a required
  ``sp_size_`` positional the CI signature lacks; the shim wraps
  ``make`` to pass ``sp_size_=1`` only when the local signature
  requires it.

Import order matters: run ``ensure_importable_env()`` before importing
any ``vllm_fl.worker`` module. Importing this module is idempotent.
"""

from __future__ import annotations

import enum
import importlib
import inspect
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
    if _importable("flag_gems"):
        return
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
            self.name = "cuda"  # vllm_fl DeviceInfo.device_type reads .name
            self.dispatch_key = "CUDA"  # vllm_fl DeviceInfo.dispatch_key

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


class _CompatBatchGrammarMatcher:
    """Recorder-shaped stand-in; tests replace instance state anyway."""

    def __init__(self, max_threads: int = 1):
        self.max_threads = max_threads

    def batch_fill_next_token_bitmask(self, matchers, bitmask, indices):
        return None


def _xgrammar_extra_symbols(sub_name: str = "") -> dict:
    """Symbols this area's chain needs beyond a foreign stub's surface.

    ``vllm_fl.worker.scheduler_fl`` instantiates
    ``xgrammar.BatchGrammarMatcher``; a narrower placeholder from another
    area (e.g. the platform shim) does not define it. ``vllm`` 0.33's
    parser modules read the rest of the names below from the top-level
    package and its ``structural_tag`` / ``openai_tool_call_schema``
    submodules; on the CI image the real xgrammar provides them.
    """
    if sub_name == "structural_tag":
        return {
            name: object
            for name in (
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
        }
    if sub_name == "openai_tool_call_schema":
        return {"BuiltinToolParam": object, "FunctionToolParam": object}
    return {
        "Grammar": object,
        "StructuralTag": object,
        "normalize_tool_choice": lambda *a, **k: None,
        "get_model_structural_tag": lambda *a, **k: None,
        "BatchGrammarMatcher": _CompatBatchGrammarMatcher,
    }


def _install_xgrammar_compat() -> None:
    """Install an inert xgrammar with the name set vllm's parser reads.

    A previous suite may already have put an xgrammar placeholder into
    sys.modules (e.g. the platform-area shim), and its symbol set can be
    narrower than what vllm_fl.worker.scheduler_fl reads. A real install
    (has a ``__file__``) is never touched; a foreign placeholder gets the
    missing symbols merged in so both areas' tests run in one process.
    """
    if "xgrammar" in sys.modules:
        existing = sys.modules["xgrammar"]
        if getattr(existing, "__file__", None) is not None:
            return  # real package: nothing to add
        for name, value in _xgrammar_extra_symbols().items():
            if not hasattr(existing, name):
                setattr(existing, name, value)
        for sub_name in ("structural_tag", "openai_tool_call_schema"):
            sub = getattr(existing, sub_name, None)
            if sub is None:
                continue
            for name, value in _xgrammar_extra_symbols(sub_name).items():
                if not hasattr(sub, name):
                    setattr(sub, name, value)
        return
    if _importable("xgrammar"):
        return
    xgrammar = types.ModuleType("xgrammar")
    xgrammar.Grammar = object
    xgrammar.StructuralTag = object
    xgrammar.normalize_tool_choice = lambda *a, **k: None
    xgrammar.get_model_structural_tag = lambda *a, **k: None
    xgrammar.__path__ = []
    xgrammar.BatchGrammarMatcher = _CompatBatchGrammarMatcher
    sys.modules["xgrammar"] = xgrammar
    for name, value in _xgrammar_extra_symbols().items():
        if not hasattr(xgrammar, name):
            setattr(xgrammar, name, value)

    structural_tag = types.ModuleType("xgrammar.structural_tag")
    for attr in (
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
    ):
        setattr(structural_tag, attr, object)
    sys.modules["xgrammar.structural_tag"] = structural_tag
    for name, value in _xgrammar_extra_symbols("structural_tag").items():
        if not hasattr(structural_tag, name):
            setattr(structural_tag, name, value)

    tool_call_schema = types.ModuleType("xgrammar.openai_tool_call_schema")
    tool_call_schema.BuiltinToolParam = object
    tool_call_schema.FunctionToolParam = object
    sys.modules["xgrammar.openai_tool_call_schema"] = tool_call_schema
    xgrammar.openai_tool_call_schema = tool_call_schema


def _install_vllm_symbol_backfills() -> None:
    """Backfill vllm symbols absent from dev-only vllm builds.

    On the CI image (vllm 0.24-line) every symbol below exists at the
    path the plugin imports, so each ``hasattr`` guard skips. Locally
    (vllm 0.33) the imports resolve after the backfill; placeholders
    that would mask behavior raise when called.
    """
    import vllm.v1.worker.utils as worker_utils

    if not hasattr(worker_utils, "is_residual_scattered_for_sp"):

        def _sp_unavailable(*args, **kwargs):
            raise AssertionError("SP path is not exercised by CPU tests")

        worker_utils.is_residual_scattered_for_sp = _sp_unavailable

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

    import vllm.distributed.parallel_state as parallel_state

    if not hasattr(parallel_state, "prepare_communication_buffer_for_model"):
        parallel_state.prepare_communication_buffer_for_model = lambda model: None

    import vllm.distributed.weight_transfer.base as weight_base

    if not hasattr(weight_base, "SparseWeightPatch"):
        weight_base.SparseWeightPatch = importlib.import_module(
            "vllm.distributed.weight_transfer.sparse_nccl_engine"
        ).SparseWeightPatch

    import vllm.model_executor.layers.rotary_embedding as rotary

    if not hasattr(rotary, "XDRotaryEmbedding"):

        class _XDRotaryEmbeddingCompat:
            @staticmethod
            def get_next_input_positions_tensor(*args, **kwargs):
                raise AssertionError("XD-rope helper is not exercised by CPU tests")

        rotary.XDRotaryEmbedding = _XDRotaryEmbeddingCompat

    import vllm.model_executor.models.interfaces as interfaces

    if not hasattr(interfaces, "SupportsXDRoPE"):

        class _SupportsXDRoPECompat:
            supports_xdrope: bool = False

        interfaces.SupportsXDRoPE = _SupportsXDRoPECompat
    if not hasattr(interfaces, "supports_xdrope"):
        interfaces.supports_xdrope = lambda model: getattr(
            model, "supports_xdrope", False
        )

    import vllm.utils.torch_utils as torch_utils

    if not hasattr(torch_utils, "is_quantized_kv_cache"):
        torch_utils.is_quantized_kv_cache = importlib.import_module(
            "vllm.v1.kv_cache_interface"
        ).is_quantized_kv_cache

    import vllm.v1.outputs as v1_outputs

    if not hasattr(v1_outputs, "RoutedExpertsLists"):
        v1_outputs.RoutedExpertsLists = type(
            "_RoutedExpertsListsCompat", (), {"__init__": lambda self, **kw: None}
        )
    if not hasattr(v1_outputs, "RoutedExpertsTensors"):
        v1_outputs.RoutedExpertsTensors = type(
            "_RoutedExpertsTensorsCompat", (), {"__init__": lambda self, **kw: None}
        )

    try:
        importlib.import_module("vllm.v1.spec_decode.medusa")
    except ModuleNotFoundError:
        medusa = types.ModuleType("vllm.v1.spec_decode.medusa")

        class _MedusaProposerCompat:
            def __init__(self, *args, **kwargs):
                raise AssertionError("Medusa drafter is not exercised by CPU tests")

        medusa.MedusaProposer = _MedusaProposerCompat
        sys.modules["vllm.v1.spec_decode.medusa"] = medusa

    import vllm.v1.worker.cp_utils as cp_utils

    if not hasattr(cp_utils, "get_total_cp_world_size"):

        def _get_total_cp_world_size():
            from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group

            return get_pcp_group().world_size * get_dcp_group().world_size

        cp_utils.get_total_cp_world_size = _get_total_cp_world_size

    import vllm.v1.worker.gpu.attn_utils as attn_utils

    if not hasattr(attn_utils, "_reshape_attention_kv_cache"):

        def _reshape_attention_kv_cache_compat(*args, **kwargs):
            raise AssertionError("KV-cache reshape is not exercised by CPU tests")

        attn_utils._reshape_attention_kv_cache = _reshape_attention_kv_cache_compat

    try:
        importlib.import_module("vllm.v1.worker.gpu.pool.late_interaction_runner")
    except ModuleNotFoundError:
        moved = types.ModuleType("vllm.v1.worker.gpu.pool.late_interaction_runner")
        moved.LateInteractionRunner = importlib.import_module(
            "vllm.v1.pool.late_interaction_runner"
        ).LateInteractionRunner
        sys.modules["vllm.v1.worker.gpu.pool.late_interaction_runner"] = moved

    try:
        batch_invariant = importlib.import_module(
            "vllm.model_executor.layers.batch_invariant"
        )
    except ModuleNotFoundError:
        # vllm 0.33 moved batch_invariant under determinism/; alias the
        # old path so worker.py's function-local import resolves.
        batch_invariant = importlib.import_module(
            "vllm.model_executor.determinism.batch_invariant"
        )
        sys.modules["vllm.model_executor.layers.batch_invariant"] = batch_invariant
    if not hasattr(batch_invariant, "init_batch_invariance"):
        raise AssertionError(  # pragma: no cover - guards the alias above
            "batch_invariant module lacks init_batch_invariance"
        )


def _install_fused_moe_make_compat() -> None:
    """Wrap FusedMoEParallelConfig.make when local vllm needs sp_size_.

    The CI image's vllm 0.24-line signature is
    ``make(tp_size_, pcp_size_, dp_size_, vllm_parallel_config)``;
    vllm 0.33 added a required ``sp_size_`` positional. The plugin calls
    the 4-argument form, so on 0.33 builds the shim installs a wrapper
    that supplies ``sp_size_=1`` (no sequence parallelism) — the same
    default the CI signature implies. On the CI image the guard detects
    the 4-parameter signature and installs nothing.
    """
    import vllm.model_executor.layers.fused_moe.config as moe_config

    make = moe_config.FusedMoEParallelConfig.make
    required = [
        name
        for name, param in inspect.signature(make).parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if "sp_size_" not in required:
        return

    def _make_compat(tp_size_, pcp_size_, dp_size_, vllm_parallel_config, **kwargs):
        return make(
            tp_size_=tp_size_,
            pcp_size_=pcp_size_,
            dp_size_=dp_size_,
            sp_size_=1,
            vllm_parallel_config=vllm_parallel_config,
            **kwargs,
        )

    moe_config.FusedMoEParallelConfig.make = staticmethod(_make_compat)
    # The layer module re-exports the config class; keep its binding aligned.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    if getattr(fused_moe_layer.FusedMoEParallelConfig, "make", None) is not None:
        fused_moe_layer.FusedMoEParallelConfig = moe_config.FusedMoEParallelConfig


def _rebind_vllm_fl_platform_references(real_platform) -> None:
    """Repoint ``vllm_fl`` from-import bindings of current_platform.

    Modules that did ``from vllm.platforms import current_platform``
    while the stand-in was installed keep a direct reference to the
    stand-in object even after ``vllm.platforms.current_platform`` is
    restored. Sibling suites (e.g. the compilation-area graph tests)
    monkeypatch attributes on the platform object those modules hold, so
    leaving the stand-in bound in any of them poisons those tests. Every
    already-imported ``vllm_fl`` module gets the real platform back; on
    the CI image the swap never happens, so this loop rebinds nothing
    (every reference already is the real platform).
    """
    for name in list(sys.modules):
        if not name.startswith("vllm_fl"):
            continue
        module = sys.modules.get(name)
        if module is None or getattr(module, "current_platform", None) is None:
            continue
        if module.current_platform is real_platform:
            continue
        module.current_platform = real_platform


def _install_model_runner_platform() -> None:
    """Import the worker chain under a CUDA-like platform, then restore.

    ``vllm_fl.compilation.graph`` selects its Graph class at import time
    from ``current_platform.device_type`` and raises NotImplementedError
    for anything CUDA-less, and ``vllm_fl.worker.model_runner`` binds
    several module attributes the same way. The real platform object is
    a module attribute of ``vllm.platforms`` (em-dash) an input other
    suites (platform-area tests) also read through
    ``vllm.config.device.DeviceConfig.__post_init__``; leaving the
    stand-in installed would poison them. So the swap below covers only
    the import window: the chain is imported here (sys.modules caches it
    for the test files own imports) and the real platform is restored
    immediately afterwards, exactly the pattern the compilation-area
    tests use for their own import.
    """
    import torch

    import vllm.platforms as platforms

    if "vllm_fl.worker.worker" in sys.modules:
        return  # already imported by an earlier load; nothing to do

    real_platform = platforms.current_platform
    if real_platform.device_type == "cuda" and hasattr(
        real_platform, "torch_device_fn"
    ):
        return  # The NVIDIA CI plugin already supplies the runtime platform.

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
        import vllm_fl.worker.model_runner  # noqa: F401  (import side effects)
        import vllm_fl.worker.model_state  # noqa: F401  (target under test)
        import vllm_fl.worker.scheduler_fl  # noqa: F401  (target under test)
        import vllm_fl.worker.worker  # noqa: F401  (target under test)
    finally:
        platforms.current_platform = real_platform
        # Modules imported inside the window did
        # ``from vllm.platforms import current_platform`` and hold a direct
        # reference to the stand-in object; repoint them at the real
        # platform now that it is restored. The Graph class body has
        # already read device_type once (its only import-time platform
        # read), so rebinding does not affect it.
        _rebind_vllm_fl_platform_references(real_platform)


def ensure_importable_env() -> None:
    """Install every placeholder/backfill needed for vllm_fl.worker imports.

    Idempotent; on the A100 CI image each step is a no-op. The platform
    step imports the ``vllm_fl.worker`` chain under a CUDA-like platform
    and restores the real platform afterwards (see
    ``_install_model_runner_platform``), so nothing this shim changes
    outlives the import window except module caches.
    """
    _install_placeholder_uvloop()
    _install_placeholder_flag_gems()
    _install_xgrammar_compat()
    _install_vllm_symbol_backfills()
    _install_fused_moe_make_compat()
    _install_model_runner_platform()


ensure_importable_env()

# Copyright (c) 2026 BAAI. All rights reserved.

"""Backend-selection and TritonExpertsFL tests for ``vllm_fl.ops.fused_moe.fused_moe_utils``.

The OOT backend-selection oracle (``select_unquantized_moe_backend_oot``,
``_get_priority_backends``, ``_move_to_back``, the log closures and
``_return_or_raise``), ``_prepare_expert_assignment``'s naive-block fast
path, the ``FlashinferMoeBackend`` version shim, and ``TritonExpertsFL``'s
``apply``/``moe_sum``/``workspace_shapes`` dispatch boundaries — the parts of
``vllm_fl.ops.fused_moe.fused_moe_utils`` the core suite leaves uncovered.

The module imports ``vllm`` at module level.  On machines without the real
package a stub ``vllm`` graph is installed first (external boundary only);
with the real ``vllm`` present the module imports against it.  Either
way, every platform probe, config lookup, kernel class, and dispatched op is
swapped at test time via ``monkeypatch`` on module attributes, so the code
under test always runs its own logic.

The FlagGems-MoE policy gate ``_should_use_fl_triton_experts`` is covered by
``tests/unit_tests/ops/test_fused_moe_factory.py``'s
``test_fl_triton_experts_policy`` truth table and is not re-tested here.
"""

import sys
import types
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Windows (no real vllm): install the external vllm boundary the fused_moe
# modules read at import time.  With the real vllm installed the modules
# import against it directly and this block is skipped.
# ---------------------------------------------------------------------------

_HAS_VLLM = True
try:  # pragma: no cover - dual-environment guard
    import vllm  # noqa: F401
except ImportError:  # pragma: no cover - Windows machines without vllm
    _HAS_VLLM = False


def _make_module(name: str, **attrs: Any) -> types.ModuleType:
    """Create a plain module with the given attributes attached."""
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _StubLogger:
    """vllm.logger stand-in supporting the *_once APIs the selector uses.

    Records are forwarded to the standard library's module-level logger so
    pytest's ``caplog`` can observe the selection messages.
    """

    def __init__(self):
        import logging

        self._log = logging.getLogger("vllm_stub")

    def info(self, msg, *args):
        self._log.info(msg % args if args else msg)

    def debug(self, msg, *args):
        self._log.debug(msg % args if args else msg)

    def warning(self, msg, *args):
        self._log.warning(msg % args if args else msg)

    def info_once(self, msg, *args):
        self._log.info(msg % args if args else msg)

    def debug_once(self, msg, *args):
        self._log.debug(msg % args if args else msg)

    def warning_once(self, msg, *args):
        self._log.warning(msg % args if args else msg)


def _install_stub_vllm_graph() -> None:
    """Install the fake vllm graph the target modules import at module level.

    Only names read at import time are provided (enums, config classes,
    helper signatures); every function under test still runs its own logic.
    """
    import enum

    class _StubMoEActivation(enum.Enum):
        SILU = "silu"
        GELU = "gelu"
        SILU_NO_MUL = "silu_no_mul"
        GELU_NO_MUL = "gelu_no_mul"
        GELU_TANH = "gelu_tanh"
        RELU2 = "relu2"
        RELU2_NO_MUL = "relu2_no_mul"
        SITU = "situ"
        SWIGLUOAI = "swigluoai"
        SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
        SWIGLUSTEP = "swiglustep"

        @property
        def is_gated(self):
            return not self.name.endswith("NO_MUL")

    class _StubBackend(enum.Enum):
        AITER = "ROCm AITER"
        TRITON = "TRITON"
        BATCHED_TRITON = "BATCHED_TRITON"
        FLASHINFER_TRTLLM = "FlashInfer TRTLLM"
        FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
        XPU = "XPU"
        CPU = "CPU"
        TPU = "TPU"

    class _StubActivationFormat(enum.Enum):
        BatchedExperts = "batched"
        Standard = "standard"

    vllm = _make_module("vllm")
    platforms = _make_module(
        "vllm.platforms",
        current_platform=types.SimpleNamespace(
            device_type="cpu", vendor_name=None, is_cuda=lambda: False
        ),
    )
    logger_mod = _make_module("vllm.logger", init_logger=lambda name: _StubLogger())
    envs_mod = _make_module(
        "vllm.envs",
        is_set=lambda name: False,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    aiter_ops = _make_module(
        "vllm._aiter_ops",
        # The namespace the FL router probes at call time; parity with the
        # real vllm._aiter_ops surface so monkeypatch targets always exist.
        rocm_aiter_ops=types.SimpleNamespace(
            is_rdna_aiter_enabled=lambda: False,
            is_fused_moe_enabled=lambda: False,
            is_fusion_moe_shared_experts_enabled=lambda: False,
        ),
    )
    me = _make_module("vllm.model_executor")
    layers = _make_module("vllm.model_executor.layers")
    fm = _make_module("vllm.model_executor.layers.fused_moe")
    fm.FusedMoE = object
    fm.FusedMoEFactory = None
    experts_dir = _make_module("vllm.model_executor.layers.fused_moe.experts")
    router_dir = _make_module("vllm.model_executor.layers.fused_moe.router")
    runner_dir = _make_module("vllm.model_executor.layers.fused_moe.runner")
    oracle_dir = _make_module("vllm.model_executor.layers.fused_moe.oracle")
    fm_config = _make_module(
        "vllm.model_executor.layers.fused_moe.config",
        FusedMoEConfig=object,
        FUSED_MOE_UNQUANTIZED_CONFIG=None,
        FusedMoEQuantConfig=object,
        _get_config_dtype_str=lambda **kw: "float32",
    )
    fm_runner = _make_module(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner", MoERunner=object
    )
    unq_method = _make_module(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
        UnquantizedFusedMoEMethod=object,
        select_unquantized_moe_backend=lambda *a, **k: None,
    )
    fm_utils_v = _make_module(
        "vllm.model_executor.layers.fused_moe.utils",
        _resize_cache=lambda t, shape: t.view(*shape),
        moe_kernel_quantize_input=lambda A, *a, **kw: (A, None),
    )
    fm_activation = _make_module(
        "vllm.model_executor.layers.fused_moe.activation",
        MoEActivation=_StubMoEActivation,
        ApplyMoEActivationConfig=object,
        apply_moe_activation=lambda *a, **k: None,
    )
    fm_fused_v = _make_module(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        try_get_optimal_moe_config=lambda *a, **k: {"BLOCK_SIZE_M": 16},
        _get_config_quant_dtype=lambda **kw: None,
        invoke_fused_moe_triton_kernel=lambda *a, **k: None,
    )
    oracle_unq = _make_module(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized",
        UnquantizedMoeBackend=_StubBackend,
        _trtllm_bf16_lora_supported=lambda *a, **k: False,
        backend_to_kernel_cls=lambda backend: [],
        map_unquantized_backend=lambda name: None,
        select_unquantized_moe_backend=lambda *a, **k: None,
    )
    modular = _make_module(
        "vllm.model_executor.layers.fused_moe.modular_kernel",
        FusedMoEExperts=object,
        ExpertTokensMetadata=object,
        FusedMoEActivationFormat=_StubActivationFormat,
    )

    class _StubTritonExperts:
        """Upstream TritonExperts stand-in accepting the constructor args."""

        def __init__(self, moe_config, *args, **kwargs):
            self.moe_config = moe_config

        def workspace_shapes(self, *args, **kwargs):
            raise AssertionError("upstream workspace_shapes reached")

        def apply(self, *args, **kwargs):
            raise AssertionError("upstream apply reached")

    experts_triton = _make_module(
        "vllm.model_executor.layers.fused_moe.experts.triton_moe",
        TritonExperts=_StubTritonExperts,
    )
    experts_trtllm = _make_module(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe",
        TrtLlmBf16LoRAExperts=type("TrtLlmBf16LoRAExperts", (object,), {}),
    )
    aiter_rocm = _make_module(
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
        rocm_aiter_grouped_topk=lambda *a, **k: None,
    )

    class _StubUpstreamRouter:
        """Upstream router stand-in: replace_router_with_fl overwrites this."""

        def _compute_routing(self, *args, **kwargs):
            raise AssertionError("unpatched upstream router reached")

    router_fused = _make_module(
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router",
        FusedTopKRouter=type("FusedTopKRouter", (_StubUpstreamRouter,), {}),
        _get_padding_mask=lambda n: None,
        fused_topk=lambda **k: None,
    )
    router_bias = _make_module(
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
        FusedTopKBiasRouter=type("FusedTopKBiasRouter", (_StubUpstreamRouter,), {}),
        fused_topk_bias=lambda **k: None,
    )
    router_grouped = _make_module(
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
        GroupedTopKRouter=type("GroupedTopKRouter", (_StubUpstreamRouter,), {}),
    )
    triton_utils = _make_module(
        "vllm.triton_utils",
        tl=types.SimpleNamespace(bfloat16="bf16", float16="f16", float32="f32"),
        triton=None,
    )

    vllm.platforms = platforms
    vllm.model_executor = me
    vllm.envs = envs_mod
    me.layers = layers
    layers.fused_moe = fm
    fm.experts = experts_dir
    fm.router = router_dir
    fm.config = fm_config
    fm.runner = runner_dir
    runner_dir.moe_runner = fm_runner
    fm.unquantized_fused_moe_method = unq_method
    fm.utils = fm_utils_v
    fm.activation = fm_activation
    fm.oracle = oracle_dir
    oracle_dir.unquantized = oracle_unq
    fm.fused_moe = fm_fused_v

    for name, module in {
        "vllm": vllm,
        "vllm.platforms": platforms,
        "vllm.logger": logger_mod,
        "vllm.envs": envs_mod,
        "vllm._aiter_ops": aiter_ops,
        "vllm.model_executor": me,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.layers.fused_moe": fm,
        "vllm.model_executor.layers.fused_moe.config": fm_config,
        "vllm.model_executor.layers.fused_moe.runner": runner_dir,
        "vllm.model_executor.layers.fused_moe.runner.moe_runner": fm_runner,
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method": unq_method,
        "vllm.model_executor.layers.fused_moe.utils": fm_utils_v,
        "vllm.model_executor.layers.fused_moe.activation": fm_activation,
        "vllm.model_executor.layers.fused_moe.oracle": oracle_dir,
        "vllm.model_executor.layers.fused_moe.oracle.unquantized": oracle_unq,
        "vllm.model_executor.layers.fused_moe.modular_kernel": modular,
        "vllm.model_executor.layers.fused_moe.fused_moe": fm_fused_v,
        "vllm.model_executor.layers.fused_moe.experts": experts_dir,
        "vllm.model_executor.layers.fused_moe.experts.triton_moe": experts_triton,
        "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe": experts_trtllm,
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe": aiter_rocm,
        "vllm.model_executor.layers.fused_moe.router": router_dir,
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router": router_fused,
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router": router_bias,
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router": router_grouped,
        "vllm.triton_utils": triton_utils,
    }.items():
        sys.modules[name] = module


# Snapshot every pre-existing vllm/vllm_fl entry, install the stub graph
# when the real vllm is absent, import the module under test, then
# restore. The imports stay alive after the restore because the tested code
# lives in the vllm_fl modules themselves, which remain in sys.modules; but
# nothing of the stub graph leaks into sys.modules for later files — without
# the restore, a later file's ``import vllm`` probe would succeed against the
# stubs and files gated on HAS_VLLM would collect stale/partial modules
# (e.g. vllm.model_executor.layers without rotary_embedding).
_STUB_ROOTS = ("vllm", "vllm_fl")
_SAVED = {
    name: sys.modules.get(name, None)
    for name in list(sys.modules)
    if name.split(".")[0] in _STUB_ROOTS
}
if not _HAS_VLLM:  # pragma: no cover - Windows machines without vllm
    _install_stub_vllm_graph()
try:
    import vllm_fl.ops.fused_moe.fused_moe_utils as fmu  # noqa: E402
finally:
    if not _HAS_VLLM:  # pragma: no cover - restore only what stubs touched
        for name in [
            name
            for name in list(sys.modules)
            if name.split(".")[0] in _STUB_ROOTS and name not in _SAVED
        ]:
            sys.modules.pop(name, None)
        for name, module in _SAVED.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _fl_enum(name: str):
    """Fetch an enum member present on both the stub and the real vllm."""
    return getattr(fmu.MoEActivation, name)


def _fl_backend(name: str):
    """Fetch a backend member present on both the stub and the real vllm."""
    return getattr(fmu.UnquantizedMoeBackend, name)


class _RecordingCachedOp:
    """A stand-in for CachedOp instances recording every dispatched call.

    The modules under test call their module-level ``CachedOp`` handles as
    plain callables (``_moe_sum(input, output)``), so replacing the module
    attribute with an instance of this class preserves the call contract.
    """

    def __init__(self, impl):
        self.calls = []
        self._impl = impl

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._impl is None:
            return None
        return self._impl(*args, **kwargs)


def _replace_cached_op(monkeypatch, module, attr, impl):
    """Swap a module's CachedOp attribute for a recording stand-in."""
    recorder = _RecordingCachedOp(impl)
    monkeypatch.setattr(module, attr, recorder, raising=False)
    return recorder


def _clear_info_once_memo():
    """Reset the real vLLM logger's per-process ``info_once`` memoization.

    ``vllm.logger.info_once`` is ``lru_cache``-memoized on
    ``(logger, message)``, so once one test triggers a selection message any
    later test emitting the identical message stays silent and a ``caplog``
    assertion fails.  Clearing the cache restores observability; on the stub
    graph the helper simply does nothing.
    """
    try:
        from vllm.logger import _print_info_once
    except (ImportError, AttributeError):  # stub graph: nothing memoized
        return
    _print_info_once.cache_clear()


class _FakePlatform:
    """Configurable platform probe: only the oracle's questions are answered."""

    def __init__(
        self,
        device_type="cpu",
        vendor_name=None,
        is_tpu=False,
        is_cuda=False,
        is_rocm=False,
        is_cpu=False,
        is_xpu=False,
        is_out_of_tree=False,
        capability_family=None,
    ):
        self.device_type = device_type
        self.vendor_name = vendor_name
        self._capability_family = capability_family
        self._flags = {
            "tpu": is_tpu,
            "cuda": is_cuda,
            "rocm": is_rocm,
            "cpu": is_cpu,
            "xpu": is_xpu,
            "out_of_tree": is_out_of_tree,
        }

    def __getattr__(self, name):
        # The flags dict is keyed without the ``is_`` prefix (``rocm``), so
        # ``is_rocm`` resolves through ``name[3:]``; anything else the oracle
        # does not ask about is a hard AttributeError.
        if name.startswith("is_") and name[3:] in self._flags:
            return lambda: self._flags[name[3:]]
        if name == "is_device_capability_family":

            def _is_family(family):
                return self._capability_family == family

            return _is_family
        raise AttributeError(name)


def _fake_moe_config(
    *,
    moe_backend="auto",
    activation_name="SILU",
    dp_size=1,
    use_ep=False,
    use_batched_activation_format=False,
    is_lora_enabled=False,
):
    """A config object answering only the oracle's attribute probes."""
    return types.SimpleNamespace(
        moe_parallel_config=types.SimpleNamespace(
            dp_size=dp_size,
            use_ep=use_ep,
            use_batched_activation_format=use_batched_activation_format,
        ),
        moe_backend=moe_backend,
        activation=_fl_enum(activation_name),
        is_lora_enabled=is_lora_enabled,
    )


class _Kernel:
    """A stand-in kernel class for backend_to_kernel_cls, scripted per test."""

    def __init__(self, supported=True, reason=None):
        self.supported = supported
        self.reason = reason

    @staticmethod
    def is_supported_config(cls, config, weight_key, activation_key, fmt):
        return (cls.supported, cls.reason)


@pytest.fixture
def clean_fl_env(monkeypatch):
    """Clear every ``VLLM_FL_*`` env var for the test (restored on teardown).

    The supplementary conftest's helper of the same name: the selector gate
    below pins the FlagGems policy off, and a stray ``VLLM_FL_*`` variable
    set by an earlier test would flip the gate.
    """
    import os

    fl_vars = [k for k in os.environ if k.startswith("VLLM_FL_")]
    for key in fl_vars:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch.setenv


@pytest.fixture(autouse=True)
def _pin_selector_env(monkeypatch, clean_fl_env):
    """Keep FlagGems policy off around every test (selector gate baseline)."""
    monkeypatch.setenv("USE_FLAGGEMS", "0")
    yield


@pytest.fixture(autouse=True)
def _real_vllm_dtype_namespace(monkeypatch):
    """Give the real vllm's Triton ``tl`` placeholder the float dtypes.

    On a real-vllm CPU host ``vllm.triton_utils.tl`` is vLLM's
    ``TritonLanguagePlaceholder``, which only defines index types and math
    helpers — no ``float16``/``bfloat16``/``float32``.  The FL modules read
    ``tl.bfloat16`` etc. only to pick the kernel's compute type, a value that
    is handed straight to the (stubbed) kernel boundary, so distinct string
    sentinels preserve the code under test while keeping the attribute reads
    working.  With the stub graph (Windows) ``tl`` already carries the same
    sentinels, making the patch a no-op there.  ``fmu.tl`` (and the same
    object the fused_moe module binds) is patched here so the attribute reads
    in this file's tests keep working; the values are set (not replaced) on
    the shared placeholder module and restored afterwards.
    """
    for module in (fmu.tl,):
        for name in ("float16", "bfloat16", "float32"):
            monkeypatch.setattr(module, name, f"<{name}-sentinel>", raising=False)


# ---------------------------------------------------------------------------
# FlashinferMoeBackend version shim (bound on both platforms: the real
# flashinfer_utils import fails on vLLM 0.32 and on the stub graph alike)
# ---------------------------------------------------------------------------


class TestFlashinferMoeBackendShim:
    """The vLLM 0.28-compatibility FlashinferMoeBackend value shim."""

    def test_init_normalizes_value(self):
        """The constructor lowercases and strips the value."""
        shim = fmu.FlashinferMoeBackend("  CUTLASS ")
        assert shim.value == "cutlass"
        assert fmu.FlashinferMoeBackend("TrtLlm").value == "trtllm"

    def test_class_constants(self):
        """The two shipped constants are their lowercase names."""
        assert fmu.FlashinferMoeBackend.CUTLASS == "cutlass"
        assert fmu.FlashinferMoeBackend.TENSORRT_LLM == "trtllm"

    def test_eq_matches_value_and_substring(self):
        """Equality holds for equal values and for a member of this value."""
        shim = fmu.FlashinferMoeBackend("trtllm")
        assert shim == "trtllm"
        assert shim == fmu.FlashinferMoeBackend("TRTLLM")
        # A shorter string inside the value also matches ("trt" in "trtllm").
        assert shim == "trt"
        assert shim != "cutlass"
        assert shim != "trtllm-v2"

    def test_eq_reads_other_value_attribute(self):
        """Equality falls back to the other object's .value attribute."""
        shim = fmu.FlashinferMoeBackend("cutlass")
        # An object whose .value normalizes to the same string is equal...
        same = types.SimpleNamespace(value="  CUTLASS ")
        assert shim == same
        # ...and one whose normalized value is a substring of this shim's
        # value is equal too (the substring direction the selector relies on).
        narrower = types.SimpleNamespace(value="cut")
        assert shim == narrower
        # An unrelated value attribute is not.
        unrelated = types.SimpleNamespace(value="trtllm")
        assert shim != unrelated

    def test_hash_and_str_roundtrip(self):
        """Hash and str are the normalized value."""
        shim = fmu.FlashinferMoeBackend(" TrtLlm ")
        assert hash(shim) == hash("trtllm")
        assert str(shim) == "trtllm"
        assert {shim: 1}["trtllm"] == 1


# ---------------------------------------------------------------------------
# _get_priority_backends
# ---------------------------------------------------------------------------


class TestGetPriorityBackends:
    """Platform-conditioned priority ordering of unquantized MoE backends."""

    def test_rocm_lists_aiter_first(self, monkeypatch):
        """ROCm puts AITER ahead of the two Triton variants."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_rocm=True))
        backends = fmu._get_priority_backends(_fake_moe_config())
        assert [b.name for b in backends] == ["AITER", "TRITON", "BATCHED_TRITON"]

    def test_cuda_default_flashinfer_first(self, monkeypatch):
        """Default CUDA order: FlashInfer variants before the Triton pair."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_cuda=True))
        backends = fmu._get_priority_backends(_fake_moe_config())
        assert [b.name for b in backends] == [
            "FLASHINFER_TRTLLM",
            "FLASHINFER_CUTLASS",
            "TRITON",
            "BATCHED_TRITON",
        ]

    def test_hopper_demotes_flashinfer(self, monkeypatch):
        """Device capability family 90 moves both FlashInfer variants to the back."""
        monkeypatch.setattr(
            fmu,
            "current_platform",
            _FakePlatform(is_cuda=True, capability_family=90),
        )
        backends = fmu._get_priority_backends(_fake_moe_config())
        assert [b.name for b in backends] == [
            "TRITON",
            "BATCHED_TRITON",
            "FLASHINFER_TRTLLM",
            "FLASHINFER_CUTLASS",
        ]

    def test_dp_greater_than_one_demotes_flashinfer_cutlass(self, monkeypatch):
        """DP>1 moves only FLASHINFER_CUTLASS behind the Triton variants."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_cuda=True))
        backends = fmu._get_priority_backends(_fake_moe_config(dp_size=2))
        assert [b.name for b in backends] == [
            "FLASHINFER_TRTLLM",
            "TRITON",
            "BATCHED_TRITON",
            "FLASHINFER_CUTLASS",
        ]

    def test_swigluoai_demotes_both_flashinfer(self, monkeypatch):
        """SWIGLUOAI activation demotes both FlashInfer variants."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_cuda=True))
        backends = fmu._get_priority_backends(
            _fake_moe_config(activation_name="SWIGLUOAI")
        )
        assert [b.name for b in backends] == [
            "TRITON",
            "BATCHED_TRITON",
            "FLASHINFER_TRTLLM",
            "FLASHINFER_CUTLASS",
        ]

    def test_xpu_platform_lists_xpu_only(self, monkeypatch):
        """XPU platforms offer only the XPU backend."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_xpu=True))
        backends = fmu._get_priority_backends(_fake_moe_config())
        assert [b.name for b in backends] == ["XPU"]

    def test_cpu_platform_lists_cpu_only(self, monkeypatch):
        """CPU platforms offer only the CPU backend."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_cpu=True))
        backends = fmu._get_priority_backends(_fake_moe_config())
        assert [b.name for b in backends] == ["CPU"]

    def test_iluvatar_vendor_lists_triton_variants(self, monkeypatch):
        """The iluvatar vendor branch keeps the Triton variants reachable."""
        monkeypatch.setattr(
            fmu,
            "current_platform",
            _FakePlatform(device_type="cuda", vendor_name="iluvatar"),
        )
        backends = fmu._get_priority_backends(_fake_moe_config())
        assert [b.name for b in backends] == ["TRITON", "BATCHED_TRITON"]


# ---------------------------------------------------------------------------
# select_unquantized_moe_backend_oot
# ---------------------------------------------------------------------------


class TestSelectUnquantizedMoeBackendOot:
    """The OOT backend-selection oracle driven with fake configs and probes."""

    @pytest.fixture(autouse=True)
    def _cpu_platform(self, monkeypatch):
        """Run the selector with a neutral CPU platform by default."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_cpu=True))

    def test_tpu_short_circuits(self, monkeypatch):
        """TPU platforms return (TPU, None) before any config probe."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_tpu=True))
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "TPU"
        assert experts is None

    def test_npu_rejects_parallel_configs(self, monkeypatch):
        """NPU raises NotImplementedError for EP / DP / LoRA / batched configs."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="npu"))
        for cfg in (
            _fake_moe_config(dp_size=2),
            _fake_moe_config(use_ep=True),
            _fake_moe_config(use_batched_activation_format=True),
            _fake_moe_config(is_lora_enabled=True),
        ):
            with pytest.raises(NotImplementedError, match="non-EP, DP1"):
                fmu.select_unquantized_moe_backend_oot(cfg)

    def test_npu_rejects_unsupported_explicit_backend(self, monkeypatch):
        """NPU only accepts moe_backend in ('auto', 'triton')."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="npu"))
        with pytest.raises(NotImplementedError, match="Unsupported explicit"):
            fmu.select_unquantized_moe_backend_oot(
                _fake_moe_config(moe_backend="flashinfer_cutlass")
            )

    def test_npu_returns_fl_triton_experts(self, monkeypatch):
        """NPU with a supported config selects TRITON and TritonExpertsFL."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="npu"))
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "TRITON"
        assert experts is fmu.TritonExpertsFL

    def test_fl_triton_experts_policy_short_circuit(self, monkeypatch):
        """OOT accelerators with FlagGems enabled select TritonExpertsFL."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_out_of_tree=True))
        monkeypatch.setattr(fmu, "use_flaggems", lambda: True)
        monkeypatch.setattr(fmu, "get_oot_blacklist", lambda: None)
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "TRITON"
        assert experts is fmu.TritonExpertsFL

    def test_lora_with_trtllm_support_selects_trtllm_experts(self, monkeypatch):
        """BF16 LoRA support routes to FLASHINFER_TRTLLM with the LoRA experts."""
        # select_unquantized_moe_backend_oot imports the experts module
        # lazily at call time; on stub hosts the stub graph was removed after
        # the file-scope import, so the lazy target is re-seeded here with
        # monkeypatch (auto-restored) instead of a permanent install.
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe",
            _make_module(
                "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe",
                TrtLlmBf16LoRAExperts=type("TrtLlmBf16LoRAExperts", (object,), {}),
            ),
        )
        monkeypatch.setattr(fmu, "_trtllm_bf16_lora_supported", lambda cfg: True)
        backend, experts = fmu.select_unquantized_moe_backend_oot(
            _fake_moe_config(is_lora_enabled=True)
        )
        assert backend.name == "FLASHINFER_TRTLLM"
        assert experts is not None
        assert experts.__name__ == "TrtLlmBf16LoRAExperts"

    def test_lora_without_trtllm_support_selects_triton(self, monkeypatch):
        """Without BF16 LoRA support, LoRA falls back to the TRITON backend."""
        monkeypatch.setattr(fmu, "_trtllm_bf16_lora_supported", lambda cfg: False)
        triton_cls = _Kernel()
        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [triton_cls])
        backend, experts = fmu.select_unquantized_moe_backend_oot(
            _fake_moe_config(is_lora_enabled=True)
        )
        assert backend.name == "TRITON"
        assert experts is triton_cls

    def test_explicit_runner_backend_maps_and_selects(self, monkeypatch):
        """A runner backend other than auto/humming is mapped and returned."""
        requested = _fl_backend("XPU")
        monkeypatch.setattr(fmu, "map_unquantized_backend", lambda name: requested)
        xpu_cls = _Kernel()
        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [xpu_cls])
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_xpu=True))
        backend, experts = fmu.select_unquantized_moe_backend_oot(
            _fake_moe_config(moe_backend="explicit_choice")
        )
        assert backend is requested
        assert experts is xpu_cls

    def test_explicit_runner_backend_unsupported_raises_value_error(self, monkeypatch):
        """An explicit backend with no supporting kernel raises ValueError."""
        requested = _fl_backend("XPU")
        monkeypatch.setattr(fmu, "map_unquantized_backend", lambda name: requested)
        monkeypatch.setattr(
            fmu,
            "backend_to_kernel_cls",
            lambda backend: [_Kernel(False, "kernel does not support xpu here")],
        )
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_xpu=True))
        with pytest.raises(ValueError, match="does not support"):
            fmu.select_unquantized_moe_backend_oot(
                _fake_moe_config(moe_backend="explicit_choice")
            )

    def test_explicit_triton_upgrades_to_batched_format(self, monkeypatch):
        """TRITON under batched activation format upgrades to BATCHED_TRITON."""
        requested = _fl_backend("TRITON")
        monkeypatch.setattr(fmu, "map_unquantized_backend", lambda name: requested)
        seen_formats = []

        class _FmtKernel:
            @staticmethod
            def is_supported_config(cls, config, weight_key, activation_key, fmt):
                seen_formats.append(fmt)
                return (True, None)

        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [_FmtKernel])
        backend, _experts = fmu.select_unquantized_moe_backend_oot(
            _fake_moe_config(moe_backend="triton", use_batched_activation_format=True)
        )
        assert backend.name == "BATCHED_TRITON"
        assert seen_formats and seen_formats[0].name == "BatchedExperts"

    def test_explicit_backend_logs_selection_message(self, monkeypatch, caplog):
        """A supported explicit backend logs the selection message once."""
        requested = _fl_backend("XPU")
        monkeypatch.setattr(fmu, "map_unquantized_backend", lambda name: requested)
        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [_Kernel()])
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_xpu=True))
        # The real vLLM logger's info_once memoizes on (logger, message) for
        # the whole process, so an earlier test that selected the same XPU
        # backend already consumed this exact message. Clear the memo cache
        # (absent on the stub graph) so this call is observable again.
        _clear_info_once_memo()
        with caplog.at_level("INFO"):
            fmu.select_unquantized_moe_backend_oot(
                _fake_moe_config(moe_backend="explicit_choice")
            )
        assert any("out of potential backends" in rec.message for rec in caplog.records)

    def test_auto_selects_first_supported_priority_backend(self, monkeypatch):
        """auto walks the priority list and returns the first supported pair."""
        cpu_cls = _Kernel()
        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [cpu_cls])
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "CPU"
        assert experts is cpu_cls

    def test_auto_skips_unsupported_and_selects_next(self, monkeypatch):
        """auto continues past unsupported kernel classes to the next one."""
        classes = [_Kernel(False, "not on this device"), _Kernel()]

        def fake_b2k(backend):
            return classes

        monkeypatch.setattr(fmu, "backend_to_kernel_cls", fake_b2k)
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "CPU"
        assert experts is classes[1]

    def test_auto_raises_when_no_backend_supports(self, monkeypatch):
        """auto with zero supported backends raises NotImplementedError."""
        monkeypatch.setattr(
            fmu,
            "backend_to_kernel_cls",
            lambda backend: [_Kernel(False, "reason")],
        )
        with pytest.raises(NotImplementedError, match="No Unquantized MoE backend"):
            fmu.select_unquantized_moe_backend_oot(_fake_moe_config())

    def test_auto_raises_without_any_backend_list(self, monkeypatch):
        """A platform matching no branch leaves the list unset and fails."""
        monkeypatch.setattr(
            fmu,
            "current_platform",
            _FakePlatform(device_type="musa", vendor_name="mthreads"),
        )
        with pytest.raises((NotImplementedError, UnboundLocalError, AttributeError)):
            fmu.select_unquantized_moe_backend_oot(_fake_moe_config())

    def test_aiter_env_forces_aiter_selection(self, monkeypatch):
        """VLLM_ROCM_USE_AITER(_MOE) fully on selects AITER directly."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_rocm=True))
        monkeypatch.setattr(fmu.envs, "is_set", lambda name: True)
        monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER", True)
        monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER_MOE", True)
        monkeypatch.setattr(fmu.rocm_aiter_ops, "is_rdna_aiter_enabled", lambda: False)
        aiter_cls = _Kernel()
        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [aiter_cls])
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "AITER"
        assert experts is aiter_cls

    def test_aiter_env_rdna_drops_aiter_from_list(self, monkeypatch):
        """RDNA-enabled builds drop AITER and continue down the priority list."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_rocm=True))
        monkeypatch.setattr(fmu.envs, "is_set", lambda name: True)
        monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER", True)
        monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER_MOE", True)
        monkeypatch.setattr(fmu.rocm_aiter_ops, "is_rdna_aiter_enabled", lambda: True)
        triton_cls = _Kernel()
        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [triton_cls])
        backend, _experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "TRITON"  # AITER was removed from the list

    def test_aiter_env_partial_flag_drops_aiter(self, monkeypatch):
        """AITER env set but a flag off falls through without AITER."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_rocm=True))
        monkeypatch.setattr(fmu.envs, "is_set", lambda name: True)
        monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER", True)
        monkeypatch.setattr(fmu.envs, "VLLM_ROCM_USE_AITER_MOE", False)
        monkeypatch.setattr(fmu.rocm_aiter_ops, "is_rdna_aiter_enabled", lambda: False)
        triton_cls = _Kernel()
        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [triton_cls])
        backend, _experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "TRITON"


# ---------------------------------------------------------------------------
# _prepare_expert_assignment
# ---------------------------------------------------------------------------


class TestPrepareExpertAssignment:
    """The naive-block fast path versus the moe_align_block_size dispatch."""

    def test_small_problem_without_map_uses_naive_path(self, monkeypatch):
        """tokens*topk*4 <= experts with no map skips block alignment."""
        called = []
        monkeypatch.setattr(
            fmu,
            "_moe_align_block_size",
            lambda *args, **kwargs: (called.append(1), (None, None, None))[1],
        )
        topk_ids = torch.tensor([[0, 1], [1, 2]])
        sorted_ids, expert_ids, num_post = fmu._prepare_expert_assignment(
            topk_ids,
            {"BLOCK_SIZE_M": 16},
            num_tokens=2,
            top_k_num=2,
            global_num_experts=64,
            expert_map=None,
        )
        assert called == []  # naive path: the dispatcher was never consulted
        assert sorted_ids is None
        torch.testing.assert_close(expert_ids, topk_ids.view(-1))
        torch.testing.assert_close(
            num_post, torch.tensor([topk_ids.numel() * 16], dtype=torch.int32)
        )

    def test_large_problem_falls_back_to_align(self, monkeypatch):
        """tokens*topk*4 > experts dispatches to moe_align_block_size."""
        captured = {}

        def fake_align(topk_ids, block_size, num_experts, expert_map, **kwargs):
            captured.update(
                block_size=block_size,
                num_experts=num_experts,
                expert_map=expert_map,
                **kwargs,
            )
            return ("sorted", "expert_ids", "num_post")

        monkeypatch.setattr(fmu, "_moe_align_block_size", fake_align)
        topk_ids = torch.tensor([[0, 1], [1, 2]])
        result = fmu._prepare_expert_assignment(
            topk_ids,
            {"BLOCK_SIZE_M": 16},
            num_tokens=2,
            top_k_num=2,
            global_num_experts=4,
            expert_map=None,
            ignore_invalid_experts=True,
        )
        assert result == ("sorted", "expert_ids", "num_post")
        assert captured["block_size"] == 16
        assert captured["num_experts"] == 4
        assert captured["ignore_invalid_experts"] is True

    def test_expert_map_forces_align(self, monkeypatch):
        """An expert map present always takes the aligned path."""
        called = []
        monkeypatch.setattr(
            fmu,
            "_moe_align_block_size",
            lambda *args, **kwargs: (called.append(1), (None, None, None))[1],
        )
        fmu._prepare_expert_assignment(
            torch.tensor([[0]]),
            {"BLOCK_SIZE_M": 16},
            num_tokens=1,
            top_k_num=1,
            global_num_experts=1024,
            expert_map=torch.tensor([0]),
        )
        assert called == [1]

    def test_int8_with_block_shape_forces_align(self, monkeypatch):
        """int8/int4 with block shapes never take the naive path."""
        called = []
        monkeypatch.setattr(
            fmu,
            "_moe_align_block_size",
            lambda *args, **kwargs: (called.append(1), (None, None, None))[1],
        )
        fmu._prepare_expert_assignment(
            torch.tensor([[0]]),
            {"BLOCK_SIZE_M": 16},
            num_tokens=1,
            top_k_num=1,
            global_num_experts=1024,
            expert_map=None,
            use_int8_w8a16=True,
            block_shape=[128, 128],
        )
        assert called == [1]

    def test_block_shape_second_dim_zero_keeps_naive(self, monkeypatch):
        """block_shape[1] == 0 disables the quantized guard (naive still taken)."""
        called = []
        monkeypatch.setattr(
            fmu,
            "_moe_align_block_size",
            lambda *args, **kwargs: (called.append(1), (None, None, None))[1],
        )
        topk_ids = torch.tensor([[0, 1]])
        sorted_ids, expert_ids, _num_post = fmu._prepare_expert_assignment(
            topk_ids,
            {"BLOCK_SIZE_M": 16},
            num_tokens=1,
            top_k_num=2,
            global_num_experts=64,
            expert_map=None,
            use_int8_w8a16=True,
            block_shape=[128, 0],
        )
        assert called == []
        assert sorted_ids is None
        torch.testing.assert_close(expert_ids, topk_ids.view(-1))


# ---------------------------------------------------------------------------
# TritonExpertsFL.workspace_shapes / moe_sum / apply
# ---------------------------------------------------------------------------


class TestTritonExpertsFLWorkspace:
    """TritonExpertsFL workspace and reduction dispatch boundaries."""

    def _make_experts(self, monkeypatch, base_shapes, max_num_tokens=None):
        """Build a TritonExpertsFL instance around a scripted base class."""
        captured = {}

        # TritonExpertsFL binds its base at class-creation time, so the base
        # implementation is swapped on the actual base class (the module-level
        # TritonExperts the FL class subclasses), not on the module attribute.
        base_cls = fmu.TritonExpertsFL.__bases__[0]

        def _fake_init(self, moe_config, quant_config):
            self.moe_config = moe_config
            self.quant_config = quant_config

        def _fake_workspace_shapes(self, *args):
            captured["base_args"] = args
            return base_shapes

        monkeypatch.setattr(base_cls, "__init__", _fake_init)
        monkeypatch.setattr(base_cls, "workspace_shapes", _fake_workspace_shapes)
        moe_config = types.SimpleNamespace(max_num_tokens=max_num_tokens)
        experts = fmu.TritonExpertsFL(moe_config, None)
        return experts, captured

    def test_workspace_shapes_passthrough_on_non_npu(self, monkeypatch):
        """Non-NPU platforms return the base shapes untouched."""
        shapes = ((8, 16, 8), (8, 4, 16), (8, 8))
        experts, _captured = self._make_experts(monkeypatch, shapes)
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="cpu"))
        assert experts.workspace_shapes(8, 16, 8, 2, 4, 4, None, None) == shapes

    def test_workspace_shapes_npu_grows_to_max_tokens(self, monkeypatch):
        """NPU pads the workspace token dim to moe_config.max_num_tokens."""
        shapes = ((8, 16, 8), (8, 4, 16), (8, 8))
        experts, captured = self._make_experts(monkeypatch, shapes, max_num_tokens=32)
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="npu"))
        w13, w2, out = experts.workspace_shapes(8, 16, 8, 2, 4, 4, None, None)
        assert w13 == (32, 16, 8)
        assert w2 == (32, 4, 16)
        assert out == (8, 8)
        # The base implementation got the arguments unchanged.
        assert captured["base_args"] == (8, 16, 8, 2, 4, 4, None, None)

    def test_workspace_shapes_npu_keeps_larger_m(self, monkeypatch):
        """NPU keeps M when it already exceeds max_num_tokens."""
        shapes = ((48, 16, 8), (48, 4, 16), (48, 8))
        experts, _captured = self._make_experts(monkeypatch, shapes, max_num_tokens=32)
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="npu"))
        w13, w2, _out = experts.workspace_shapes(48, 16, 8, 2, 4, 4, None, None)
        assert w13 == (48, 16, 8)
        assert w2 == (48, 4, 16)

    def test_moe_sum_forwards_to_cached_op(self, monkeypatch):
        """moe_sum hands both tensors to the moe_sum CachedOp unchanged."""

        def impl(inp, out):
            torch.sum(inp, dim=1, out=out)

        recorder = _replace_cached_op(monkeypatch, fmu, "_moe_sum", impl)
        experts = object.__new__(fmu.TritonExpertsFL)
        inp = torch.ones(2, 2, 3)
        out = torch.zeros(2, 3)
        experts.moe_sum(inp, out)
        assert recorder.calls == [((inp, out), {})]
        # Independent oracle: sum over the expert dimension.
        torch.testing.assert_close(out, torch.full((2, 3), 2.0))


class TestTritonExpertsFLApply:
    """TritonExpertsFL.apply routing across its vendor dispatch boundaries."""

    E, N, K, TOPK = 2, 8, 4, 2

    def _apply_args(self, num_tokens=3):
        hidden = torch.randn(num_tokens, self.K)
        w1 = torch.randn(self.E, 2 * self.N, self.K)
        w2 = torch.randn(self.E, self.K, self.N)
        tw = torch.rand(num_tokens, self.TOPK)
        ti = torch.randint(0, self.E, (num_tokens, self.TOPK))
        out = torch.zeros_like(hidden)
        return out, hidden, w1, w2, tw, ti

    def _make_experts(self, monkeypatch):
        """Build a TritonExpertsFL whose base class records nothing."""
        captured = {}

        def _fake_init(self, moe_config, quant_config):
            self.moe_config = moe_config
            self.quant_config = quant_config
            captured["init"] = True

        # TritonExpertsFL binds its base at class-creation time, so the base
        # constructor is swapped on the actual base class (the real upstream
        # TritonExperts on a vllm host, the stub class otherwise), never on
        # the module attribute the FL module also exposes.
        base_cls = fmu.TritonExpertsFL.__bases__[0]
        monkeypatch.setattr(base_cls, "__init__", _fake_init)
        # On the real upstream class every quantization parameter below is a
        # read-only property delegating to quant_config, so the values live on
        # the config namespace; the stub's plain class reads them as plain
        # instance attributes set right after construction. Both routes end
        # with apply() observing the same values.
        quant_config = types.SimpleNamespace(
            quant_dtype=None,
            weight_quant_dtype=None,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            config_name=lambda dtype: "float32",
            w1_bias=None,
            w2_bias=None,
            w1_scale=None,
            w2_scale=None,
            block_shape=None,
            per_act_token_quant=False,
        )
        experts = fmu.TritonExpertsFL(types.SimpleNamespace(), quant_config)
        experts._lora_context = None
        experts.activation_config = types.SimpleNamespace(clamp_limit=None)
        experts._hash_indices_table = None
        experts.quantization_emulation = None
        # Property-backed names only apply where the class has no property
        # for them (the real upstream class derives them from quant_config,
        # which already carries the same values).
        for name, value in (
            ("w1_bias", None),
            ("w2_bias", None),
            ("w1_scale", None),
            ("w2_scale", None),
            ("block_shape", None),
            ("per_act_token_quant", False),
            ("quant_dtype", None),
        ):
            for klass in type(experts).__mro__:
                if name in vars(klass):
                    if not isinstance(vars(klass)[name], property):
                        setattr(experts, name, value)
                    break
            else:
                setattr(experts, name, value)
        return experts

    def _patch_triton_pipeline(self, monkeypatch, experts):
        """Stub the generic Triton pipeline's boundaries on the module."""

        def fake_align(ids, bs, ne, em, **kw):
            return (
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                torch.tensor([8], dtype=torch.int32),
            )

        _replace_cached_op(monkeypatch, fmu, "_moe_align_block_size", fake_align)
        monkeypatch.setattr(
            fmu,
            "_invoke_fused_moe_triton_kernel",
            lambda *args, **kwargs: None,
            raising=False,
        )
        monkeypatch.setattr(
            fmu,
            "try_get_optimal_moe_config",
            lambda *args, **kwargs: {"BLOCK_SIZE_M": 16},
            raising=False,
        )

        def fake_resize_cache(cache, shape):
            numel = 1
            for dim in shape:
                numel *= dim
            if cache.numel() == numel:
                return cache.view(*shape)
            if cache.numel() > numel:
                return cache[:numel].view(*shape)
            return torch.empty(shape)

        monkeypatch.setattr(fmu, "_resize_cache", fake_resize_cache, raising=False)
        experts.moe_problem_size = lambda a1, w1, w2, topk_ids: (
            self.E,
            a1.size(0),
            w1.size(1),
            a1.size(1),
            topk_ids.size(1),
        )
        experts.adjust_N_for_activation = staticmethod(lambda N, activation: N // 2)

    def test_apply_npu_routes_to_grouped_moe_experts_op(self, monkeypatch):
        """NPU tensors route through the grouped_moe_experts dispatch op."""

        class _NpuTensor:
            """CPU tensor double reporting an NPU device.

            ``apply``'s NPU branch only reads ``device.type`` before handing
            the tensors to the dispatch op, so a lightweight double carrying
            the real tensors is enough to drive that branch on a CPU host.
            """

            def __init__(self, t):
                self._t = t

            @property
            def device(self):
                return types.SimpleNamespace(type="npu")

        out, hidden, w1, w2, tw, ti = self._apply_args()
        experts = self._make_experts(monkeypatch)
        recorded = {}

        def fake_call_op(name, *args, **kwargs):
            recorded["name"] = name
            recorded["kwargs"] = kwargs
            return hidden * 2.0

        monkeypatch.setattr(fmu, "call_op", fake_call_op)
        experts.apply(
            out,
            _NpuTensor(hidden),
            _NpuTensor(w1),
            _NpuTensor(w2),
            tw,
            ti,
            _fl_enum("SILU"),
            self.E,
            None,
            None,
            None,
            torch.zeros(1),
            torch.zeros(1),
            None,
            False,
        )
        assert recorded["name"] == "grouped_moe_experts"
        assert recorded["kwargs"]["activation"] == "silu"
        assert recorded["kwargs"]["apply_router_weight_on_input"] is False
        # The recorded kwarg values are the doubles themselves; the output
        # copy came from the scripted dispatch result.
        torch.testing.assert_close(out, hidden * 2.0)

    def test_apply_kunlunxin_routes_to_fused_experts_op(self, monkeypatch):
        """The kunlunxin vendor resolves the fused_experts op directly."""
        out, hidden, w1, w2, tw, ti = self._apply_args()
        experts = self._make_experts(monkeypatch)
        monkeypatch.setattr(
            fmu,
            "current_platform",
            _FakePlatform(device_type="cuda", vendor_name="kunlunxin"),
        )
        resolved = {}

        def fake_resolve_op(name):
            assert name == "fused_experts"

            def _impl(*args, **kwargs):
                resolved["kwargs"] = kwargs
                return hidden + 1.0

            return _impl

        monkeypatch.setattr(fmu, "resolve_op", fake_resolve_op)
        experts.apply(
            out,
            hidden,
            w1,
            w2,
            tw,
            ti,
            _fl_enum("GELU"),
            self.E,
            None,
            None,
            None,
            torch.zeros(1),
            torch.zeros(1),
            None,
            True,
        )
        assert resolved["kwargs"]["activation"] == "gelu"
        assert resolved["kwargs"]["apply_router_weight_on_input"] is True
        torch.testing.assert_close(out, hidden + 1.0)

    def test_apply_cuda_fast_path_uses_flaggems_vllm(self, monkeypatch):
        """NVIDIA without LoRA delegates to flaggems_vllm.fused_experts_impl."""
        out, hidden, w1, w2, tw, ti = self._apply_args()
        experts = self._make_experts(monkeypatch)
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_cuda=True))
        # flaggems_vllm is a vendor SDK boundary: stub it in sys.modules.
        fake_pkg = types.ModuleType("flaggems_vllm")
        calls = {}

        def fake_impl(*args, **kwargs):
            calls.update(kwargs)
            return hidden * 3.0

        fake_pkg.fused_experts_impl = fake_impl
        monkeypatch.setitem(sys.modules, "flaggems_vllm", fake_pkg)
        experts.apply(
            out,
            hidden,
            w1,
            w2,
            tw,
            ti,
            _fl_enum("SILU"),
            self.E,
            None,
            None,
            None,
            torch.zeros(1),
            torch.zeros(1),
            None,
            False,
        )
        assert calls["activation"] == "silu"
        torch.testing.assert_close(out, hidden * 3.0)

    def test_apply_triton_pipeline_dispatch_contracts(self, monkeypatch):
        """The generic Triton pipeline hands both GEMMs to the kernel op."""
        out, hidden, w1, w2, tw, ti = self._apply_args()
        experts = self._make_experts(monkeypatch)
        # A vendor platform that is neither npu, kunlunxin, nor cuda.
        monkeypatch.setattr(
            fmu,
            "current_platform",
            _FakePlatform(device_type="cuda", vendor_name="iluvatar"),
        )
        kernels = []

        def fake_invoke(*args, **kwargs):
            # Positional contract: (A, B, C, A_scale, B_scale, topk_weights,
            # sorted_ids, expert_ids, num_post_pad, mul_routed, top_k, cfg)
            kernels.append(
                {"B_shape": tuple(args[1].shape), "mul": args[9], "top_k": args[10]}
            )

        self._patch_triton_pipeline(monkeypatch, experts)
        monkeypatch.setattr(
            fmu, "_invoke_fused_moe_triton_kernel", fake_invoke, raising=False
        )
        sum_recorder = _replace_cached_op(
            monkeypatch, fmu, "_moe_sum", lambda i, o: torch.sum(i, dim=1, out=o)
        )
        monkeypatch.setattr(
            fmu, "apply_moe_activation", lambda *a, **k: None, raising=False
        )
        experts.apply(
            out,
            hidden,
            w1,
            w2,
            tw,
            ti,
            _fl_enum("SILU"),
            self.E,
            None,
            None,
            None,
            torch.zeros(4096),
            torch.zeros(4096),
            None,
            False,
        )
        # The first GEMM never multiplies routed weights; the second does.
        assert len(kernels) == 2
        assert kernels[0]["B_shape"] == (self.E, 2 * self.N, self.K)
        assert kernels[0]["mul"] is False
        assert kernels[1]["B_shape"] == (self.E, self.K, self.N)
        assert kernels[1]["mul"] is True
        assert kernels[1]["top_k"] == 1
        assert sum_recorder.calls

    def test_apply_triton_pipeline_lora_step_injection(self, monkeypatch):
        """A LoRA context inserts w13/w2 adapter calls between the stages."""
        out, hidden, w1, w2, tw, ti = self._apply_args()
        experts = self._make_experts(monkeypatch)
        monkeypatch.setattr(
            fmu,
            "current_platform",
            _FakePlatform(device_type="cuda", vendor_name="iluvatar"),
        )
        self._patch_triton_pipeline(monkeypatch, experts)
        _replace_cached_op(monkeypatch, fmu, "_moe_sum", lambda i, o: None)
        monkeypatch.setattr(
            fmu, "apply_moe_activation", lambda *a, **k: None, raising=False
        )
        order = []
        lora_ctx = object()
        experts._lora_context = lora_ctx

        def fake_w13(ctx, **kwargs):
            order.append("w13")
            return (None, None, None, None)

        def fake_w2(ctx, **kwargs):
            order.append("w2")
            return None

        experts.apply_w13_lora = fake_w13
        experts.apply_w2_lora = fake_w2
        experts.apply(
            out,
            hidden,
            w1,
            w2,
            tw,
            ti,
            _fl_enum("SILU"),
            self.E,
            None,
            None,
            None,
            torch.zeros(4096),
            torch.zeros(4096),
            None,
            False,
        )
        assert order == ["w13", "w2"]

# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for fused MoE backend selection, routing, and experts.

Three modules previously uncovered by the core suite:

* ``vllm_fl.ops.fused_moe.fused_moe_utils`` -- the OOT backend-selection
  oracle (``select_unquantized_moe_backend_oot``, ``_get_priority_backends``,
  ``_move_to_back``, the log closures and ``_return_or_raise``),
  ``_prepare_expert_assignment``'s naive-block fast path, the
  ``FlashinferMoeBackend`` version shim, and ``TritonExpertsFL``'s
  ``apply``/``moe_sum``/``workspace_shapes`` dispatch boundaries.
* ``vllm_fl.ops.fused_moe.router`` -- ``_sqrtsoftplus_topk`` and
  ``_fl_grouped_topk`` numeric math (checked against independent plain-Python
  oracles), ``_has_native_moe_topk``, both FL router ``_compute_routing``
  methods, and ``replace_router_with_fl``.
* ``vllm_fl.ops.fused_moe.fused_moe`` -- ``fused_experts_impl``'s constraint
  assertions, activation normalization, and the boundary calls it hands to the
  dispatch system.

The modules import ``vllm`` at module level.  On machines without the real
package a stub ``vllm`` graph is installed first (external boundary only);
with the real ``vllm`` present the same modules import against it.  Either
way, every platform probe, config lookup, kernel class, and dispatched op is
swapped at test time via ``monkeypatch`` on module attributes, so the code
under test always runs its own logic.  Numeric expectations are computed
independently with plain-Python softmax / softplus arithmetic; where logits
tie exactly only the contract -- distinct indices sharing the tied
probability -- is asserted, never the order (same contract as
``TestTopkSoftmax`` in ``test_reference_backend.py``).
"""

import math
import sys
import types
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.supplementary


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
# when the real vllm is absent, import the three modules under test, then
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
    import vllm_fl.ops.fused_moe.fused_moe as fl_fused  # noqa: E402
    import vllm_fl.ops.fused_moe.fused_moe_utils as fmu  # noqa: E402
    import vllm_fl.ops.fused_moe.router as fl_router  # noqa: E402
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
    sentinels, making the patch a no-op there.  Both ``fl_fused.tl`` and
    ``fmu.tl`` reference the same module object, and the assertion in
    ``test_compute_type_follows_dtype`` reads the attribute back through
    ``vllm.triton_utils.tl``, so the values are set (not replaced) on the
    shared placeholder module and restored afterwards.
    """
    for module in (fl_fused.tl, fmu.tl):
        for name in ("float16", "bfloat16", "float32"):
            monkeypatch.setattr(module, name, f"<{name}-sentinel>", raising=False)


# ---------------------------------------------------------------------------
# FlashinferMoeBackend version shim (bound on both platforms: the real
# flashinfer_utils import fails on vLLM 0.32 and on the stub graph alike)
# ---------------------------------------------------------------------------


class TestFlashinferMoeBackendShim:
    """The vLLM 0.28-compatibility FlashinferMoeBackend value shim."""

    @pytest.mark.supplementary
    def test_init_normalizes_value(self):
        """The constructor lowercases and strips the value."""
        shim = fmu.FlashinferMoeBackend("  CUTLASS ")
        assert shim.value == "cutlass"
        assert fmu.FlashinferMoeBackend("TrtLlm").value == "trtllm"

    @pytest.mark.supplementary
    def test_class_constants(self):
        """The two shipped constants are their lowercase names."""
        assert fmu.FlashinferMoeBackend.CUTLASS == "cutlass"
        assert fmu.FlashinferMoeBackend.TENSORRT_LLM == "trtllm"

    @pytest.mark.supplementary
    def test_eq_matches_value_and_substring(self):
        """Equality holds for equal values and for a member of this value."""
        shim = fmu.FlashinferMoeBackend("trtllm")
        assert shim == "trtllm"
        assert shim == fmu.FlashinferMoeBackend("TRTLLM")
        # A shorter string inside the value also matches ("trt" in "trtllm").
        assert shim == "trt"
        assert shim != "cutlass"
        assert shim != "trtllm-v2"

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
    def test_hash_and_str_roundtrip(self):
        """Hash and str are the normalized value."""
        shim = fmu.FlashinferMoeBackend(" TrtLlm ")
        assert hash(shim) == hash("trtllm")
        assert str(shim) == "trtllm"
        assert {shim: 1}["trtllm"] == 1


# ---------------------------------------------------------------------------
# _should_use_fl_triton_experts
# ---------------------------------------------------------------------------


class TestShouldUseFlTritonExperts:
    """The FlagGems-MoE policy gate combining platform and blacklist probes."""

    @pytest.mark.supplementary
    def test_policy_gate_truth_table(self, monkeypatch):
        """Each combination of the four policy inputs is honoured."""
        cases = [
            # is_cuda, is_oot, is_cpu, flaggems, blacklist, expected
            (True, False, False, True, None, True),
            (False, True, False, True, None, True),
            (True, False, True, True, None, False),
            (True, False, False, False, None, False),
            (True, False, False, True, ["fused_moe"], False),
        ]
        for is_cuda, is_oot, is_cpu, gems, blacklist, expected in cases:
            monkeypatch.setattr(
                fmu,
                "current_platform",
                _FakePlatform(is_cuda=is_cuda, is_out_of_tree=is_oot, is_cpu=is_cpu),
            )
            monkeypatch.setattr(fmu, "use_flaggems", lambda gems=gems: gems)
            monkeypatch.setattr(
                fmu, "get_oot_blacklist", lambda blacklist=blacklist: blacklist
            )
            assert fmu._should_use_fl_triton_experts() is expected


# ---------------------------------------------------------------------------
# _get_priority_backends
# ---------------------------------------------------------------------------


class TestGetPriorityBackends:
    """Platform-conditioned priority ordering of unquantized MoE backends."""

    @pytest.mark.supplementary
    def test_rocm_lists_aiter_first(self, monkeypatch):
        """ROCm puts AITER ahead of the two Triton variants."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_rocm=True))
        backends = fmu._get_priority_backends(_fake_moe_config())
        assert [b.name for b in backends] == ["AITER", "TRITON", "BATCHED_TRITON"]

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
    def test_xpu_platform_lists_xpu_only(self, monkeypatch):
        """XPU platforms offer only the XPU backend."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_xpu=True))
        backends = fmu._get_priority_backends(_fake_moe_config())
        assert [b.name for b in backends] == ["XPU"]

    @pytest.mark.supplementary
    def test_cpu_platform_lists_cpu_only(self, monkeypatch):
        """CPU platforms offer only the CPU backend."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_cpu=True))
        backends = fmu._get_priority_backends(_fake_moe_config())
        assert [b.name for b in backends] == ["CPU"]

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
    def test_tpu_short_circuits(self, monkeypatch):
        """TPU platforms return (TPU, None) before any config probe."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_tpu=True))
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "TPU"
        assert experts is None

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
    def test_npu_rejects_unsupported_explicit_backend(self, monkeypatch):
        """NPU only accepts moe_backend in ('auto', 'triton')."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="npu"))
        with pytest.raises(NotImplementedError, match="Unsupported explicit"):
            fmu.select_unquantized_moe_backend_oot(
                _fake_moe_config(moe_backend="flashinfer_cutlass")
            )

    @pytest.mark.supplementary
    def test_npu_returns_fl_triton_experts(self, monkeypatch):
        """NPU with a supported config selects TRITON and TritonExpertsFL."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="npu"))
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "TRITON"
        assert experts is fmu.TritonExpertsFL

    @pytest.mark.supplementary
    def test_fl_triton_experts_policy_short_circuit(self, monkeypatch):
        """OOT accelerators with FlagGems enabled select TritonExpertsFL."""
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(is_out_of_tree=True))
        monkeypatch.setattr(fmu, "use_flaggems", lambda: True)
        monkeypatch.setattr(fmu, "get_oot_blacklist", lambda: None)
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "TRITON"
        assert experts is fmu.TritonExpertsFL

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
    def test_auto_selects_first_supported_priority_backend(self, monkeypatch):
        """auto walks the priority list and returns the first supported pair."""
        cpu_cls = _Kernel()
        monkeypatch.setattr(fmu, "backend_to_kernel_cls", lambda backend: [cpu_cls])
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "CPU"
        assert experts is cpu_cls

    @pytest.mark.supplementary
    def test_auto_skips_unsupported_and_selects_next(self, monkeypatch):
        """auto continues past unsupported kernel classes to the next one."""
        classes = [_Kernel(False, "not on this device"), _Kernel()]

        def fake_b2k(backend):
            return classes

        monkeypatch.setattr(fmu, "backend_to_kernel_cls", fake_b2k)
        backend, experts = fmu.select_unquantized_moe_backend_oot(_fake_moe_config())
        assert backend.name == "CPU"
        assert experts is classes[1]

    @pytest.mark.supplementary
    def test_auto_raises_when_no_backend_supports(self, monkeypatch):
        """auto with zero supported backends raises NotImplementedError."""
        monkeypatch.setattr(
            fmu,
            "backend_to_kernel_cls",
            lambda backend: [_Kernel(False, "reason")],
        )
        with pytest.raises(NotImplementedError, match="No Unquantized MoE backend"):
            fmu.select_unquantized_moe_backend_oot(_fake_moe_config())

    @pytest.mark.supplementary
    def test_auto_raises_without_any_backend_list(self, monkeypatch):
        """A platform matching no branch leaves the list unset and fails."""
        monkeypatch.setattr(
            fmu,
            "current_platform",
            _FakePlatform(device_type="musa", vendor_name="mthreads"),
        )
        with pytest.raises((NotImplementedError, UnboundLocalError, AttributeError)):
            fmu.select_unquantized_moe_backend_oot(_fake_moe_config())

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
    def test_workspace_shapes_passthrough_on_non_npu(self, monkeypatch):
        """Non-NPU platforms return the base shapes untouched."""
        shapes = ((8, 16, 8), (8, 4, 16), (8, 8))
        experts, _captured = self._make_experts(monkeypatch, shapes)
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="cpu"))
        assert experts.workspace_shapes(8, 16, 8, 2, 4, 4, None, None) == shapes

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
    def test_workspace_shapes_npu_keeps_larger_m(self, monkeypatch):
        """NPU keeps M when it already exceeds max_num_tokens."""
        shapes = ((48, 16, 8), (48, 4, 16), (48, 8))
        experts, _captured = self._make_experts(monkeypatch, shapes, max_num_tokens=32)
        monkeypatch.setattr(fmu, "current_platform", _FakePlatform(device_type="npu"))
        w13, w2, _out = experts.workspace_shapes(48, 16, 8, 2, 4, 4, None, None)
        assert w13 == (48, 16, 8)
        assert w2 == (48, 4, 16)

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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

    @pytest.mark.supplementary
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


# ---------------------------------------------------------------------------
# _sqrtsoftplus_topk
# ---------------------------------------------------------------------------


class TestSqrtsoftplusTopk:
    """The pure-torch DSV4 sqrt(softplus) routing fallback."""

    @staticmethod
    def _call(logits, **kwargs):
        return fl_router._sqrtsoftplus_topk(
            gating_output=logits,
            e_score_correction_bias=kwargs.get("bias"),
            topk=kwargs.get("topk", 2),
            renormalize=kwargs.get("renormalize", False),
            indices_type=kwargs.get("indices_type"),
            input_tokens=kwargs.get("input_tokens"),
            hash_indices_table=kwargs.get("hash_indices_table"),
            routed_scaling_factor=kwargs.get("routed_scaling_factor", 1.0),
        )

    @pytest.mark.supplementary
    def test_matches_independent_oracle(self):
        """Weights are sqrt(softplus(logits)) gathered at the biased top-k."""
        logits = torch.tensor([[0.0, 1.0, -1.0, 3.0]], dtype=torch.float32)
        bias = torch.tensor([[0.4, 0.0, 0.8, -0.2]], dtype=torch.float32)
        weights, ids = self._call(logits, bias=bias, topk=2)
        # Independent oracle: softplus spelled with log1p(exp), hand-sorted
        # selection scores.
        sp = [math.log1p(math.exp(v.item())) for v in logits[0]]
        scores = [math.sqrt(v) for v in sp]
        choice = [scores[j] + bias[0, j].item() for j in range(4)]
        order = sorted(range(4), key=lambda j: -choice[j])
        assert set(ids[0].tolist()) == set(order[:2])
        for slot in range(2):
            torch.testing.assert_close(
                weights[0, slot].item(),
                scores[ids[0, slot].item()],
                rtol=1e-5,
                atol=1e-6,
            )

    @pytest.mark.supplementary
    def test_nan_logits_score_last_for_selection_only(self):
        """NaN logits score as -1e30 for selection; weights stay raw scores."""
        logits = torch.tensor([[float("nan"), 1.0, 2.0]], dtype=torch.float32)
        weights, ids = self._call(logits, topk=2)
        assert set(ids[0].tolist()) == {1, 2}  # the NaN expert is never selected
        sp = [math.log1p(math.exp(v)) for v in (1.0, 2.0)]
        expected = sorted([math.sqrt(v) for v in sp], reverse=True)
        got = sorted(weights[0].tolist(), reverse=True)
        for exp, got_v in zip(expected, got):
            torch.testing.assert_close(got_v, exp, rtol=1e-5, atol=1e-6)

    @pytest.mark.supplementary
    def test_renormalize_and_scaling_factor(self):
        """Renormalize divides by the selected sum; scaling multiplies after."""
        logits = torch.tensor([[0.5, 2.0, 4.0]], dtype=torch.float32)
        weights, ids = self._call(
            logits, topk=3, renormalize=True, routed_scaling_factor=0.5
        )
        sp = [math.log1p(math.exp(v.item())) for v in logits[0]]
        scores = [math.sqrt(v) for v in sp]
        total = sum(scores)
        # topk(sorted=False) returns the three ids in arbitrary order, so the
        # oracle is checked per slot through the returned ids.
        for slot in range(3):
            expected = 0.5 * scores[ids[0, slot].item()] / total
            torch.testing.assert_close(
                weights[0, slot].item(), expected, rtol=1e-4, atol=1e-6
            )

    @pytest.mark.supplementary
    def test_zero_denominator_guard(self):
        """A non-positive denominator is replaced by 1, not NaN."""
        # softplus(-inf) = 0, so the renormalize denominator is 0 and the
        # torch.where guard replaces it with 1: weights stay 0, not NaN.
        logits = torch.tensor([[-float("inf"), -float("inf")]])
        weights, ids = self._call(logits, topk=2, renormalize=True)
        assert len(set(ids[0].tolist())) == 2
        torch.testing.assert_close(weights, torch.zeros(1, 2))

    @pytest.mark.supplementary
    def test_indices_dtype_default_and_override(self):
        """indices_type=None writes int32; an explicit dtype is honoured."""
        logits = torch.tensor([[0.0, 1.0]])
        _w, ids_default = self._call(logits, topk=1)
        assert ids_default.dtype == torch.int32
        _w, ids_i64 = self._call(logits, topk=1, indices_type=torch.int64)
        assert ids_i64.dtype == torch.int64

    @pytest.mark.supplementary
    def test_hash_table_drives_selection_instead_of_scores(self):
        """With a hash table the ids come from token lookups, not topk."""
        # Hash values are expert ids and must index the gating output, so
        # every value stays below the 3-expert logits width.
        hash_table = torch.tensor([[1, 0], [2, 0]], dtype=torch.int64)
        weights, ids = self._call(
            torch.tensor([[0.0, 9.0, 4.0]]),
            topk=2,
            indices_type=torch.int64,
            input_tokens=torch.tensor([1], dtype=torch.int64),
            hash_indices_table=hash_table,
        )
        torch.testing.assert_close(ids, torch.tensor([[2, 0]], dtype=torch.int64))
        # Weights still gather sqrt(softplus(logits)) at those indices.
        sp0 = math.sqrt(math.log1p(math.exp(0.0)))
        sp2 = math.sqrt(math.log1p(math.exp(4.0)))
        torch.testing.assert_close(weights[0, 0].item(), sp2, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(weights[0, 1].item(), sp0, rtol=1e-4, atol=1e-5)

    @pytest.mark.supplementary
    def test_hash_table_casts_input_token_dtype(self):
        """input_tokens are cast to the hash table's dtype before lookup."""
        hash_table = torch.tensor([[1, 0], [2, 0]], dtype=torch.int32)
        weights, ids = self._call(
            torch.tensor([[0.0, 9.0, 4.0]]),
            topk=2,
            input_tokens=torch.tensor([1], dtype=torch.int64),
            hash_indices_table=hash_table,
        )
        assert ids.dtype == torch.int32
        assert set(ids[0].tolist()) == {2, 0}
        assert weights.shape == (1, 2)


# ---------------------------------------------------------------------------
# _has_native_moe_topk
# ---------------------------------------------------------------------------


class TestHasNativeMoeTopk:
    """The torch.ops._moe_C.topk_softmax capability probe."""

    @pytest.mark.supplementary
    def test_reports_false_without_topk_softmax(self, monkeypatch):
        """A _moe_C namespace lacking topk_softmax keeps the probe False."""
        monkeypatch.setattr(torch.ops, "_moe_C", types.SimpleNamespace(), raising=False)
        assert fl_router._has_native_moe_topk() is False

    @pytest.mark.supplementary
    def test_reports_true_with_topk_softmax(self, monkeypatch):
        """A _moe_C namespace exposing topk_softmax makes the probe True."""
        monkeypatch.setattr(
            torch.ops,
            "_moe_C",
            types.SimpleNamespace(topk_softmax=lambda *a: None),
            raising=False,
        )
        assert fl_router._has_native_moe_topk() is True


# ---------------------------------------------------------------------------
# _fl_grouped_topk: numeric math (pure torch, no dispatch)
# ---------------------------------------------------------------------------


class TestFlGroupedTopk:
    """The grouped top-k router math against independent plain-Python oracles."""

    @pytest.mark.supplementary
    def test_softmax_no_bias_matches_group_oracle(self):
        """softmax scoring without bias matches the plain-loop oracle."""
        torch.manual_seed(7)
        gating = torch.randn(4, 8, dtype=torch.float32) * 1.5
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(4, 3),
            gating_output=gating,
            topk=2,
            renormalize=True,
            num_expert_group=2,
            topk_group=1,
        )
        rows = gating.tolist()
        for row in range(4):
            # Independent oracle: softmax, per-group max, mask, topk.
            exps = [math.exp(v) for v in rows[row]]
            denom = sum(exps)
            probs = [e / denom for e in exps]
            group_scores = [max(probs[:4]), max(probs[4:])]
            chosen = 0 if group_scores[0] >= group_scores[1] else 1
            allowed = set(range(chosen * 4, chosen * 4 + 4))
            picked = sorted(((probs[i], i) for i in allowed), key=lambda t: -t[0])[:2]
            assert set(ids[row].tolist()) == {t[1] for t in picked}
            total = sum(t[0] for t in picked)
            got = {ids[row, s].item(): weights[row, s].item() for s in range(2)}
            for prob, idx in picked:
                torch.testing.assert_close(got[idx], prob / total, rtol=1e-4, atol=1e-6)

    @pytest.mark.supplementary
    def test_sigmoid_no_bias_matches_group_oracle(self):
        """sigmoid scoring without bias matches the plain-loop oracle."""
        torch.manual_seed(11)
        gating = torch.randn(3, 9, dtype=torch.float32)
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(3, 3),
            gating_output=gating,
            topk=3,
            renormalize=False,
            num_expert_group=3,
            topk_group=2,
            scoring_func="sigmoid",
        )
        rows = gating.tolist()
        for row in range(3):
            probs = [1.0 / (1.0 + math.exp(-v)) for v in rows[row]]
            group_scores = [max(probs[g * 3 : g * 3 + 3]) for g in range(3)]
            chosen = sorted(range(3), key=lambda g: -group_scores[g])[:2]
            allowed = {g * 3 + j for g in chosen for j in range(3)}
            picked = sorted(((probs[i], i) for i in allowed), key=lambda t: -t[0])[:3]
            assert set(ids[row].tolist()) == {t[1] for t in picked}
            got = {ids[row, s].item(): weights[row, s].item() for s in range(3)}
            for prob, idx in picked:
                torch.testing.assert_close(got[idx], prob, rtol=1e-4, atol=1e-6)

    @pytest.mark.supplementary
    def test_unknown_scoring_func_raises(self):
        """An unsupported scoring function raises ValueError (no bias)."""
        with pytest.raises(ValueError, match="Unsupported scoring function"):
            fl_router._fl_grouped_topk(
                hidden_states=torch.zeros(1, 3),
                gating_output=torch.zeros(1, 8),
                topk=1,
                renormalize=False,
                num_expert_group=2,
                topk_group=1,
                scoring_func="bogus",
            )

    @pytest.mark.supplementary
    def test_unknown_scoring_func_with_bias_raises(self):
        """The biased path also rejects unsupported scoring functions."""
        with pytest.raises(ValueError, match="Unsupported scoring function"):
            fl_router._fl_grouped_topk(
                hidden_states=torch.zeros(1, 3),
                gating_output=torch.zeros(1, 8),
                topk=1,
                renormalize=False,
                num_expert_group=2,
                topk_group=1,
                scoring_func="bogus",
                e_score_correction_bias=torch.zeros(8),
            )

    @pytest.mark.supplementary
    def test_routed_scaling_factor_scales_weights(self):
        """routed_scaling_factor multiplies the selected weights."""
        torch.manual_seed(13)
        gating = torch.randn(2, 8, dtype=torch.float32)
        base_w, base_i = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(2, 3),
            gating_output=gating,
            topk=2,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
        )
        scaled_w, scaled_i = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(2, 3),
            gating_output=gating,
            topk=2,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
            routed_scaling_factor=2.5,
        )
        # Tie order is unspecified: compare value multisets per row.
        for row in range(2):
            assert set(scaled_i[row].tolist()) == set(base_i[row].tolist())
            for b, s in zip(
                sorted(base_w[row].tolist()), sorted(scaled_w[row].tolist())
            ):
                torch.testing.assert_close(s, b * 2.5, rtol=1e-4, atol=1e-6)

    @pytest.mark.supplementary
    def test_token_count_mismatch_raises(self):
        """hidden_states / gating_output row mismatch fails the assert."""
        with pytest.raises(AssertionError, match="Number of tokens mismatch"):
            fl_router._fl_grouped_topk(
                hidden_states=torch.zeros(2, 3),
                gating_output=torch.zeros(3, 8),
                topk=1,
                renormalize=False,
                num_expert_group=2,
                topk_group=1,
            )

    @pytest.mark.supplementary
    def test_output_dtypes_are_normalized(self):
        """Weights come back float32 and ids int32 regardless of input dtype."""
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(1, 3),
            gating_output=torch.zeros(1, 8, dtype=torch.float64),
            topk=1,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
        )
        assert weights.dtype == torch.float32
        assert ids.dtype == torch.int32


# ---------------------------------------------------------------------------
# _fl_grouped_topk biased path (dispatched through the CachedOp boundary)
# ---------------------------------------------------------------------------


class TestFlGroupedTopkBiased:
    """The e_score_correction_bias path routed through the grouped_topk op."""

    @pytest.mark.supplementary
    def test_sigmoid_bias_forwards_to_grouped_topk_dispatch(self, monkeypatch):
        """sigmoid+bias hands the raw logits to grouped_topk with mode 1."""
        captured = {}

        def fake_grouped_topk(
            scores,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            scaling,
            bias,
            mode,
        ):
            captured.update(
                scores=scores,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                topk=topk,
                renormalize=renormalize,
                scaling=scaling,
                bias=bias,
                mode=mode,
            )
            return (
                torch.tensor([[0.5, 0.25]]),
                torch.tensor([[1, 0]], dtype=torch.int32),
            )

        _replace_cached_op(monkeypatch, fl_router, "_grouped_topk", fake_grouped_topk)
        bias = torch.tensor([0.1, -0.1])
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(1, 3),
            gating_output=torch.tensor([[0.3, 0.7]]),
            topk=2,
            renormalize=False,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            e_score_correction_bias=bias,
        )
        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.25]]))
        assert ids.tolist() == [[1, 0]]
        # Sigmoid mode is encoded as 1; the raw logits are handed over.
        assert captured["mode"] == 1
        assert captured["num_expert_group"] == 1
        assert captured["bias"] is bias
        torch.testing.assert_close(captured["scores"], torch.tensor([[0.3, 0.7]]))

    @pytest.mark.supplementary
    def test_softmax_bias_applies_softmax_first(self, monkeypatch):
        """softmax+bias softmaxes the logits before the dispatched call."""
        captured = {}

        def fake_grouped_topk(
            scores,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            scaling,
            bias,
            mode,
        ):
            captured.update(scores=scores, mode=mode)
            return torch.tensor([[0.5]]), torch.tensor([[0]], dtype=torch.int32)

        _replace_cached_op(monkeypatch, fl_router, "_grouped_topk", fake_grouped_topk)
        fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(1, 3),
            gating_output=torch.tensor([[0.0, math.log(3.0)]]),
            topk=1,
            renormalize=False,
            num_expert_group=1,
            topk_group=1,
            scoring_func="softmax",
            e_score_correction_bias=torch.zeros(2),
        )
        assert captured["mode"] == 0
        # Independent oracle: softmax([0, ln3]) = [1/4, 3/4].
        torch.testing.assert_close(
            captured["scores"], torch.tensor([[0.25, 0.75]]), rtol=1e-6, atol=1e-6
        )

    @pytest.mark.supplementary
    def test_dispatched_result_is_returned_verbatim(self, monkeypatch):
        """The dispatched kernel's outputs are returned unchanged."""
        sentinel_w = torch.tensor([[0.125]])
        sentinel_i = torch.tensor([[3]], dtype=torch.int32)
        _replace_cached_op(
            monkeypatch,
            fl_router,
            "_grouped_topk",
            lambda *a, **k: (sentinel_w, sentinel_i),
        )
        weights, ids = fl_router._fl_grouped_topk(
            hidden_states=torch.zeros(1, 3),
            gating_output=torch.zeros(1, 4),
            topk=1,
            renormalize=False,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            e_score_correction_bias=torch.zeros(4),
        )
        assert weights is sentinel_w
        assert ids is sentinel_i


# ---------------------------------------------------------------------------
# GroupedTopKRouterFL._compute_routing
# ---------------------------------------------------------------------------


class TestGroupedTopKRouterComputeRouting:
    """GroupedTopKRouterFL's grouping-validity gate and dispatch choice."""

    def _router(self, **overrides):
        """A router instance answering only the routing probes."""
        params = dict(
            top_k=2,
            renormalize=True,
            num_expert_group=2,
            topk_group=1,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
            num_fused_shared_experts=0,
        )
        params.update(overrides)
        router = types.SimpleNamespace(**params)
        # Invoke the FL method unbound, exactly as the class-level patch does.
        router._compute_routing = (
            fl_router.GroupedTopKRouterFL._compute_routing.__get__(router)
        )
        return router

    @pytest.mark.supplementary
    def test_invalid_grouping_without_bias_uses_fused_topk(self, monkeypatch):
        """num_experts <= num_expert_group routes through plain fused_topk."""
        router = self._router()
        recorded = {}

        def fake_fused_topk(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.5, 0.5]]),
                torch.tensor([[0, 1]], dtype=torch.int32),
                None,
            )

        monkeypatch.setattr(fl_router, "fused_topk", fake_fused_topk)
        # 2 logits columns <= num_expert_group=2 makes the grouping invalid.
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 2), None
        )
        assert recorded["topk"] == 2
        assert recorded["renormalize"] is True
        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5]]))
        assert ids.tolist() == [[0, 1]]

    @pytest.mark.supplementary
    def test_invalid_grouping_with_bias_uses_fused_topk_bias(self, monkeypatch):
        """A bias present on invalid grouping routes through fused_topk_bias."""
        bias = torch.tensor([0.1, 0.2, 0.3, 0.4])
        router = self._router(e_score_correction_bias=bias)
        recorded = {}

        def fake_fused_topk_bias(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.6, 0.4]]),
                torch.tensor([[3, 0]], dtype=torch.int32),
            )

        monkeypatch.setattr(fl_router, "fused_topk_bias", fake_fused_topk_bias)
        # 3 logits columns % num_expert_group=2 != 0 makes the grouping
        # invalid, exercising the bias branch of the fallback.
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 3), None
        )
        # .data detaches into a fresh view each access, so the recorded value
        # is checked for shared storage against the tensor's own data.
        assert recorded["e_score_correction_bias"].data_ptr() == bias.data_ptr()
        assert recorded["topk"] == 2
        torch.testing.assert_close(weights, torch.tensor([[0.6, 0.4]]))

    @pytest.mark.supplementary
    def test_invalid_grouping_bias_applies_scaling_after(self, monkeypatch):
        """routed_scaling_factor is applied on top of the bias-router result."""
        bias = torch.tensor([0.1, 0.2])
        router = self._router(e_score_correction_bias=bias, routed_scaling_factor=2.0)
        monkeypatch.setattr(
            fl_router,
            "fused_topk_bias",
            lambda **k: (torch.tensor([[0.5, 0.25]]), torch.tensor([[0, 1]])),
        )
        weights, _ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 2), None
        )
        torch.testing.assert_close(weights, torch.tensor([[1.0, 0.5]]))

    @pytest.mark.supplementary
    def test_aiter_enabled_routes_to_rocm_partial(self, monkeypatch):
        """AITER-enabled builds route to rocm_aiter_grouped_topk."""
        router = self._router()
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True
        )
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            lambda: True,
        )
        recorded = {}

        def fake_aiter_topk(**kwargs):
            recorded.update(kwargs)
            return torch.tensor([[0.9]]), torch.tensor([[1]], dtype=torch.int32)

        monkeypatch.setattr(fl_router, "rocm_aiter_grouped_topk", fake_aiter_topk)
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 8), None
        )
        assert recorded["num_fused_shared_experts"] == 0
        assert recorded["num_expert_group"] == 2
        torch.testing.assert_close(weights, torch.tensor([[0.9]]))
        assert ids.tolist() == [[1]]

    @pytest.mark.supplementary
    def test_aiter_shared_experts_mismatch_asserts(self, monkeypatch):
        """AITER with shared experts disabled asserts num_fused_shared_experts==0."""
        router = self._router(num_fused_shared_experts=2)
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True
        )
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            lambda: False,
        )
        with pytest.raises(AssertionError):
            router._compute_routing(torch.zeros(1, 3), torch.zeros(1, 8), None)

    @pytest.mark.supplementary
    def test_aiter_disabled_routes_to_fl_grouped_topk(self, monkeypatch):
        """AITER disabled routes valid groupings through _fl_grouped_topk."""
        router = self._router()
        monkeypatch.setattr(
            fl_router.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False
        )
        recorded = {}

        def fake_fl_grouped_topk(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.7, 0.3]]),
                torch.tensor([[2, 5]], dtype=torch.int32),
            )

        monkeypatch.setattr(fl_router, "_fl_grouped_topk", fake_fl_grouped_topk)
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 8), None
        )
        assert recorded["num_expert_group"] == 2
        assert recorded["topk_group"] == 1
        torch.testing.assert_close(weights, torch.tensor([[0.7, 0.3]]))
        assert ids.tolist() == [[2, 5]]

    @pytest.mark.supplementary
    def test_indivisible_grouping_is_invalid(self, monkeypatch):
        """num_experts not divisible by num_expert_group is invalid grouping."""
        router = self._router(num_expert_group=3)
        recorded = {}

        def fake_fl_grouped_topk(**kwargs):
            recorded.update(kwargs)
            return torch.tensor([[1.0]]), torch.tensor([[0]], dtype=torch.int32)

        monkeypatch.setattr(
            fl_router.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False
        )
        monkeypatch.setattr(fl_router, "_fl_grouped_topk", fake_fl_grouped_topk)
        # 4 experts / 3 groups is indivisible: valid_grouping() returns False,
        # so the no-bias fallback (plain fused_topk) runs instead.
        monkeypatch.setattr(
            fl_router,
            "fused_topk",
            lambda **k: (
                torch.tensor([[0.4]]),
                torch.tensor([[0]], dtype=torch.int32),
                None,
            ),
        )
        weights, _ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 4), None
        )
        assert recorded == {}  # _fl_grouped_topk was never reached
        torch.testing.assert_close(weights, torch.tensor([[0.4]]))


# ---------------------------------------------------------------------------
# FusedTopKBiasRouterFL._compute_routing
# ---------------------------------------------------------------------------


class TestFusedTopKBiasRouterComputeRouting:
    """FusedTopKBiasRouterFL's sqrtsoftplus-ascend and shared-expert paths."""

    def _router(self, **overrides):
        params = dict(
            top_k=2,
            renormalize=False,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
            num_fused_shared_experts=0,
            global_num_experts=8,
            shared_expert_weight=0.5,
            _hash_indices_table=None,
            bias_vl=None,
            image_sentinel_lo=None,
        )
        params.update(overrides)
        router = types.SimpleNamespace(**params)
        router._compute_routing = (
            fl_router.FusedTopKBiasRouterFL._compute_routing.__get__(router)
        )
        return router

    @pytest.mark.supplementary
    def test_softmax_routes_through_fused_topk_bias(self, monkeypatch):
        """Non-sqrtsoftplus scoring routes through upstream fused_topk_bias."""
        router = self._router()
        recorded = {}

        def fake_fused_topk_bias(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.5, 0.5]]),
                torch.tensor([[0, 1]], dtype=torch.int32),
            )

        monkeypatch.setattr(fl_router, "fused_topk_bias", fake_fused_topk_bias)
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 8), None
        )
        assert recorded["topk"] == 2
        assert recorded["bias_vl"] is None
        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5]]))
        assert ids.tolist() == [[0, 1]]

    @pytest.mark.supplementary
    def test_correction_bias_data_is_forwarded(self, monkeypatch):
        """The registered nn.Parameter .data is what reaches fused_topk_bias."""
        bias = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        router = self._router(e_score_correction_bias=bias)
        recorded = {}

        def fake_fused_topk_bias(**kwargs):
            recorded.update(kwargs)
            return (
                torch.tensor([[0.5, 0.5]]),
                torch.tensor([[0, 1]], dtype=torch.int32),
            )

        monkeypatch.setattr(fl_router, "fused_topk_bias", fake_fused_topk_bias)
        router._compute_routing(torch.zeros(1, 3), torch.zeros(1, 4), None)
        # .data detaches into a fresh view each access, so the recorded value
        # is checked for shared storage against the parameter's own data.
        torch.testing.assert_close(recorded["e_score_correction_bias"], bias.data)
        assert recorded["e_score_correction_bias"].data_ptr() == bias.data_ptr()

    @pytest.mark.supplementary
    def test_shared_experts_are_appended(self, monkeypatch):
        """num_fused_shared_experts > 0 appends the shared expert slots."""
        router = self._router(
            num_fused_shared_experts=2,
            global_num_experts=8,
            shared_expert_weight=0.5,
        )
        monkeypatch.setattr(
            fl_router,
            "fused_topk_bias",
            lambda **k: (
                torch.tensor([[0.25, 0.75]]),
                torch.tensor([[1, 0]], dtype=torch.int32),
            ),
        )
        weights, ids = router._compute_routing(
            torch.zeros(1, 3), torch.zeros(1, 8), None
        )
        # Independent oracle: ids gain [8, 9], weights gain [0.5, 0.5].
        assert ids.tolist() == [[1, 0, 8, 9]]
        torch.testing.assert_close(weights, torch.tensor([[0.25, 0.75, 0.5, 0.5]]))


# ---------------------------------------------------------------------------
# replace_router_with_fl
# ---------------------------------------------------------------------------


class TestReplaceRouterWithFl:
    """The class-level monkey-patch installation of the FL router methods."""

    @pytest.mark.supplementary
    def test_patch_assigns_fl_compute_routing_to_all_three(self, monkeypatch):
        """The three upstream router classes receive the FL _compute_routing."""
        # The upstream classes are bound in fl_router's own namespace at its
        # import (the same objects replace_router_with_fl patches), so they
        # are reached through it rather than a fresh vllm import — which on
        # stub hosts would fail after the file-scope restore removed the
        # stub graph.
        m1 = fl_router
        m2 = fl_router
        m3 = fl_router

        # Pin the originals through monkeypatch so the patch is auto-reversed.
        for module, cls_name in (
            (m1, "FusedTopKRouter"),
            (m2, "FusedTopKBiasRouter"),
            (m3, "GroupedTopKRouter"),
        ):
            cls = getattr(module, cls_name)
            monkeypatch.setattr(
                cls, "_compute_routing", lambda self, *a, **k: ("orig", "orig")
            )
        fl_router.replace_router_with_fl()
        assert (
            m1.FusedTopKRouter._compute_routing
            is fl_router.FusedTopKRouterFL._compute_routing
        )
        assert (
            m3.GroupedTopKRouter._compute_routing
            is fl_router.GroupedTopKRouterFL._compute_routing
        )
        assert (
            m2.FusedTopKBiasRouter._compute_routing
            is fl_router.FusedTopKBiasRouterFL._compute_routing
        )


# ---------------------------------------------------------------------------
# fused_experts_impl dispatch contracts
# ---------------------------------------------------------------------------


class TestFusedExpertsImplContracts:
    """fused_experts_impl shape assertions and boundary-call contracts.

    The Triton GEMM kernels and the config lookup are external boundaries;
    recorders replace them so the tests assert what the function hands to the
    dispatch system, not the kernels themselves.
    """

    E, N, K, TOPK = 2, 8, 4, 2

    def _make_inputs(self, num_tokens=3, dtype=torch.float32):
        hidden = torch.randn(num_tokens, self.K, dtype=dtype)
        w1 = torch.randn(self.E, 2 * self.N, self.K, dtype=dtype)
        w2 = torch.randn(self.E, self.K, self.N, dtype=dtype)
        tw = torch.rand(num_tokens, self.TOPK, dtype=dtype)
        ti = torch.randint(0, self.E, (num_tokens, self.TOPK))
        return hidden, w1, w2, tw, ti

    @pytest.fixture
    def recorders(self, monkeypatch):
        """Record every boundary call fused_experts_impl makes."""
        calls = {"align": [], "kernel": [], "moe_sum": [], "config": []}

        def fake_align(ids, block_size, num_experts, expert_map, **kwargs):
            calls["align"].append((tuple(ids.shape), block_size, num_experts))
            return (
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                torch.tensor([8], dtype=torch.int32),
            )

        def fake_kernel(*args, **kwargs):
            # Positional contract of _invoke_fused_moe_triton_kernel:
            # (A, B, C, A_scale, B_scale, topk_weights, sorted_ids,
            #  expert_ids, num_post_pad, mul_routed_weight, top_k, config)
            calls["kernel"].append(
                {
                    "B_shape": tuple(args[1].shape),
                    "C_shape": tuple(args[2].shape),
                    "mul_routed_weight": args[9],
                    "top_k": args[10],
                    "compute_type": kwargs.get("compute_type"),
                }
            )

        def fake_moe_sum(inp, out):
            calls["moe_sum"].append((tuple(inp.shape), tuple(out.shape)))
            torch.sum(inp, dim=1, out=out)

        def fake_silu_and_mul(unused, x):
            # Independent oracle: silu over the first half, multiplied by the
            # second half — the contract the real dispatched kernel provides.
            d = x.shape[-1] // 2
            return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

        def fake_config(*args, **kwargs):
            calls["config"].append(args)
            return {"BLOCK_SIZE_M": 16}

        monkeypatch.setattr(fl_fused, "_moe_align_block_size", fake_align)
        monkeypatch.setattr(fl_fused, "_invoke_fused_moe_triton_kernel", fake_kernel)
        _replace_cached_op(monkeypatch, fl_fused, "_moe_sum", fake_moe_sum)
        # _silu_and_mul is a CachedOp too: resolving it initializes the
        # dispatch system, which probes vllm.platforms and cannot run on a
        # stub host. Replacing it with the numeric oracle keeps the dispatch
        # boundary (like the other kernels) out of the contract under test.
        _replace_cached_op(monkeypatch, fl_fused, "_silu_and_mul", fake_silu_and_mul)
        # vLLM 0.32's _get_config_quant_dtype lacks the ocp_mx_scheme kwarg
        # the FL call site passes, and the config lookup signature differs per
        # vLLM version: pin both to the contract fused_experts_impl expects.
        monkeypatch.setattr(
            fl_fused, "_get_config_quant_dtype", lambda **kwargs: None, raising=False
        )
        monkeypatch.setattr(
            fl_fused, "try_get_optimal_moe_config", fake_config, raising=False
        )
        return calls

    @pytest.mark.supplementary
    def test_happy_path_dispatch_contract(self, recorders):
        """Both GEMMs, the activation, and moe_sum receive proper shapes."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        out = fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert out.shape == hidden.shape
        assert len(recorders["kernel"]) == 2
        first, second = recorders["kernel"]
        # The implementation reads N from w1's second dim — the fused
        # gate+up width, i.e. 2 * the per-gate N the inputs are built with.
        fused_n = 2 * self.N
        assert first["B_shape"] == (self.E, fused_n, self.K)
        assert first["C_shape"] == (hidden.size(0), self.TOPK, fused_n)
        assert second["B_shape"] == (self.E, self.K, self.N)
        assert second["C_shape"] == (hidden.size(0), self.TOPK, self.K)
        # First GEMM multiplies routed weights iff requested on input;
        # the second GEMM applies them iff the first did not.
        assert first["mul_routed_weight"] is False
        assert second["mul_routed_weight"] is True
        assert second["top_k"] == 1
        # Alignment gets the block size from the config lookup.
        assert recorders["align"][0] == (ti.shape, 16, self.E)
        assert recorders["moe_sum"][0] == (
            (hidden.size(0), self.TOPK, self.K),
            (hidden.size(0), self.K),
        )

    @pytest.mark.supplementary
    def test_inplace_writes_into_hidden_states(self, recorders):
        """inplace=True writes the result into the hidden_states tensor."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        out = fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti, inplace=True)
        assert out is hidden

    @pytest.mark.supplementary
    def test_router_weight_on_input_flips_gemm_flags(self, recorders):
        """apply_router_weight_on_input inverts both GEMM weight flags."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        fl_fused.fused_experts_impl(
            hidden, w1, w2, tw, ti, apply_router_weight_on_input=True
        )
        first, second = recorders["kernel"]
        assert first["mul_routed_weight"] is True
        assert second["mul_routed_weight"] is False

    @pytest.mark.supplementary
    def test_compute_type_follows_dtype(self, recorders):
        """The Triton compute type is chosen from the activation dtype."""
        # fl_fused binds the upstream tl namespace at its own import; the
        # stub graph is gone by test time on vllm-less hosts, so the bound
        # reference is used instead of importing vllm.triton_utils.
        _vtu = fl_fused

        hidden, w1, w2, tw, ti = self._make_inputs(dtype=torch.float16)
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert recorders["kernel"][0]["compute_type"] is _vtu.tl.float16
        recorders["kernel"].clear()
        hidden, w1, w2, tw, ti = self._make_inputs(dtype=torch.bfloat16)
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert recorders["kernel"][0]["compute_type"] is _vtu.tl.bfloat16

    @pytest.mark.supplementary
    def test_gelu_activation_routes_to_gelu_and_mul(self, recorders, monkeypatch):
        """activation='gelu' dispatches gelu_and_mul instead of silu_and_mul."""
        hidden, w1, w2, tw, ti = self._make_inputs()

        def fake_gelu(out, x):
            d = x.shape[-1] // 2
            return torch.nn.functional.gelu(x[..., :d]) * x[..., d:]

        gelu_rec = _replace_cached_op(monkeypatch, fl_fused, "_gelu_and_mul", fake_gelu)
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti, activation="gelu")
        assert gelu_rec.calls

    @pytest.mark.supplementary
    def test_no_mul_activations_stay_in_torch(self, recorders):
        """silu_no_mul keeps the activation inside torch (F.silu path)."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        out = fl_fused.fused_experts_impl(
            hidden, w1, w2, tw, ti, activation="silu_no_mul"
        )
        assert len(recorders["kernel"]) == 2  # GEMMs still ran
        assert out.shape == hidden.shape

    @pytest.mark.supplementary
    def test_unsupported_activation_raises(self, recorders):
        """An unknown activation name raises ValueError."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        with pytest.raises(ValueError, match="Unsupported FusedMoe activation"):
            fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti, activation="relu")

    @pytest.mark.supplementary
    def test_hidden_size_mismatch_raises(self, recorders):
        """A hidden size inconsistent with w1 fails the shape assertion."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        bad_w1 = torch.randn(self.E, 2 * self.N, self.K + 2)
        with pytest.raises(AssertionError, match="Hidden size mismatch"):
            fl_fused.fused_experts_impl(hidden, bad_w1, w2, tw, ti)

    @pytest.mark.supplementary
    def test_topk_shape_mismatch_raises(self, recorders):
        """topk_weights / topk_ids shape disagreement fails the assertion."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        with pytest.raises(AssertionError, match="topk shape mismatch"):
            fl_fused.fused_experts_impl(hidden, w1, w2, tw[:, :1], ti)

    @pytest.mark.supplementary
    def test_int4_hidden_size_halved_assertion(self, recorders):
        """use_int4_w4a16 checks hidden // 2 == w1.size(2)."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        # int4 packs 2 weights per element: w1 must see K columns packed
        # from K/2 hidden elements. A matching w1 passes; a mismatched one
        # raises.
        ok_w1 = torch.randn(self.E, 2 * self.N, self.K // 2)
        fl_fused.fused_experts_impl(hidden, ok_w1, w2, tw, ti, use_int4_w4a16=True)
        bad_w1 = torch.randn(self.E, 2 * self.N, self.K // 2 + 1)
        with pytest.raises(AssertionError, match="Hidden size mismatch"):
            fl_fused.fused_experts_impl(hidden, bad_w1, w2, tw, ti, use_int4_w4a16=True)

    @pytest.mark.supplementary
    def test_enum_activation_is_normalized_to_string(self, recorders, monkeypatch):
        """A MoEActivation enum is normalized via its .value attribute."""
        hidden, w1, w2, tw, ti = self._make_inputs()

        def fake_silu(out, x):
            d = x.shape[-1] // 2
            return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

        silu_rec = _replace_cached_op(monkeypatch, fl_fused, "_silu_and_mul", fake_silu)
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti, activation=_fl_enum("SILU"))
        assert silu_rec.calls

    @pytest.mark.supplementary
    def test_global_num_experts_defaults_from_w1(self, recorders):
        """global_num_experts=-1 derives the expert count from w1.size(0)."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert recorders["align"][0][2] == self.E

    @pytest.mark.supplementary
    def test_explicit_global_num_experts_forwarded(self, recorders):
        """An explicit global_num_experts reaches the alignment boundary."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti, global_num_experts=17)
        assert recorders["align"][0][2] == 17

    @pytest.mark.supplementary
    def test_quantized_activation_paths_dispatch_kernel(self, recorders):
        """fp8 flags flow through to the kernel boundary unchanged."""
        hidden, w1, w2, tw, ti = self._make_inputs()
        fl_fused.fused_experts_impl(
            hidden, w1, w2, tw, ti, use_fp8_w8a8=True, per_channel_quant=True
        )
        assert len(recorders["kernel"]) == 2

    @pytest.mark.supplementary
    def test_single_chunk_runs_one_config_lookup(self, recorders):
        """num_tokens < CHUNK_SIZE runs one chunk and one config lookup."""
        hidden, w1, w2, tw, ti = self._make_inputs(num_tokens=3)
        fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
        assert len(recorders["config"]) == 1
        assert len(recorders["moe_sum"]) == 1

    @pytest.mark.supplementary
    def test_out_dtype_matches_input(self, recorders):
        """float16 and bfloat16 inputs flow through with matching dtypes."""
        for dtype in (torch.float16, torch.bfloat16):
            recorders["kernel"].clear()
            recorders["moe_sum"].clear()
            hidden, w1, w2, tw, ti = self._make_inputs(dtype=dtype)
            out = fl_fused.fused_experts_impl(hidden, w1, w2, tw, ti)
            assert out.dtype == dtype
            assert len(recorders["kernel"]) == 2

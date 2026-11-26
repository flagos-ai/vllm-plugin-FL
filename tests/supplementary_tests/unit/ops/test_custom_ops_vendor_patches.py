# Copyright 2025 BAAI. All rights reserved.
"""PR-#809 vendor-patch increments for ``vllm_fl.ops.custom_ops.register_oot_ops``.

Coverage boundary (post-refactor): the common registration wiring of
``register_oot_ops`` (disabled/oracle-only path, env whitelist/blacklist
precedence, unknown-op warning, parameter whitelist, prefer-enabled
override, registration name contract, the fused_moe-free OOT_OPS table)
and the ``_patch_unquantized_moe_oracle`` import-site rewrite are owned
by PR #799's ``tests/supplementary_tests/unit/ops/test_custom_ops_registration.py``.
This file keeps only the #809-specific vendor parameterization branches —
the device_type / vendor dispatch inside the op loop that steers
``register_oot_ops`` to each vendor's patch module:

* npu device_type -> apply_ascend_patches;
* ptpu device_type -> apply_sunrise_patches;
* gcu vendor_name -> apply_gcu_patches;
* kunlunxin vendor_name -> apply_kunlunxin_patches (even when no OOT
  operator was selected);
* fused_moe exclusion from registration still patching the oracle.

``custom_ops`` and the operator classes it registers import vllm at
module level, so the tests below install a stub vllm module graph first.
Only the external vllm boundary is stubbed; ``register_oot_ops`` itself
runs unmodified. The stubbed ``register_oot`` classmethods record
registrations so each test can assert exactly which ops were registered,
with which names, and that the vendor patch entry points were invoked.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.supplementary]

_STUB_ROOTS = ("vllm", "vllm_fl", "flag_gems", "flaggems_vllm")
_MISSING = object()


@pytest.fixture(autouse=True)
def _fl_sys_modules():
    """Snapshot ``sys.modules`` around stub installs and restore exactly.

    Records every ``vllm*`` / ``vllm_fl*`` / ``flag_gems*`` /
    ``flaggems_vllm*`` entry present before the test, removes entries the
    test added, and puts back the original module objects for entries the
    test replaced.
    """
    saved = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_ROOTS:
            saved[name] = sys.modules.get(name, _MISSING)
    try:
        yield saved
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name.split(".")[0] in _STUB_ROOTS and name not in saved
        ]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _StubRegistry:
    """Records CustomOp.register_oot / PluggableLayer.register_oot calls."""

    def __init__(self):
        self.custom_ops = []
        self.pluggable_layers = []


def _make_module(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


def _install_vllm_graph():
    """Build the vllm module graph the ops subpackage needs at import time.

    Returns the registry object holding the recorded register_oot calls.
    """
    vllm = _make_module("vllm")
    platforms = _make_module("vllm.platforms")
    logger_mod = _make_module("vllm.logger")
    logger_mod.init_logger = lambda name: __import__("logging").getLogger(name)
    envs_mod = _make_module("vllm.envs")
    envs_mod.GEMS = None
    me = _make_module("vllm.model_executor")
    layers = _make_module("vllm.model_executor.layers")
    fused_moe = _make_module("vllm.model_executor.layers.fused_moe")
    custom_op = _make_module("vllm.model_executor.custom_op")
    activation_mod = _make_module("vllm.model_executor.layers.activation")
    layernorm_mod = _make_module("vllm.model_executor.layers.layernorm")
    rotary_mod = _make_module("vllm.model_executor.layers.rotary_embedding")
    fused_moe_layer = _make_module("vllm.model_executor.layers.fused_moe.layer")
    fm_config = _make_module("vllm.model_executor.layers.fused_moe.config")
    fm_runner = _make_module(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner"
    )
    fm_modular = _make_module(
        "vllm.model_executor.layers.fused_moe.modular_kernel"
    )
    fm_activation = _make_module(
        "vllm.model_executor.layers.fused_moe.activation"
    )
    fm_experts_triton = _make_module(
        "vllm.model_executor.layers.fused_moe.experts.triton_moe"
    )
    fm_experts_dir = _make_module("vllm.model_executor.layers.fused_moe.experts")
    fm_experts_aiter = _make_module(
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe"
    )
    router_dir = _make_module("vllm.model_executor.layers.fused_moe.router")
    router_fused = _make_module(
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router"
    )
    router_bias = _make_module(
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
    )
    router_grouped = _make_module(
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router"
    )
    fm_fused = _make_module("vllm.model_executor.layers.fused_moe.fused_moe")
    fm_utils = _make_module("vllm.model_executor.layers.fused_moe.utils")
    aiter_ops = _make_module("vllm._aiter_ops")
    triton_utils = _make_module("vllm.triton_utils")

    class _EnumStub:
        """Values used by fused_moe_utils at import time only as markers."""

        AITER = "aiter"
        TRITON = "triton"
        BATCHED_TRITON = "batched_triton"
        FLASHINFER_TRTLLM = "flashinfer_trtllm"
        FLASHINFER_CUTLASS = "flashinfer_cutlass"
        XPU = "xpu"
        CPU = "cpu"

    oracle_unquantized = _make_module(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized"
    )
    unquantized_method = _make_module(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
    )

    class BaseOp:
        """Stub CustomOp base: register / register_oot classmethod recorders."""

        registry: _StubRegistry = None

        @classmethod
        def register(cls, name):
            def decorator(obj):
                return obj

            return decorator

        @classmethod
        def register_oot(cls, _decorated_op_cls=None, name=None):
            cls.registry.custom_ops.append((name, _decorated_op_cls))

            def decorator(obj):
                return obj

            return decorator

    class PluggableLayerStub(BaseOp):
        """Stub PluggableLayer: register_oot goes to the pluggable bucket."""

        @classmethod
        def register_oot(cls, _decorated_op_cls=None, name=None):
            cls.registry.pluggable_layers.append((name, _decorated_op_cls))

            def decorator(obj):
                return obj

            return decorator

    registry = _StubRegistry()
    BaseOp.registry = registry
    PluggableLayerStub.registry = registry

    class _Platform:
        device_type = "cuda"
        vendor_name = "nvidia"

        def is_cuda(self):
            return True

        def is_cuda_alike(self):
            return True

    platforms.current_platform = _Platform()
    platforms.PlatformEnum = None

    class SiluAndMul(BaseOp):
        pass

    class GeluAndMul(BaseOp):
        pass

    class RMSNorm(BaseOp):
        pass

    class RotaryEmbedding(BaseOp):
        pass

    activation_mod.SiluAndMul = SiluAndMul
    activation_mod.GeluAndMul = GeluAndMul
    layernorm_mod.RMSNorm = RMSNorm
    rotary_mod.RotaryEmbedding = RotaryEmbedding
    custom_op.CustomOp = BaseOp
    custom_op.PluggableLayer = PluggableLayerStub
    unquantized_method.UnquantizedFusedMoEMethod = BaseOp
    oracle_unquantized.UnquantizedMoeBackend = _EnumStub
    oracle_unquantized._trtllm_bf16_lora_supported = lambda *a, **k: False
    oracle_unquantized.backend_to_kernel_cls = {}
    oracle_unquantized.map_unquantized_backend = lambda *a, **k: None
    fm_modular.FusedMoEExperts = BaseOp
    fm_modular.ExpertTokensMetadata = object
    fm_activation.MoEActivation = _EnumStub
    fm_activation.ApplyMoEActivationConfig = object
    fm_activation.apply_moe_activation = lambda *a, **k: None
    fm_experts_triton.TritonExperts = BaseOp
    fm_fused.try_get_optimal_moe_config = lambda *a, **k: None
    fm_utils._resize_cache = lambda *a, **k: None
    fm_utils.moe_kernel_quantize_input = lambda *a, **k: None
    aiter_ops.rocm_aiter_ops = None
    triton_utils.tl = None
    triton_utils.triton = None
    fm_config.FusedMoEConfig = object
    fm_runner.MoERunner = object
    fm_experts_aiter.rocm_aiter_grouped_topk = lambda *a, **k: None
    router_fused.FusedTopKRouter = BaseOp
    router_fused._get_padding_mask = lambda *a, **k: None
    router_fused.fused_topk = lambda *a, **k: None
    router_bias.FusedTopKBiasRouter = BaseOp
    router_bias.fused_topk_bias = lambda *a, **k: None
    router_grouped.GroupedTopKRouter = BaseOp

    class _OrigFactory:
        def __call__(self, *args, **kwargs):
            raise AssertionError("not called in these tests")

    # The factory patch path imports and rewrites these modules.
    fused_moe_layer.FusedMoE = _OrigFactory()
    fused_moe_layer.FusedMoEFactory = fused_moe_layer.FusedMoE
    fused_moe.FusedMoE = fused_moe_layer.FusedMoE
    fused_moe.FusedMoEFactory = fused_moe_layer.FusedMoE
    oracle_unquantized.select_unquantized_moe_backend = lambda *a, **k: None
    unquantized_method.select_unquantized_moe_backend = (
        oracle_unquantized.select_unquantized_moe_backend
    )

    # Parent-attribute wiring so `import vllm.platforms` style imports work.
    vllm.platforms = platforms
    vllm.model_executor = me
    me.layers = layers
    layers.fused_moe = fused_moe
    me.custom_op = custom_op
    layers.activation = activation_mod
    layers.layernorm = layernorm_mod
    layers.rotary_embedding = rotary_mod
    fused_moe.layer = fused_moe_layer
    fused_moe.oracle = _make_module("vllm.model_executor.layers.fused_moe.oracle")
    fused_moe.oracle.unquantized = oracle_unquantized
    fused_moe.unquantized_fused_moe_method = unquantized_method

    for name, module in {
        "vllm": vllm,
        "vllm.platforms": platforms,
        "vllm.logger": logger_mod,
        "vllm.envs": envs_mod,
        "vllm.model_executor": me,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.layers.fused_moe": fused_moe,
        "vllm.model_executor.layers.fused_moe.config": fm_config,
        "vllm.model_executor.layers.fused_moe.runner.moe_runner": fm_runner,
        "vllm.model_executor.layers.fused_moe.modular_kernel": fm_modular,
        "vllm.model_executor.layers.fused_moe.activation": fm_activation,
        "vllm.model_executor.layers.fused_moe.experts": fm_experts_dir,
        "vllm.model_executor.layers.fused_moe.experts.triton_moe": fm_experts_triton,
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe": fm_experts_aiter,
        "vllm.model_executor.layers.fused_moe.router": router_dir,
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router": router_fused,
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router": (
            router_bias
        ),
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router": (
            router_grouped
        ),
        "vllm.model_executor.layers.fused_moe.fused_moe": fm_fused,
        "vllm.model_executor.layers.fused_moe.utils": fm_utils,
        "vllm._aiter_ops": aiter_ops,
        "vllm.triton_utils": triton_utils,
        "vllm.model_executor.custom_op": custom_op,
        "vllm.model_executor.layers.activation": activation_mod,
        "vllm.model_executor.layers.layernorm": layernorm_mod,
        "vllm.model_executor.layers.rotary_embedding": rotary_mod,
        "vllm.model_executor.layers.fused_moe.layer": fused_moe_layer,
        "vllm.model_executor.layers.fused_moe.oracle": fused_moe.oracle,
        "vllm.model_executor.layers.fused_moe.oracle.unquantized": oracle_unquantized,
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method": (
            unquantized_method
        ),
    }.items():
        sys.modules[name] = module
    return registry


def _ensure_imported():
    """Import custom_ops once against the stub graph; return (module, registry).

    If a prior file in this session left a smaller vllm stub graph in
    place (without the model_executor.layers packages), re-install the full
    one before importing; an already-imported custom_ops keeps working
    either way because the fixture restored its original import state.
    """
    if "vllm_fl.ops.custom_ops" not in sys.modules:
        _install_vllm_graph()
    from vllm.model_executor.custom_op import CustomOp

    import vllm_fl.ops.custom_ops as custom_ops

    return custom_ops, CustomOp.registry


def _reset_registry(registry):
    """Clear recorded registrations between tests."""
    registry.custom_ops.clear()
    registry.pluggable_layers.clear()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, clean_fl_env):
    """Keep every OOT / FlagGems env var unset around each test."""
    for var in (
        "VLLM_FL_OOT_WHITELIST",
        "VLLM_FL_OOT_BLACKLIST",
        "VLLM_FL_OOT_ENABLED",
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_FL_PREFER",
        "USE_FLAGGEMS",
        "VLLM_FL_PLATFORM",
        "GEMS_VENDOR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("USE_FLAGGEMS", "0")
    yield


@pytest.fixture
def ops_registry(monkeypatch):
    """Import custom_ops against the stub graph and reset the recorder."""
    custom_ops, registry = _ensure_imported()
    # _patch_fused_moe_factory walks the whole sys.modules table and calls
    # vars() on every entry; other supplementary files install stub modules
    # without a __dict__ (and real C-extension modules lack one too), which
    # raises TypeError from inside register_oot_ops. Neutralize the factory
    # walk for every test here — the oracle/factory patch itself is covered
    # by #799's TestPatchUnquantizedMoeOracle.
    monkeypatch.setattr(
        custom_ops, "_patch_fused_moe_factory", lambda: None, raising=False
    )
    _reset_registry(registry)
    yield registry
    _reset_registry(registry)


class TestVendorPatchBranches:
    """register_oot_ops device_type / vendor dispatch to vendor patches."""

    def test_fused_moe_not_registered_patches_oracle(
        self, monkeypatch, ops_registry
    ):
        """'fused_moe' excluded from registration patches the oracle."""
        custom_ops, _ = _ensure_imported()
        patched = []
        monkeypatch.setattr(
            custom_ops,
            "_patch_unquantized_moe_oracle",
            lambda: patched.append(True),
        )
        # Whitelisting only silu leaves fused_moe out of ops_to_register.
        monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", "silu_and_mul")
        custom_ops.register_oot_ops()
        assert patched == [True]

    def test_npu_device_type_applies_ascend_patches(
        self, monkeypatch, ops_registry
    ):
        """device_type npu runs apply_ascend_patches inside the op loop."""
        custom_ops, _ = _ensure_imported()
        from vllm.platforms import current_platform

        calls = []
        ascend_patch_mod = _make_module(
            "vllm_fl.dispatch.backends.vendor.ascend.patch"
        )
        ascend_patch_mod.apply_ascend_patches = lambda: calls.append("ascend")
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.vendor.ascend.patch",
            ascend_patch_mod,
        )
        monkeypatch.setattr(
            current_platform, "device_type", "npu", raising=False
        )
        try:
            monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", "silu_and_mul")
            custom_ops.register_oot_ops()
            assert calls == ["ascend"]
        finally:
            monkeypatch.setattr(
                current_platform, "device_type", "cuda", raising=False
            )

    def test_ptpu_device_type_applies_sunrise_patches(
        self, monkeypatch, ops_registry
    ):
        """device_type ptpu runs apply_sunrise_patches inside the op loop."""
        custom_ops, _ = _ensure_imported()
        from vllm.platforms import current_platform

        calls = []
        sunrise_patch_mod = _make_module(
            "vllm_fl.dispatch.backends.vendor.sunrise.patch"
        )
        sunrise_patch_mod.apply_sunrise_patches = lambda: calls.append("sunrise")
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.vendor.sunrise.patch",
            sunrise_patch_mod,
        )
        monkeypatch.setattr(
            current_platform, "device_type", "ptpu", raising=False
        )
        try:
            monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", "silu_and_mul")
            custom_ops.register_oot_ops()
            assert calls == ["sunrise"]
        finally:
            monkeypatch.setattr(
                current_platform, "device_type", "cuda", raising=False
            )

    def test_gcu_vendor_applies_gcu_patches(self, monkeypatch, ops_registry):
        """vendor gcu runs apply_gcu_patches inside the op loop."""
        custom_ops, _ = _ensure_imported()
        from vllm.platforms import current_platform

        calls = []
        gcu_patch_mod = _make_module("vllm_fl.dispatch.backends.vendor.gcu.patch")
        gcu_patch_mod.apply_gcu_patches = lambda: calls.append("gcu")
        monkeypatch.setitem(
            sys.modules, "vllm_fl.dispatch.backends.vendor.gcu.patch", gcu_patch_mod
        )
        monkeypatch.setattr(
            current_platform, "vendor_name", "gcu", raising=False
        )
        try:
            monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", "silu_and_mul")
            custom_ops.register_oot_ops()
            assert calls == ["gcu"]
        finally:
            monkeypatch.setattr(
                current_platform, "vendor_name", "nvidia", raising=False
            )

    def test_kunlunxin_platform_applies_patches_regardless(
        self, monkeypatch, ops_registry
    ):
        """Kunlunxin patches run even when no OOT operator was selected."""
        custom_ops, _ = _ensure_imported()
        from vllm.platforms import current_platform

        # get_platform_name reads current_platform.vendor_name; steer the
        # stub platform to kunlunxin for the duration of the call.
        monkeypatch.setattr(
            current_platform, "vendor_name", "kunlunxin", raising=False
        )
        kx_patch_mod = _make_module(
            "vllm_fl.dispatch.backends.vendor.kunlunxin.patch"
        )
        calls = []
        kx_patch_mod.apply_kunlunxin_patches = lambda: calls.append("kunlunxin")
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.vendor.kunlunxin.patch",
            kx_patch_mod,
        )
        # OOT registration must stay enabled (the disabled branch returns
        # before the kunlunxin block); pass a whitelist naming only an
        # unknown op so the op loop registers nothing yet still reaches the
        # platform block afterwards.
        monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "1")
        monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", "no_such_op")
        monkeypatch.setattr(
            custom_ops, "_patch_unquantized_moe_oracle", lambda: None
        )
        custom_ops.register_oot_ops()
        assert calls == ["kunlunxin"]
        assert ops_registry.custom_ops == []

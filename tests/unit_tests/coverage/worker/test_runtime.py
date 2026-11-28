# Copyright (c) 2025 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl.worker.runtime._initialize_fl_runtime``.

The runtime initializer is the FlagGems bootstrap shared by every vendor
worker.  It cannot import on CPU-only machines because it pulls the real
``vllm`` and ``flag_gems`` packages at import time, so these tests install
minimal *external-boundary* stubs (fake ``vllm`` module tree, fake
``flag_gems``) in ``sys.modules`` and then drive the real function:

* ``USE_FLAGGEMS=False`` returns before flag_gems is touched;
* whitelist  -> ``flag_gems.only_enable(include=..., ...)``;
* blacklist  -> ``flag_gems.enable(unused=...)`` on non-AMD platforms;
* blacklist  -> AMD's ``_enable_except`` derives an explicit include list
  from ``flag_gems._FULL_CONFIG`` (or falls back to ``enable(unused=...)``
  when that internal table is unavailable);
* rank 0 records the operator list, other ranks do not.
"""

import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _VllmStubTree:
    """Install/restore a fake ``vllm`` module tree in ``sys.modules``.

    Only the names ``vllm_fl`` actually imports are provided; the platform
    object is configurable so tests can select the vendor branch.  All
    installed entries are recorded for teardown.
    """

    def __init__(self, vendor_name="nvidia", device_type="cuda"):
        self._installed = []
        self._previous = {}
        self.mm_encoder = None
        self.custom_op_cls = None
        self.platform = SimpleNamespace(
            vendor_name=vendor_name,
            device_type=device_type,
            is_cuda=lambda: False,
            is_tpu=lambda: False,
            is_cpu=lambda: False,
            is_rocm=lambda: False,
            is_out_of_tree=lambda: False,
            is_device_capability_family=lambda family: False,
        )
        self._build()

    def _put(self, name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        if name not in self._previous:
            self._previous[name] = sys.modules.get(name)
        sys.modules[name] = module
        self._installed.append(name)
        return module

    def _build(self):
        self._put("vllm")
        self._put("vllm.logger", init_logger=logging.getLogger)
        self._put("vllm.platforms", current_platform=self.platform)
        self._put("vllm.envs")
        self._put("vllm.triton_utils", tl=object(), triton=object())
        self._put("vllm._aiter_ops", rocm_aiter_ops=object())
        self._put("vllm.model_executor")

        class _CustomOp:
            register_oot = MagicMock()

        class _PluggableLayer:
            register_oot = MagicMock()

        self.custom_op_cls = _CustomOp
        self._put(
            "vllm.model_executor.custom_op",
            CustomOp=_CustomOp,
            PluggableLayer=_PluggableLayer,
        )
        self._put("vllm.model_executor.layers")
        self._put(
            "vllm.model_executor.layers.activation",
            SiluAndMul=type("SiluAndMul", (), {}),
            GeluAndMul=type("GeluAndMul", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.layernorm",
            RMSNorm=type("RMSNorm", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.rotary_embedding",
            RotaryEmbedding=type("RotaryEmbedding", (), {}),
        )

        class _MMEncoderAttention:
            def forward_native(self):
                pass

            def forward_cuda(self):
                pass

            def forward_oot(self):
                pass

        self._put("vllm.model_executor.layers.attention")
        self.mm_encoder = self._put(
            "vllm.model_executor.layers.attention.mm_encoder_attention",
            MMEncoderAttention=_MMEncoderAttention,
            maybe_get_vit_flash_attn_backend=lambda backend: None,
        )
        self._put("vllm.v1")
        self._put("vllm.v1.attention")
        self._put("vllm.v1.attention.backends")
        self._put(
            "vllm.v1.attention.backends.registry",
            AttentionBackendEnum=SimpleNamespace(
                FLASH_ATTN=object(), ROCM_AITER_FA=object(), TORCH_SDPA=object()
            ),
        )
        # fused_moe subpackage names imported by vllm_fl.ops.*
        self.fused_moe_pkg = self._put(
            "vllm.model_executor.layers.fused_moe",
            FusedMoEFactory=type("FusedMoEFactory", (), {}),
            FusedMoE=type("FusedMoE", (), {}),
        )
        self.fused_moe_layer = self._put(
            "vllm.model_executor.layers.fused_moe.layer",
            FusedMoEFactory=type("UpstreamFactory", (), {}),
            FusedMoE=type("UpstreamMoE", (), {}),
        )
        self._put("vllm.model_executor.layers.fused_moe.runner")
        self._put(
            "vllm.model_executor.layers.fused_moe.runner.moe_runner",
            MoERunner=type("MoERunner", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.modular_kernel",
            FusedMoEExperts=type("FusedMoEExperts", (), {}),
            FusedMoEActivationFormat=SimpleNamespace(
                BatchedExperts=object(), Standard=object()
            ),
            ExpertTokensMetadata=type("ExpertTokensMetadata", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.config",
            FusedMoEConfig=type("FusedMoEConfig", (), {}),
            FUSED_MOE_UNQUANTIZED_CONFIG={},
            FusedMoEQuantConfig=type("FusedMoEQuantConfig", (), {}),
            _get_config_dtype_str=lambda *a, **k: "fp16",
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
            UnquantizedFusedMoEMethod=type("UnquantizedFusedMoEMethod", (), {}),
            select_unquantized_moe_backend=object(),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.activation",
            ApplyMoEActivationConfig=type("ApplyMoEActivationConfig", (), {}),
            MoEActivation=type("MoEActivation", (), {}),
            apply_moe_activation=lambda *a, **k: None,
        )
        self._put("vllm.model_executor.layers.fused_moe.experts")
        self._put(
            "vllm.model_executor.layers.fused_moe.experts.triton_moe",
            TritonExperts=type("TritonExperts", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
            rocm_aiter_grouped_topk=lambda *a, **k: None,
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.fused_moe",
            try_get_optimal_moe_config=lambda *a, **k: None,
            _get_config_quant_dtype=lambda *a, **k: None,
        )
        ume_backend = type(
            "UnquantizedMoeBackend",
            (),
            {
                "AITER": object(),
                "TRITON": object(),
                "BATCHED_TRITON": object(),
                "TPU": object(),
                "FLASHINFER_TRTLLM": object(),
                "FLASHINFER_CUTLASS": object(),
            },
        )
        self._put("vllm.model_executor.layers.fused_moe.oracle")
        self._put(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized",
            UnquantizedMoeBackend=ume_backend,
            _trtllm_bf16_lora_supported=lambda *a, **k: True,
            backend_to_kernel_cls={},
            map_unquantized_backend=lambda *a, **k: None,
            select_unquantized_moe_backend=object(),
        )
        self._put("vllm.model_executor.layers.fused_moe.router")
        self._put(
            "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
            FusedTopKBiasRouter=type("FusedTopKBiasRouter", (), {}),
            fused_topk_bias=lambda *a, **k: None,
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.router.fused_topk_router",
            FusedTopKRouter=type("FusedTopKRouter", (), {}),
            _get_padding_mask=lambda *a, **k: None,
            fused_topk=lambda *a, **k: None,
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
            GroupedTopKRouter=type("GroupedTopKRouter", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.utils",
            _resize_cache=lambda *a, **k: None,
            moe_kernel_quantize_input=lambda *a, **k: None,
        )

    def restore(self):
        """Remove/restore every sys.modules entry this tree installed."""
        for name in self._installed:
            previous = self._previous[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        self._installed.clear()
        self._previous.clear()


class _FlagGemsStub:
    """Fake ``flag_gems`` module recording every configuration call."""

    def __init__(self, full_config=None):
        self.module = ModuleType("flag_gems")
        self.calls = []
        self.recorded = {}

        def recorder(name):
            def fn(*args, **kwargs):
                self.calls.append((name, args, kwargs))

            return fn

        self.module.only_enable = recorder("only_enable")
        self.module.enable = recorder("enable")
        if full_config is not None:
            # Real flag_gems._FULL_CONFIG is a sequence of (op_name, ...) rows;
            # the AMD path reads entry[0] from each row.
            self.module._FULL_CONFIG = full_config
        sys.modules["flag_gems"] = self.module

    def restore(self):
        """Drop the fake flag_gems entry."""
        sys.modules.pop("flag_gems", None)

    def method_names(self):
        """Return the ordered list of configuration methods invoked."""
        return [name for name, _, _ in self.calls]


@pytest.fixture
def vllm_stubs(fl_sys_modules):
    """Yield a builder that installs (and auto-restores) the vllm stub tree."""
    trees = []

    def _install(vendor_name="nvidia", device_type="cuda"):
        tree = _VllmStubTree(vendor_name, device_type)
        trees.append(tree)
        return tree

    yield _install
    for tree in reversed(trees):
        tree.restore()


@pytest.fixture
def flag_gems_stub(fl_sys_modules):
    """Yield a builder that installs (and auto-restores) a fake flag_gems."""
    gems = []

    def _install(full_config=None):
        gem = _FlagGemsStub(full_config)
        gems.append(gem)
        return gem

    yield _install
    for gem in reversed(gems):
        gem.restore()


@pytest.fixture(autouse=True)
def _runtime_env(monkeypatch):
    """Neutralize the env vars that steer the runtime under test."""
    for key in (
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_FL_PREFER",
        "USE_FLAGGEMS",
        "VLLM_FL_OOT_ENABLED",
        "VLLM_FL_OOT_WHITELIST",
        "VLLM_FL_OOT_BLACKLIST",
        "FLAGGEMS_ENABLE_OPLIST_PATH",
        "VLLM_FL_CONFIG",
        "VLLM_FL_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def _fresh_runtime(vllm_stubs, vendor_name="nvidia", device_type="cuda"):
    """Install stubs and import the real runtime module against them.

    Returns ``(runtime_module, stub_tree)`` so tests can also inspect the
    fake vllm modules the initializer mutated.
    """
    tree = vllm_stubs(vendor_name=vendor_name, device_type=device_type)
    import importlib

    import vllm_fl.worker.runtime as runtime

    return importlib.reload(runtime), tree


class TestInitializeFlRuntime:
    """Branch matrix of ``_initialize_fl_runtime``."""

    def test_flaggems_disabled_skips_flag_gems_entirely(
        self, vllm_stubs, flag_gems_stub, monkeypatch
    ):
        """USE_FLAGGEMS=0 returns before flag_gems is ever imported."""
        monkeypatch.setenv("USE_FLAGGEMS", "0")
        gems = flag_gems_stub()
        runtime, tree = _fresh_runtime(vllm_stubs)
        runtime._initialize_fl_runtime(0)
        assert gems.calls == []
        # The OOT registration half still ran: the mm-encoder patch replaced
        # the stubbed vllm attention helper module-level function.
        assert (
            tree.mm_encoder.maybe_get_vit_flash_attn_backend.__name__
            == "_patched_maybe_get_vit_flash_attn_backend"
        )


class TestWhitelistBranch:
    """``VLLM_FL_FLAGOS_WHITELIST`` routes to only_enable with include=."""

    def test_whitelist_calls_only_enable_with_record_on_rank0(
        self, vllm_stubs, flag_gems_stub, monkeypatch
    ):
        """Whitelisted ops are passed via include= and recorded on rank 0."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "op_a, op_b")
        gems = flag_gems_stub()
        runtime, _ = _fresh_runtime(vllm_stubs)
        runtime._initialize_fl_runtime(0)
        assert gems.method_names() == ["only_enable"]
        _, kwargs = gems.calls[0][1:]
        assert kwargs["include"] == ["op_a", "op_b"]
        assert kwargs["record"] is True
        assert kwargs["once"] is True

    def test_whitelist_on_nonzero_rank_disables_recording(
        self, vllm_stubs, flag_gems_stub, monkeypatch
    ):
        """Only rank 0 records the enabled-op list."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "op_a")
        gems = flag_gems_stub()
        runtime, _ = _fresh_runtime(vllm_stubs)
        runtime._initialize_fl_runtime(3)
        _, kwargs = gems.calls[0][1:]
        assert kwargs["record"] is False

    def test_flaggems_oplist_path_is_forwarded(
        self, vllm_stubs, flag_gems_stub, monkeypatch
    ):
        """FLAGGEMS_ENABLE_OPLIST_PATH is forwarded as path=."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "op_a")
        monkeypatch.setenv("FLAGGEMS_ENABLE_OPLIST_PATH", "my/oplist.txt")
        gems = flag_gems_stub()
        runtime, _ = _fresh_runtime(vllm_stubs)
        runtime._initialize_fl_runtime(0)
        _, kwargs = gems.calls[0][1:]
        assert kwargs["path"] == "my/oplist.txt"


class TestBlacklistBranch:
    """``VLLM_FL_FLAGOS_BLACKLIST`` behaviour per platform vendor."""

    def test_non_amd_platform_uses_enable_unused(
        self, vllm_stubs, flag_gems_stub, monkeypatch
    ):
        """A non-AMD vendor keeps the proven enable(unused=...) call."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "op_x,op_y")
        gems = flag_gems_stub()
        runtime, _ = _fresh_runtime(vllm_stubs, vendor_name="nvidia")
        runtime._initialize_fl_runtime(0)
        assert gems.method_names() == ["enable"]
        _, kwargs = gems.calls[0][1:]
        assert kwargs["unused"] == ["op_x", "op_y"]
        assert kwargs["record"] is True

    def test_amd_platform_derives_explicit_include_list(
        self, vllm_stubs, flag_gems_stub, monkeypatch
    ):
        """AMD derives include= from _FULL_CONFIG minus the blacklist."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "op_x,op_y")
        full_config = [("op_a", 1), ("op_x", 2), ("op_b", 3), ("op_y", 4)]
        gems = flag_gems_stub(full_config=full_config)
        runtime, _ = _fresh_runtime(vllm_stubs, vendor_name="amd")
        runtime._initialize_fl_runtime(0)
        assert gems.method_names() == ["only_enable"]
        _, kwargs = gems.calls[0][1:]
        assert kwargs["include"] == ["op_a", "op_b"]

    def test_amd_blacklist_entry_absent_from_full_config(
        self, vllm_stubs, flag_gems_stub, monkeypatch, caplog
    ):
        """Blacklist names missing from _FULL_CONFIG are skipped safely."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "op_a,missing_op")
        full_config = [("op_a", 1), ("op_b", 2), ("op_c", 3)]
        gems = flag_gems_stub(full_config=full_config)
        runtime, _ = _fresh_runtime(vllm_stubs, vendor_name="amd")
        with caplog.at_level(logging.WARNING, logger="vllm_fl.worker.runtime"):
            runtime._initialize_fl_runtime(0)
        assert gems.method_names() == ["only_enable"]
        _, kwargs = gems.calls[0][1:]
        assert kwargs["include"] == ["op_b", "op_c"]
        assert any("missing_op" in rec.message for rec in caplog.records)

    def test_amd_falls_back_when_full_config_unavailable(
        self, vllm_stubs, flag_gems_stub, monkeypatch, caplog
    ):
        """Without _FULL_CONFIG AMD degrades to enable(unused=...)."""
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "op_x")
        gems = flag_gems_stub(full_config=None)
        runtime, _ = _fresh_runtime(vllm_stubs, vendor_name="amd")
        with caplog.at_level(logging.ERROR, logger="vllm_fl.worker.runtime"):
            runtime._initialize_fl_runtime(0)
        assert gems.method_names() == ["enable"]
        _, kwargs = gems.calls[0][1:]
        assert kwargs["unused"] == ["op_x"]
        assert any("_FULL_CONFIG unavailable" in rec.message for rec in caplog.records)


class TestEnableAllBranch:
    """Neither whitelist nor blacklist: enable everything."""

    def test_no_lists_enables_all_ops(self, vllm_stubs, flag_gems_stub):
        """With both lists unset only plain enable() is invoked.

        The stub platform uses an unknown vendor name so no platform YAML
        contributes a default flagos_blacklist, isolating the pure
        "no lists configured" branch of the initializer.
        """
        gems = flag_gems_stub()
        runtime, _ = _fresh_runtime(vllm_stubs, vendor_name="unknown_vendor")
        runtime._initialize_fl_runtime(0)
        assert gems.method_names() == ["enable"]
        _, kwargs = gems.calls[0][1:]
        assert "include" not in kwargs
        assert "unused" not in kwargs
        assert kwargs["record"] is True

    def test_platform_default_blacklist_uses_enable_unused(
        self, vllm_stubs, flag_gems_stub
    ):
        """The nvidia platform config's built-in blacklist drives enable()."""
        gems = flag_gems_stub()
        runtime, _ = _fresh_runtime(vllm_stubs, vendor_name="nvidia")
        runtime._initialize_fl_runtime(0)
        assert gems.method_names() == ["enable"]
        _, kwargs = gems.calls[0][1:]
        assert "include" not in kwargs
        # nvidia.yaml ships a non-empty default flagos_blacklist.
        assert kwargs["unused"]
        assert "pad" in kwargs["unused"]


class TestEnableExcept:
    """Direct tests of the AMD-only ``_enable_except`` helper."""

    def test_include_list_preserves_full_config_order(self, vllm_stubs, flag_gems_stub):
        """Ordering of the derived include list follows _FULL_CONFIG order."""
        gems = flag_gems_stub(full_config=[("op_a", 1), ("op_b", 2), ("op_c", 3)])
        runtime, _ = _fresh_runtime(vllm_stubs)
        common = {"record": True, "once": True, "path": "p.txt"}
        runtime._enable_except(gems.module, ["op_b"], common)
        assert gems.method_names() == ["only_enable"]
        _, kwargs = gems.calls[0][1:]
        assert kwargs["include"] == ["op_a", "op_c"]
        assert kwargs["path"] == "p.txt"

    def test_missing_full_config_falls_back_to_unused(
        self, vllm_stubs, flag_gems_stub, caplog
    ):
        """AttributeError from _FULL_CONFIG triggers the unused= fallback."""
        gems = flag_gems_stub(full_config=None)
        runtime, _ = _fresh_runtime(vllm_stubs)
        common = {"record": False, "once": True, "path": "p.txt"}
        with caplog.at_level(logging.ERROR, logger="vllm_fl.worker.runtime"):
            runtime._enable_except(gems.module, ["op_z"], common)
        assert gems.method_names() == ["enable"]
        _, kwargs = gems.calls[0][1:]
        assert kwargs["unused"] == ["op_z"]
        assert kwargs["record"] is False


class TestOotRegistrationStillRuns:
    """The pre-flag_gems half of the initializer must not be skipped."""

    def test_oot_ops_registered_and_fused_moe_patched(self, vllm_stubs, monkeypatch):
        """register_oot_ops replaces the upstream FusedMoE factory."""
        monkeypatch.setenv("USE_FLAGGEMS", "0")
        tree = vllm_stubs()
        import importlib

        import vllm_fl.worker.runtime as runtime

        runtime = importlib.reload(runtime)
        runtime._initialize_fl_runtime(0)
        import vllm_fl.ops.custom_ops as custom_ops

        assert custom_ops.FusedMoEFL is custom_ops.FusedMoEFactoryFL
        # The factory replacement was applied to both upstream spellings.
        assert tree.fused_moe_layer.FusedMoEFactory is custom_ops.FusedMoEFactoryFL
        assert tree.fused_moe_layer.FusedMoE is custom_ops.FusedMoEFactoryFL
        assert tree.fused_moe_pkg.FusedMoE is custom_ops.FusedMoEFactoryFL

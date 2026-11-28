# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for ``PlatformFL.check_and_update_config``.

``check_and_update_config`` (vllm_fl/platform.py lines 372-730) is the
per-vendor configuration gate vLLM invokes before building an engine.
This suite drives every branch of that classmethod through the same
stub-vllm loading pattern as ``test_platform_fl_methods.py``: the module
under test is loaded via ``spec_from_file_location`` against a stub vllm
graph, the class attributes ``device_type`` / ``vendor_name`` /
``dist_backend`` are monkeypatched per test, and a ``_VllmConfig`` double
carries the sub-configs the method reads and mutates. Vendor-only lazy
imports (``vllm_fl.attention.utils``, the ascend/iluvatar vendor modules)
are stubbed through ``sys.modules`` entries that are NOT managed by the
``platform_env`` fixture, so ``monkeypatch.setitem`` restores the real
module (or nothing) on teardown without leaking stubs.

Observable effects asserted per test: raised ``ValueError`` messages,
mutated config fields, and calls recorded by stub functions.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_PLATFORM_SOURCE = _ROOT / "vllm_fl" / "platform.py"
_PLATFORM_MODULE = "supp_platform_fl_checkcfg_under_test"

_STUB_ENTRIES = [
    "vllm",
    "vllm.logger",
    "vllm.platforms",
    "vllm.platforms.interface",
    "vllm.v1.attention.backends.registry",
    "vllm.config",
    "vllm.config.compilation",
    "vllm_fl.utils",
]


class _FakeLogger:
    """``vllm.logger.init_logger`` stand-in recording level calls."""

    def __init__(self):
        self.records = []

    def _log(self, level, msg, *args):
        self.records.append((level, msg % args if args else msg))

    def info(self, msg, *args):
        self._log("info", msg, *args)

    def debug(self, msg, *args):
        self._log("debug", msg, *args)

    def warning(self, msg, *args):
        self._log("warning", msg, *args)

    def info_once(self, msg, *args):
        self._log("info_once", msg, *args)

    def warning_once(self, msg, *args):
        self._log("warning_once", msg, *args)


def _make_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _CUDAGraphModeEnum:
    """Enum double for ``vllm.config.CUDAGraphMode``.

    Members are instances (assigned after the class body) so that values
    the method assigns from ``CUDAGraphMode.PIECEWISE`` keep the helper
    methods ``has_full_cudagraphs`` / ``has_piecewise_cudagraphs``.
    ``__eq__`` also accepts the raw member string so comparisons against
    class attributes work whichever side the double is on.
    """

    def __init__(self, value):
        self.value = value

    def has_full_cudagraphs(self):
        return self.value in ("full", "full_decode_only")

    def has_piecewise_cudagraphs(self):
        return self.value in ("piecewise", "full")

    def __eq__(self, other):
        if isinstance(other, _CUDAGraphModeEnum):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"_CUDAGraphModeEnum({self.value!r})"


_CUDAGraphModeEnum.NONE = _CUDAGraphModeEnum("none")
_CUDAGraphModeEnum.PIECEWISE = _CUDAGraphModeEnum("piecewise")
_CUDAGraphModeEnum.FULL = _CUDAGraphModeEnum("full")
_CUDAGraphModeEnum.FULL_DECODE_ONLY = _CUDAGraphModeEnum("full_decode_only")


class _CompilationModeEnum:
    """Enum double for ``vllm.config.compilation.CompilationMode``."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, _CompilationModeEnum):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"_CompilationModeEnum({self.value!r})"


_CompilationModeEnum.NONE = _CompilationModeEnum("NONE")
_CompilationModeEnum.VLLM_COMPILE = _CompilationModeEnum("VLLM_COMPILE")


class _DeviceInfoStub:
    """Vendor-neutral DeviceInfo double: probes report nothing."""

    vendor_name = ""
    device_type = ""
    device = SimpleNamespace(__class__=SimpleNamespace(__name__="NoDevice"))
    dispatch_key = "CUDA"
    torch_device_fn = torch.cuda


@pytest.fixture
def platform_env(monkeypatch):
    """Load platform.py against a stub vllm graph; yield (module, logger)."""
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setenv("GEMS_VENDOR", "nvidia")

    logger = _FakeLogger()
    saved = {name: sys.modules.get(name) for name in _STUB_ENTRIES}

    class _Capability:
        def __init__(self, major=8, minor=0):
            self.major = major
            self.minor = minor

        def as_version_str(self):
            return f"{self.major}.{self.minor}"

    class _PlatformEnum:
        CUDA = "cuda"
        OOT = "oot"
        CPU = "cpu"

    class _BasePlatform:
        """Stub of vllm.platforms.Platform: records super() delegations."""

        dist_backend = "nccl"
        super_calls = []

        @classmethod
        def import_kernels(cls):
            cls.super_calls.append("import_kernels")

        @classmethod
        def get_compile_backend(cls):
            cls.super_calls.append("get_compile_backend")
            return "super-backend"

        @classmethod
        def _align_hybrid_block_size(cls, vllm_config, backend_cls):
            cls.super_calls.append("_align_hybrid_block_size")

        @classmethod
        def get_device_capability(cls, device_id=0):
            return _Capability()

        @classmethod
        def has_device_capability(cls, major):
            return major <= 8

        @classmethod
        def is_device_capability(cls, major):
            return major == 8

        @classmethod
        def is_device_capability_family(cls, family):
            return family == 80

    class _AttentionBackendEnum:
        TORCH_SDPA = "torch_sdpa"
        FLASH_ATTN = "flash_attn"

        def get_class(self):  # pragma: no cover - not reached in these tests
            raise ImportError("no backend class")

    vllm_pkg = _make_module("vllm")
    logger_mod = _make_module("vllm.logger", init_logger=lambda name: logger)
    platforms_mod = _make_module(
        "vllm.platforms", Platform=_BasePlatform, PlatformEnum=_PlatformEnum
    )
    interface_mod = _make_module(
        "vllm.platforms.interface", DeviceCapability=_Capability
    )
    registry_mod = _make_module(
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=_AttentionBackendEnum,
    )
    config_mod = _make_module("vllm.config")
    config_mod.CUDAGraphMode = _CUDAGraphModeEnum
    compilation_mod = _make_module("vllm.config.compilation")
    compilation_mod.CompilationMode = _CompilationModeEnum

    utils_stub = _make_module("vllm_fl.utils")
    utils_stub.DeviceInfo = _DeviceInfoStub
    utils_stub._env_vendor_name = lambda: "nvidia"
    utils_stub.get_device_control_env_var = lambda vendor: "CUDA_VISIBLE_DEVICES"
    utils_stub.get_device_type = lambda vendor: "cuda"
    # vllm_fl/__init__.py (re)imports get_op_config from vllm_fl.utils when the
    # platform module under test pulls the gcu compilation config (upstream #675
    # enflame S60 added that import chain); the loader resolves the stub tree.
    utils_stub.get_op_config = lambda: None

    stubs = {
        "vllm": vllm_pkg,
        "vllm.logger": logger_mod,
        "vllm.platforms": platforms_mod,
        "vllm.platforms.interface": interface_mod,
        "vllm.v1.attention.backends.registry": registry_mod,
        "vllm.config": config_mod,
        "vllm.config.compilation": compilation_mod,
        "vllm_fl.utils": utils_stub,
    }
    # upstream #675 (enflame S60): gcu/enflame vendors now import
    # vllm_fl.dispatch.backends.vendor.gcu.compilation_config (which pulls
    # vllm.third_party.flash_linear_attention) and a GCU worker module —
    # stub them so the new branch stays loadable on hosts without the
    # vendor packages.
    gcu_compilation_stub = _make_module(
        "vllm_fl.dispatch.backends.vendor.gcu.compilation_config"
    )
    gcu_compilation_stub.configure_codegen_cache = lambda compilation_config: None
    fla_stub = _make_module("vllm.third_party")
    fla_fla_stub = _make_module("vllm.third_party.flash_linear_attention")
    gcu_worker_stub = _make_module("vllm_fl.dispatch.backends.vendor.gcu.worker.worker")
    gcu_worker_stub.WorkerGCU = type("WorkerGCU", (), {})
    stubs.update(
        {
            "vllm_fl.dispatch.backends.vendor.gcu.compilation_config": gcu_compilation_stub,
            "vllm.third_party": fla_stub,
            "vllm.third_party.flash_linear_attention": fla_fla_stub,
            "vllm_fl.dispatch.backends.vendor.gcu.worker.worker": gcu_worker_stub,
        }
    )
    for name, mod in stubs.items():
        sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location(_PLATFORM_MODULE, _PLATFORM_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PLATFORM_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_PLATFORM_MODULE, None)
        for name in _STUB_ENTRIES:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]
        raise
    yield SimpleNamespace(module=module, logger=logger, base=_BasePlatform)
    sys.modules.pop(_PLATFORM_MODULE, None)
    for name in _STUB_ENTRIES:
        if saved[name] is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved[name]


# ---------------------------------------------------------------------------
# Config doubles
# ---------------------------------------------------------------------------


def make_compilation(
    mode: Any = "NONE",
    cudagraph_mode: Any = None,
    backend: Any = None,
    compile_sizes: Any = None,
    use_inductor_graph_partition: bool = False,
    splitting_ops: list | None = None,
    cudagraph_capture_sizes: Any = None,
) -> "_RecordingSpec":
    """Build a compilation-config double for check_and_update_config."""
    return _RecordingSpec(
        mode=mode,
        cudagraph_mode=cudagraph_mode,
        backend=backend,
        compile_sizes=compile_sizes,
        use_inductor_graph_partition=use_inductor_graph_partition,
        splitting_ops=splitting_ops,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )


class _RecordingSpec:
    """Compilation-config double recording splitting-ops interactions."""

    def __init__(
        self,
        mode="NONE",
        cudagraph_mode=None,
        backend=None,
        compile_sizes=None,
        use_inductor_graph_partition=False,
        splitting_ops=None,
        cudagraph_capture_sizes=None,
    ):
        if isinstance(mode, str):
            mode = _CompilationModeEnum(mode)
        if cudagraph_mode is None:
            cudagraph_mode = _CUDAGraphModeEnum("none")
        elif isinstance(cudagraph_mode, str):
            cudagraph_mode = _CUDAGraphModeEnum(cudagraph_mode)
        self.mode = mode
        self.cudagraph_mode = cudagraph_mode
        self.backend = backend
        self.compile_sizes = compile_sizes
        self.use_inductor_graph_partition = use_inductor_graph_partition
        self.splitting_ops = list(splitting_ops or [])
        self.cudagraph_capture_sizes = cudagraph_capture_sizes
        self.inductor_compile_config = {}
        self.splitting_calls = []

    def set_splitting_ops_for_v1(self, all2all_backend=None, data_parallel_size=None):
        self.splitting_calls.append((all2all_backend, data_parallel_size))


def make_vllm_config(**overrides: Any) -> SimpleNamespace:
    """Build a VllmConfig double with happy-path defaults (cuda/nvidia).

    Every sub-config check_and_update_config touches is present with a
    benign default; tests override just the fields their branch needs
    (either wholesale via kwargs or by mutating the sub-namespaces).
    """
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_diffusion=False,
            is_hybrid=False,
            is_attention_free=False,
            is_mm_prefix_lm=False,
            is_moe=False,
            use_mla=False,
            disable_cascade_attn=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            world_size_across_dp=None,
            disable_custom_all_reduce=False,
            all2all_backend="deepep_low_latency",
            worker_cls=None,
        ),
        scheduler_config=SimpleNamespace(
            is_multimodal_model=False,
            disable_chunked_mm_input=False,
            async_scheduling=False,
        ),
        cache_config=SimpleNamespace(
            block_size=None,
            mamba_cache_mode="eager",
        ),
        compilation_config=make_compilation(),
        speculative_config=None,
        use_v2_model_runner=False,
        attention_config=None,
        kernel_config=None,
        _dflash_needs_multi_kv_group_calls=[],
        _is_dflash_candidate_draft_calls=[],
    )
    cfg._dflash_needs_multi_kv_group = lambda: (
        cfg._dflash_needs_multi_kv_group_calls.append(1) or False
    )
    cfg._is_dflash_candidate_draft = lambda: (
        cfg._is_dflash_candidate_draft_calls.append(1) or False
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _set_platform(
    monkeypatch, cls, device_type=None, vendor_name=None, dist_backend=None
):
    """Override PlatformFL class attributes for one test."""
    if device_type is not None:
        monkeypatch.setattr(cls, "device_type", device_type, raising=False)
    if vendor_name is not None:
        monkeypatch.setattr(cls, "vendor_name", vendor_name, raising=False)
    if dist_backend is not None:
        monkeypatch.setattr(cls, "dist_backend", dist_backend, raising=False)


def _messages(records, *levels):
    """Extract message strings recorded at the given levels."""
    wanted = set(levels or ("info", "debug", "warning", "info_once", "warning_once"))
    return [msg for level, msg in records if level in wanted]


def _install_npu_stubs(monkeypatch):
    """Install recording stubs for the three npu lazy imports.

    Returns a SimpleNamespace of call lists: glm_calls (list of
    model_config args), glm_result (mutable), refresh_calls,
    ascend_compilation_calls.
    """
    glm_calls = []
    state = SimpleNamespace(glm_result=False)

    def _prepare_glm_dsa_dense_fallback(model_config):
        glm_calls.append(model_config)
        return state.glm_result

    refresh_calls = []
    ascend_compilation_calls = []
    mods = {
        "vllm_fl.dispatch.backends.vendor.ascend.patches.ascend_glm_dsa": {
            "prepare_glm_dsa_dense_fallback": _prepare_glm_dsa_dense_fallback,
        },
        "vllm_fl.dispatch.backends.vendor.ascend.patch": {
            "refresh_block_size": lambda cfg: refresh_calls.append(cfg),
        },
        "vllm_fl.dispatch.backends.vendor.ascend.patches.compilation": {
            "_configure_ascend_compilation": lambda cfg: (
                ascend_compilation_calls.append(cfg)
            ),
        },
    }
    for name, attrs in mods.items():
        monkeypatch.setitem(sys.modules, name, _make_module(name, **attrs))
    return SimpleNamespace(
        glm_calls=glm_calls,
        glm_result_state=state,
        refresh_calls=refresh_calls,
        ascend_compilation_calls=ascend_compilation_calls,
    )


def _install_iluvatar_stub(monkeypatch):
    """Install a recording stub for the iluvatar compilation lazy import."""
    calls = []
    mod = _make_module(
        "vllm_fl.dispatch.backends.vendor.iluvatar.compilation",
        configure_compilation=lambda comp, model: calls.append((comp, model)),
    )
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.iluvatar.compilation", mod
    )
    return calls


def _install_mm_encoder_stub(monkeypatch):
    """Install a recording stub for vllm_fl.attention.utils.patch_..."""
    calls = []
    mod = _make_module(
        "vllm_fl.attention.utils",
        patch_mm_encoder_attention=lambda: calls.append("patched"),
    )
    monkeypatch.setitem(sys.modules, "vllm_fl.attention.utils", mod)
    return calls


# ---------------------------------------------------------------------------
# vendor gcu / enflame (patch_mm_encoder_attention, all-reduce, inductor)
# ---------------------------------------------------------------------------


class TestVendorGcuEnflame:
    """Branches keyed on vendor_name in (gcu, enflame)."""

    def test_vendor_gcu_patches_mm_encoder(self, platform_env, monkeypatch):
        """vendor gcu installs the mm-encoder attention patch."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="gcu", vendor_name="gcu")
        calls = _install_mm_encoder_stub(monkeypatch)
        cls.check_and_update_config(make_vllm_config())
        assert calls == ["patched"]

    def test_vendor_enflame_patches_mm_encoder(self, platform_env, monkeypatch):
        """vendor enflame installs the mm-encoder attention patch too."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="gcu", vendor_name="enflame")
        calls = _install_mm_encoder_stub(monkeypatch)
        cls.check_and_update_config(make_vllm_config())
        assert calls == ["patched"]

    def test_vendor_gcu_disables_custom_all_reduce(self, platform_env, monkeypatch):
        """GCU relies on ECCL: custom all-reduce is disabled."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="gcu", vendor_name="gcu")
        _install_mm_encoder_stub(monkeypatch)
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.parallel_config.disable_custom_all_reduce is True

    def test_vendor_enflame_disables_custom_all_reduce(self, platform_env, monkeypatch):
        """Enflame disables custom all-reduce the same way."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="gcu", vendor_name="enflame")
        _install_mm_encoder_stub(monkeypatch)
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.parallel_config.disable_custom_all_reduce is True

    def test_gcu_inductor_combo_kernels_disabled(self, platform_env, monkeypatch):
        """Inductor + non-NONE mode turns combo kernels off on GCU."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="gcu", vendor_name="gcu")
        _install_mm_encoder_stub(monkeypatch)
        cfg = make_vllm_config(
            compilation_config=make_compilation(
                backend="inductor", mode="VLLM_COMPILE"
            ),
        )
        cls.check_and_update_config(cfg)
        assert cfg.compilation_config.inductor_compile_config == {
            "combo_kernels": False,
            "benchmark_combo_kernel": False,
        }

    def test_gcu_inductor_mode_none_keeps_combo_kernels(
        self, platform_env, monkeypatch
    ):
        """CompilationMode.NONE leaves the inductor config untouched."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="gcu", vendor_name="gcu")
        _install_mm_encoder_stub(monkeypatch)
        cfg = make_vllm_config(
            compilation_config=make_compilation(backend="inductor", mode="NONE"),
        )
        cls.check_and_update_config(cfg)
        assert cfg.compilation_config.inductor_compile_config == {}

    def test_gcu_non_inductor_backend_keeps_combo_kernels(
        self, platform_env, monkeypatch
    ):
        """A non-inductor backend never touches the combo-kernel flags."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="gcu", vendor_name="gcu")
        _install_mm_encoder_stub(monkeypatch)
        cfg = make_vllm_config(
            compilation_config=make_compilation(backend="eager", mode="VLLM_COMPILE"),
        )
        cls.check_and_update_config(cfg)
        assert cfg.compilation_config.inductor_compile_config == {}


# ---------------------------------------------------------------------------
# musa / txda + use_v2_model_runner
# ---------------------------------------------------------------------------


class TestV2RunnerRejection:
    """musa/txda reject the upstream V2 model runner."""

    def test_musa_v2_runner_rejected(self, platform_env, monkeypatch):
        """MUSA names itself in the V2-runner rejection."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="musa")
        cfg = make_vllm_config(use_v2_model_runner=True)
        with pytest.raises(ValueError, match="MUSA requires ModelRunnerFL"):
            cls.check_and_update_config(cfg)

    def test_txda_v2_runner_rejected(self, platform_env, monkeypatch):
        """TXDA names itself in the V2-runner rejection."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="txda")
        cfg = make_vllm_config(use_v2_model_runner=True)
        with pytest.raises(ValueError, match="Tsingmicro TXDA requires ModelRunnerFL"):
            cls.check_and_update_config(cfg)


# ---------------------------------------------------------------------------
# kunlunxin speculative + hybrid/attention-free + align
# ---------------------------------------------------------------------------


class TestKunlunxinSpeculativeGate:
    """Kunlunxin rejects speculative + align cache on hybrid/attention-free."""

    def test_kunlunxin_hybrid_align_rejected(self, platform_env, monkeypatch):
        """is_hybrid + align + speculative raises."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="kunlunxin")
        cfg = make_vllm_config(
            speculative_config=SimpleNamespace(method="eagle"),
        )
        cfg.model_config.is_hybrid = True
        cfg.cache_config.mamba_cache_mode = "align"
        with pytest.raises(ValueError, match="Kunlunxin does not support"):
            cls.check_and_update_config(cfg)

    def test_kunlunxin_attention_free_align_rejected(self, platform_env, monkeypatch):
        """Attention-free models hit the same gate via is_attention_free."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="kunlunxin")
        cfg = make_vllm_config(
            speculative_config=SimpleNamespace(method="eagle"),
        )
        cfg.model_config.is_attention_free = True
        cfg.cache_config.mamba_cache_mode = "align"
        with pytest.raises(ValueError, match="Mamba cache mode .align."):
            cls.check_and_update_config(cfg)

    def test_kunlunxin_hybrid_eager_mode_ok(self, platform_env, monkeypatch):
        """Non-align mamba mode keeps the config accepted."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="kunlunxin")
        cfg = make_vllm_config(
            speculative_config=SimpleNamespace(method="eagle"),
        )
        cfg.model_config.is_hybrid = True
        cfg.cache_config.mamba_cache_mode = "eager"
        cls.check_and_update_config(cfg)
        assert cfg.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"

    def test_kunlunxin_align_without_hybrid_ok(self, platform_env, monkeypatch):
        """align mode alone (plain attention model) is not rejected."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="kunlunxin")
        cfg = make_vllm_config(
            speculative_config=SimpleNamespace(method="eagle"),
        )
        cfg.cache_config.mamba_cache_mode = "align"
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 128


# ---------------------------------------------------------------------------
# npu device block (GLM DSA fallback, block size, ascend compilation)
# ---------------------------------------------------------------------------


class TestNpuGate:
    """Ascend NPU: GLM DSA dense fallback and block-size defaults."""

    def test_npu_glm_dsa_fallback_triggered(self, platform_env, monkeypatch):
        """A True fallback result warns and revalidates parallel config."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="npu", vendor_name="ascend")
        stubs = _install_npu_stubs(monkeypatch)
        stubs.glm_result_state.glm_result = True
        verify_calls = []
        cfg = make_vllm_config()
        cfg.model_config.verify_with_parallel_config = lambda parallel: (
            verify_calls.append(parallel)
        )
        cls.check_and_update_config(cfg)
        assert stubs.glm_calls == [cfg.model_config]
        assert verify_calls == [cfg.parallel_config]
        assert any(
            level == "warning_once" and "GLM DSA" in msg
            for level, msg in platform_env.logger.records
        )

    def test_npu_glm_dsa_fallback_not_triggered(self, platform_env, monkeypatch):
        """A False fallback result leaves everything untouched."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="npu", vendor_name="ascend")
        stubs = _install_npu_stubs(monkeypatch)
        verify_calls = []
        cfg = make_vllm_config()
        cfg.model_config.verify_with_parallel_config = lambda parallel: (
            verify_calls.append(parallel)
        )
        cls.check_and_update_config(cfg)
        assert stubs.glm_calls == [cfg.model_config]
        assert verify_calls == []
        assert not any("GLM DSA" in msg for _, msg in platform_env.logger.records)

    def test_npu_model_config_none_skips_glm(self, platform_env, monkeypatch):
        """Without a model_config the GLM import never runs."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="npu", vendor_name="ascend")
        stubs = _install_npu_stubs(monkeypatch)
        cfg = make_vllm_config(model_config=None)
        cls.check_and_update_config(cfg)
        assert stubs.glm_calls == []

    def test_npu_block_size_128(self, platform_env, monkeypatch):
        """NPU forces a 128 block size and logs it."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="npu", vendor_name="ascend")
        _install_npu_stubs(monkeypatch)
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 128
        assert any(
            level == "info" and "128 for Ascend NPU" in msg
            for level, msg in platform_env.logger.records
        )

    def test_npu_refresh_block_size_called(self, platform_env, monkeypatch):
        """The ascend refresh_block_size hook receives the whole config."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="npu", vendor_name="ascend")
        stubs = _install_npu_stubs(monkeypatch)
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert stubs.refresh_calls == [cfg]

    def test_npu_configure_ascend_compilation_called(self, platform_env, monkeypatch):
        """The ascend compilation hook receives the whole config."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="npu", vendor_name="ascend")
        stubs = _install_npu_stubs(monkeypatch)
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert stubs.ascend_compilation_calls == [cfg]


# ---------------------------------------------------------------------------
# worker_cls selection
# ---------------------------------------------------------------------------


class TestWorkerClsSelection:
    """worker_cls routes to the iluvatar or default FL worker."""

    def test_worker_cls_default(self, platform_env, monkeypatch):
        """Non-iluvatar vendors use the generic WorkerFL path."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"

    def test_worker_cls_iluvatar(self, platform_env, monkeypatch):
        """vendor iluvatar routes to IluvatarWorkerFL."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="iluvatar")
        _install_iluvatar_stub(monkeypatch)
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.parallel_config.worker_cls == (
            "vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker.IluvatarWorkerFL"
        )


# ---------------------------------------------------------------------------
# mm prefix LM + multimodal scheduler
# ---------------------------------------------------------------------------


class TestMmPrefixLmScheduler:
    """chunked mm input is disabled for multimodal bidirectional models."""

    def test_mm_prefix_lm_forces_disable_chunked_mm_input(self, platform_env):
        """The warning fires and the flag is forced on."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cfg.model_config.is_mm_prefix_lm = True
        cfg.scheduler_config.is_multimodal_model = True
        cls.check_and_update_config(cfg)
        assert cfg.scheduler_config.disable_chunked_mm_input is True
        assert any(
            level == "warning_once" and "disable_chunked_mm_input" in msg
            for level, msg in platform_env.logger.records
        )

    def test_mm_prefix_lm_non_multimodal_untouched(self, platform_env):
        """A non-multimodal scheduler keeps chunked mm input enabled."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cfg.model_config.is_mm_prefix_lm = True
        cfg.scheduler_config.is_multimodal_model = False
        cls.check_and_update_config(cfg)
        assert cfg.scheduler_config.disable_chunked_mm_input is False

    def test_mm_prefix_lm_already_disabled_no_warning(self, platform_env):
        """An already-disabled flag is not re-warned."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cfg.model_config.is_mm_prefix_lm = True
        cfg.scheduler_config.is_multimodal_model = True
        cfg.scheduler_config.disable_chunked_mm_input = True
        cls.check_and_update_config(cfg)
        assert not any(
            "disable_chunked_mm_input" in msg for _, msg in platform_env.logger.records
        )

    def test_mm_prefix_model_config_none_untouched(self, platform_env):
        """A missing model_config skips the mm-prefix check entirely."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config(model_config=None)
        cfg.scheduler_config.is_multimodal_model = True
        cls.check_and_update_config(cfg)
        assert cfg.scheduler_config.disable_chunked_mm_input is False


# ---------------------------------------------------------------------------
# cache block_size defaults
# ---------------------------------------------------------------------------


class TestBlockSizeDefaults:
    """block_size None resolves per device type / vendor."""

    def test_block_size_default_16(self, platform_env):
        """The fallback (cuda/nvidia) block size is 16."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 16

    def test_block_size_kunlunxin_128(self, platform_env, monkeypatch):
        """Kunlunxin forces 128 and logs the Kunlunxin message."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="kunlunxin")
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 128
        assert any(
            level == "info" and "128 for Kunlunxin" in msg
            for level, msg in platform_env.logger.records
        )

    def test_block_size_musa_64(self, platform_env, monkeypatch):
        """MUSA forces 64 and logs the MUSA message."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="musa", vendor_name="mthreads")
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 64
        assert any(
            level == "info" and "64 for MUSA" in msg
            for level, msg in platform_env.logger.records
        )

    def test_block_size_existing_value_kept(self, platform_env):
        """A preset block_size is never overwritten."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cfg.cache_config.block_size = 32
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 32

    def test_block_size_cache_config_falsy_skipped(self, platform_env):
        """A falsy cache_config skips block-size resolution entirely."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config(cache_config=None)
        cls.check_and_update_config(cfg)
        assert cfg.cache_config is None

    def test_mla_block_size_forced_to_64(self, platform_env):
        """use_mla with a non-multiple-of-64 block forces 64."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cfg.model_config.use_mla = True
        cfg.cache_config.block_size = None  # resolved to 16, then forced to 64
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 64
        assert any(
            level == "info" and "64 for FlagOSMLA backend" in msg
            for level, msg in platform_env.logger.records
        )

    def test_mla_block_size_multiple_of_64_kept(self, platform_env):
        """use_mla with an aligned block size keeps it."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cfg.model_config.use_mla = True
        cfg.cache_config.block_size = 128
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 128

    def test_non_mla_block_size_16_kept(self, platform_env):
        """Without use_mla the resolved 16 stays 16."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 16


# ---------------------------------------------------------------------------
# compile_sizes default
# ---------------------------------------------------------------------------


class TestCompileSizesDefault:
    """compile_sizes None is normalized to an empty list."""

    def test_compile_sizes_none_becomes_empty_list(self, platform_env):
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        assert cfg.compilation_config.compile_sizes is None
        cls.check_and_update_config(cfg)
        assert cfg.compilation_config.compile_sizes == []

    def test_compile_sizes_existing_kept(self, platform_env):
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config(
            compilation_config=make_compilation(compile_sizes=[512, 1024]),
        )
        cls.check_and_update_config(cfg)
        assert cfg.compilation_config.compile_sizes == [512, 1024]


# ---------------------------------------------------------------------------
# musa-specific compilation updates
# ---------------------------------------------------------------------------


class TestMusaCompilation:
    """MUSA combo-kernel flags and FULL cudagraph downgrade."""

    def test_musa_full_cudagraph_downgraded(self, platform_env, monkeypatch):
        """FULL cudagraph mode is downgraded to PIECEWISE with a log line."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="musa", vendor_name="mthreads")
        cfg = make_vllm_config(
            compilation_config=make_compilation(cudagraph_mode="full"),
        )
        cls.check_and_update_config(cfg)
        assert cfg.compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
        assert not cfg.compilation_config.cudagraph_mode.has_full_cudagraphs()
        assert any(
            level == "info" and "Downgrading cudagraph_mode" in msg
            for level, msg in platform_env.logger.records
        )


# ---------------------------------------------------------------------------
# iluvatar compilation hook
# ---------------------------------------------------------------------------


class TestIluvatarCompilation:
    """vendor iluvatar routes compilation through its vendor module."""

    def test_iluvatar_configure_compilation_called(self, platform_env, monkeypatch):
        """configure_compilation receives (compilation_config, model_config)."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="cuda", vendor_name="iluvatar")
        calls = _install_iluvatar_stub(monkeypatch)
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert calls == [(cfg.compilation_config, cfg.model_config)]

    def test_iluvatar_model_config_none(self, platform_env, monkeypatch):
        """A missing model_config is forwarded as None."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, device_type="cuda", vendor_name="iluvatar")
        calls = _install_iluvatar_stub(monkeypatch)
        cfg = make_vllm_config(model_config=None)
        cls.check_and_update_config(cfg)
        assert calls == [(cfg.compilation_config, None)]


# ---------------------------------------------------------------------------
# WideEP: all2all deepep_high_throughput
# ---------------------------------------------------------------------------


class TestWideEpAll2All:
    """DeepEP high-throughput + dp>1 disables CUDA graphs."""

    def test_wideep_disables_cudagraph(self, platform_env, monkeypatch):
        """cudagraph_mode is forced to NONE with an explanatory log."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config(
            compilation_config=make_compilation(cudagraph_mode="piecewise"),
        )
        cfg.parallel_config.all2all_backend = "deepep_high_throughput"
        cfg.parallel_config.data_parallel_size = 2
        cls.check_and_update_config(cfg)
        assert cfg.compilation_config.cudagraph_mode.value == "none"
        assert any(
            level == "info" and "WideEP: Disabling CUDA Graphs" in msg
            for level, msg in platform_env.logger.records
        )

    def test_wideep_single_dp_keeps_cudagraph(self, platform_env):
        """dp == 1 keeps cudagraphs even on deepep_high_throughput."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config(
            compilation_config=make_compilation(cudagraph_mode="piecewise"),
        )
        cfg.parallel_config.all2all_backend = "deepep_high_throughput"
        cfg.parallel_config.data_parallel_size = 1
        cls.check_and_update_config(cfg)
        assert cfg.compilation_config.cudagraph_mode.value == "piecewise"

    def test_wideep_other_backend_keeps_cudagraph(self, platform_env):
        """A non-high-throughput all2all backend keeps cudagraphs."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config(
            compilation_config=make_compilation(cudagraph_mode="piecewise"),
        )
        cfg.parallel_config.all2all_backend = "deepep_low_latency"
        cfg.parallel_config.data_parallel_size = 2
        cls.check_and_update_config(cfg)
        assert cfg.compilation_config.cudagraph_mode.value == "piecewise"


# ---------------------------------------------------------------------------
# kunlunxin async scheduling
# ---------------------------------------------------------------------------


class TestKunlunxinAsyncScheduling:
    """kunlunxin forces async_scheduling off with a single warning."""

    def test_async_scheduling_forced_off(self, platform_env, monkeypatch):
        """A truthy async_scheduling is forced to False and warned once."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="kunlunxin")
        cfg = make_vllm_config()
        cfg.scheduler_config.async_scheduling = True
        cls.check_and_update_config(cfg)
        assert cfg.scheduler_config.async_scheduling is False
        assert any(
            level == "warning_once" and "disabling async scheduling" in msg
            for level, msg in platform_env.logger.records
        )

    def test_async_scheduling_already_false_no_warning(self, platform_env, monkeypatch):
        """An explicit False async_scheduling is untouched and unwarned."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="kunlunxin")
        cfg = make_vllm_config()
        cfg.scheduler_config.async_scheduling = False
        cls.check_and_update_config(cfg)
        assert cfg.scheduler_config.async_scheduling is False
        assert not any(
            "async scheduling" in msg for _, msg in platform_env.logger.records
        )

    def test_async_scheduling_none_coerced_off(self, platform_env, monkeypatch):
        """None (unresolved default, i.e. enabled) is coerced to False."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="kunlunxin")
        cfg = make_vllm_config()
        # Simulate the documented "not resolved yet" None default.
        cfg.scheduler_config.async_scheduling = None
        cls.check_and_update_config(cfg)
        assert cfg.scheduler_config.async_scheduling is False
        assert any(
            level == "warning_once" and "disabling async scheduling" in msg
            for level, msg in platform_env.logger.records
        )


# ---------------------------------------------------------------------------
# metax
# ---------------------------------------------------------------------------


class TestMetax:
    """metax: cascade attention off and attention flags cleared."""

    def test_metax_disables_cascade_attn(self, platform_env, monkeypatch):
        """model_config.disable_cascade_attn is forced on."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="metax")
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.model_config.disable_cascade_attn is True

    def test_metax_attention_config_flags_off(self, platform_env, monkeypatch):
        """Every metax attention_config flag is disabled."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="metax")
        cfg = make_vllm_config()
        cfg.attention_config = SimpleNamespace(
            use_cudnn_prefill=True,
            use_trtllm_ragged_deepseek_prefill=True,
            use_trtllm_attention=True,
            disable_flashinfer_prefill=False,
        )
        cls.check_and_update_config(cfg)
        att = cfg.attention_config
        assert att.use_cudnn_prefill is False
        assert att.use_trtllm_ragged_deepseek_prefill is False
        assert att.use_trtllm_attention is False
        assert att.disable_flashinfer_prefill is True

    def test_metax_model_config_none_skips_cascade(self, platform_env, monkeypatch):
        """Without a model_config only the attention_config is touched."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="metax")
        cfg = make_vllm_config(model_config=None)
        cfg.attention_config = SimpleNamespace(
            use_cudnn_prefill=True,
            use_trtllm_ragged_deepseek_prefill=True,
            use_trtllm_attention=True,
            disable_flashinfer_prefill=False,
        )
        cls.check_and_update_config(cfg)
        assert cfg.attention_config.use_cudnn_prefill is False

    def test_metax_attention_config_none_untouched(self, platform_env, monkeypatch):
        """A missing attention_config is skipped."""
        cls = platform_env.module.PlatformFL
        _set_platform(monkeypatch, cls, vendor_name="metax")
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.attention_config is None


# ---------------------------------------------------------------------------
# happy-path integration
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Default cuda/nvidia config passes through the whole method."""

    def test_default_cuda_nvidia_config(self, platform_env):
        """Block size, worker_cls and compile_sizes settle on their defaults."""
        cls = platform_env.module.PlatformFL
        cfg = make_vllm_config()
        cls.check_and_update_config(cfg)
        assert cfg.cache_config.block_size == 16
        assert cfg.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"
        assert cfg.parallel_config.disable_custom_all_reduce is False
        assert cfg.compilation_config.compile_sizes == []
        assert cfg.compilation_config.splitting_calls == []
        assert cfg.scheduler_config.disable_chunked_mm_input is False

# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl/platform.py`` (low-line methods).

``platform.py`` imports the real ``vllm`` platform graph at module level,
which is absent on CPU-only Windows hosts and heavyweight on the WSL host.
The ``platform_env`` fixture loads the module under test through
``spec_from_file_location`` against a stub vllm graph (the same pattern as
``test_nvidia_worker_platform.py``), with ``GEMS_VENDOR`` steering
``DeviceInfo``'s vendor resolution; every method under test then runs its
real code. Class attributes are monkeypatched per test to drive other vendors.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_PLATFORM_SOURCE = _ROOT / "vllm_fl" / "platform.py"
_PLATFORM_MODULE = "supp_platform_fl_under_test"

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


class _DeviceInfoStub:
    """Vendor-neutral DeviceInfo double: probes report nothing.

    ``GEMS_VENDOR`` (via ``_env_vendor_name``) then resolves the vendor, which
    exercises the ``_safe_device_type`` empty-probe path on every load.
    """

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

    class _RocmPlatform:
        @staticmethod
        def get_supported_vit_attn_backends():
            return ["rocm-sdpa", "rocm-flash"]

        @staticmethod
        def get_vit_attn_backend(head_size, dtype, backend):
            return "rocm-vit"

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
    rocm_mod = _make_module("vllm.platforms.rocm", RocmPlatform=_RocmPlatform)
    registry_mod = _make_module(
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=_AttentionBackendEnum,
    )
    config_mod = _make_module(
        "vllm.config",
        CUDAGraphMode=SimpleNamespace(NONE="none", PIECEWISE="piecewise"),
    )
    compilation_mod = _make_module(
        "vllm.config.compilation", CompilationMode=SimpleNamespace(NONE="NONE")
    )

    utils_stub = _make_module("vllm_fl.utils")
    utils_stub.DeviceInfo = _DeviceInfoStub
    utils_stub._env_vendor_name = lambda: "nvidia"
    utils_stub.get_device_control_env_var = lambda vendor: "CUDA_VISIBLE_DEVICES"
    utils_stub.get_device_type = lambda vendor: "cuda"

    stubs = {
        "vllm": vllm_pkg,
        "vllm.logger": logger_mod,
        "vllm.platforms": platforms_mod,
        "vllm.platforms.interface": interface_mod,
        "vllm.platforms.rocm": rocm_mod,
        "vllm.v1.attention.backends.registry": registry_mod,
        "vllm.config": config_mod,
        "vllm.config.compilation": compilation_mod,
        "vllm_fl.utils": utils_stub,
    }
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
# Module-level functions
# ---------------------------------------------------------------------------


class TestModuleLevelHelpers:
    """_vllm_c_available / _safe_device_type / _use_flagcx."""

    def test_vllm_c_available_bool(self, platform_env):
        """The finder result is normalized to a bool."""
        result = platform_env.module._vllm_c_available()
        assert isinstance(result, bool)

    def test_safe_device_type_prefers_probe(self, platform_env):
        """A non-empty runtime probe wins over the static map."""
        mod = platform_env.module
        info = SimpleNamespace(device_type="txda")
        assert mod._safe_device_type(info, "nvidia") == "txda"

    def test_safe_device_type_empty_vendor(self, platform_env):
        """No probe and no vendor resolves to the empty device type."""
        mod = platform_env.module
        info = SimpleNamespace(device_type="")
        assert mod._safe_device_type(info, "") == ""

    def test_safe_device_type_static_map(self, platform_env):
        """Empty probe + known vendor falls back to the static map."""
        mod = platform_env.module
        info = SimpleNamespace(device_type="")
        assert mod._safe_device_type(info, "nvidia") == "cuda"

    def test_safe_device_type_unknown_vendor(self, platform_env, monkeypatch):
        """An unknown vendor logs a warning and resolves to ''."""
        mod = platform_env.module
        info = SimpleNamespace(device_type="")

        def boom(vendor):
            raise ValueError("unknown")

        # platform.py bound get_device_type into its own namespace at import.
        monkeypatch.setattr(mod, "get_device_type", boom)
        assert mod._safe_device_type(info, "no-such-vendor") == ""

    def test_use_flagcx_env_var_wins(self, platform_env, monkeypatch):
        """FLAGCX_PATH forces flagcx regardless of vendor."""
        mod = platform_env.module
        monkeypatch.setenv("FLAGCX_PATH", "/opt/flagcx")
        assert mod._use_flagcx("nvidia") is True

    def test_use_flagcx_non_kunlunxin(self, platform_env, monkeypatch):
        """Without FLAGCX_PATH only kunlunxin can pick flagcx."""
        mod = platform_env.module
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
        assert mod._use_flagcx("nvidia") is False

    def test_use_flagcx_kunlunxin_import_failure(self, platform_env, monkeypatch):
        """kunlunxin without an importable flagcx.api stays on nccl."""
        mod = platform_env.module
        monkeypatch.delenv("FLAGCX_PATH", raising=False)

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def blocked(name, *args, **kwargs):
            if name == "flagcx.api":
                raise ImportError("no flagcx")
            return real_import(name, *args, **kwargs)

        import builtins

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert mod._use_flagcx("kunlunxin") is False


# ---------------------------------------------------------------------------
# Instance predicates
# ---------------------------------------------------------------------------


class TestPredicates:
    """Vendor/device predicates on the loaded class."""

    def test_is_cuda_nvidia(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.is_cuda(cls) is True

    def test_is_cuda_alike_nvidia(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.is_cuda_alike(cls) is True

    def test_is_cuda_alike_musa(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "musa", raising=False)
        assert cls.is_cuda_alike(cls) is True

    def test_is_cuda_alike_iluvatar_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "iluvatar", raising=False)
        assert cls.is_cuda_alike(cls) is False

    def test_is_cuda_alike_hygon_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "hygon", raising=False)
        assert cls.is_cuda_alike(cls) is False

    def test_is_cuda_alike_gcu_true(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "gcu", raising=False)
        assert cls.is_cuda_alike(cls) is True

    def test_is_cuda_alike_non_cuda_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "npu", raising=False)
        assert cls.is_cuda_alike(cls) is False

    def test_is_rocm(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        assert cls.is_rocm(cls) is False
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        assert cls.is_rocm(cls) is True

    def test_is_sleep_mode_available_nvidia_only(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        assert cls.is_sleep_mode_available(cls) is True
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        assert cls.is_sleep_mode_available(cls) is False

    def test_is_musa_gcu_absent_attrs(self, platform_env, monkeypatch):
        """is_musa/is_gcu answer False when torch lacks the submodule."""
        cls = platform_env.module.PlatformFL
        monkeypatch.delattr(torch, "musa", raising=False)
        monkeypatch.delattr(torch, "gcu", raising=False)
        assert cls.is_musa(cls) is False
        assert cls.is_gcu(cls) is False

    def test_supported_dtypes(self, platform_env):
        cls = platform_env.module.PlatformFL
        instance = cls.__new__(cls)
        assert instance.supported_dtypes == [
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ]

    def test_is_pin_memory_available(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        assert cls.is_pin_memory_available() is True  # cuda
        monkeypatch.setattr(cls, "device_type", "npu", raising=False)
        assert cls.is_pin_memory_available() is True
        monkeypatch.setattr(cls, "device_type", "ptpu", raising=False)
        assert cls.is_pin_memory_available() is False


# ---------------------------------------------------------------------------
# dtype support / memory / device helpers
# ---------------------------------------------------------------------------


class TestDeviceHelpers:
    """check_if_supports_dtype, memory and device plumbing."""

    def test_bf16_rejected_below_sm80(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "has_device_capability", lambda major: False)
        monkeypatch.setattr(cls, "get_device_capability", lambda device_id=0: None)
        monkeypatch.setattr(cls, "get_device_name", lambda device_id=0: "T4")
        with pytest.raises(ValueError, match="compute capability"):
            cls.check_if_supports_dtype(torch.bfloat16)

    def test_bf16_accepted_at_sm80(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "has_device_capability", lambda major: True)
        cls.check_if_supports_dtype(torch.bfloat16)

    def test_bf16_non_nvidia_skips_check(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        monkeypatch.setattr(cls, "has_device_capability", lambda major: False)
        cls.check_if_supports_dtype(torch.bfloat16)

    def test_get_current_memory_usage(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        calls = []
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(
                empty_cache=lambda: calls.append("empty_cache"),
                reset_peak_memory_stats=lambda device: calls.append(("peak", device)),
                max_memory_allocated=lambda device: 42,
            ),
        )
        assert cls.get_current_memory_usage() == 42
        assert "empty_cache" in calls

    def test_set_device_nvidia_forces_context(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        seen = []
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(set_device=lambda d: seen.append(d)),
        )
        dev = torch.device("cpu")
        cls.set_device(dev)
        assert seen == [dev]

    def test_empty_cache_delegates(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        calls = []
        monkeypatch.setattr(
            cls, "torch_device_fn", SimpleNamespace(empty_cache=lambda: calls.append(1))
        )
        cls.empty_cache()
        assert calls == [1]

    def test_import_ir_kernels(self, platform_env, monkeypatch):
        """import_ir_kernels imports vllm.kernels and stays quiet."""
        cls = platform_env.module.PlatformFL
        kernels = _make_module("vllm.kernels")
        monkeypatch.setitem(sys.modules, "vllm.kernels", kernels)
        # No exception: the import of the stubbed module resolves cleanly.
        cls.import_ir_kernels()
        # The method imported the module it names, not something else.
        assert sys.modules["vllm.kernels"] is kernels
        # The kernel import is not a warning/error condition.
        assert not [
            (level, msg)
            for level, msg in platform_env.logger.records
            if level in ("warning", "warning_once", "error", "debug")
        ]

    def test_import_kernels_super_for_plain_nvidia(self, platform_env, monkeypatch):
        """A non-metax vendor delegates kernel import to the base class."""
        cls = platform_env.module.PlatformFL
        base = platform_env.base
        base.super_calls.clear()
        cls.import_kernels()
        assert "import_kernels" in base.super_calls


# ---------------------------------------------------------------------------
# apply_config_platform_defaults
# ---------------------------------------------------------------------------


class TestApplyConfigPlatformDefaults:
    """Vendor-default installation before compilation resolution."""

    def test_kunlunxin_forces_eager(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)
        monkeypatch.setattr(cls, "vendor_name", "kunlunxin", raising=False)
        # Patch the CompilationMode enum on the stub module the fixture
        # installed instead of swapping the sys.modules entry: a
        # monkeypatch.setitem here would record the fixture's stub as the old
        # value and re-install it after the fixture's own teardown popped it,
        # leaving an orphaned stub behind for every later real import.
        stub_compilation = sys.modules["vllm.config.compilation"]
        modes = SimpleNamespace(NONE="NONE")
        monkeypatch.setattr(stub_compilation, "CompilationMode", modes)
        cfg = SimpleNamespace(compilation_config=SimpleNamespace(mode=None))
        cls.apply_config_platform_defaults(cfg)
        assert cfg.compilation_config.mode == "NONE"

    def test_non_metax_kunlunxin_untouched(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "nvidia", raising=False)
        cfg = SimpleNamespace(compilation_config=SimpleNamespace(mode=None))
        cls.apply_config_platform_defaults(cfg)
        assert cfg.compilation_config.mode is None


# ---------------------------------------------------------------------------
# Backend class selectors
# ---------------------------------------------------------------------------


class TestBackendSelectors:
    """get_punica_wrapper / communicator / graph wrapper / compile backend."""

    def test_get_punica_wrapper(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.get_punica_wrapper() == (
            "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
        )

    def test_communicator_flagcx_generic(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "dist_backend", "flagcx", raising=False)
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)
        monkeypatch.setattr(cls, "vendor_name", "nvidia", raising=False)
        assert cls.get_device_communicator_cls().endswith("CommunicatorFL")

    def test_communicator_flagcx_iluvatar(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "dist_backend", "flagcx", raising=False)
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)
        monkeypatch.setattr(cls, "vendor_name", "iluvatar", raising=False)
        assert "IluvatarCommunicatorFL" in cls.get_device_communicator_cls()

    def test_communicator_flagcx_amd(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "dist_backend", "flagcx", raising=False)
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        assert "CudaFlagcxCommunicatorFL" in cls.get_device_communicator_cls()

    def test_communicator_nccl_parent(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "dist_backend", "nccl", raising=False)
        assert cls.get_device_communicator_cls().startswith("vllm.distributed")

    def test_static_graph_wrapper_musa(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "musa", raising=False)
        assert "MUSAGraphWrapper" in cls.get_static_graph_wrapper_cls()

    def test_static_graph_wrapper_default(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.get_static_graph_wrapper_cls().endswith("graph.GraphWrapper")

    def test_compile_backend_npu_eager(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "npu", raising=False)
        assert cls.get_compile_backend() == "eager"

    def test_compile_backend_musa_eager(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "musa", raising=False)
        assert cls.get_compile_backend() == "eager"

    def test_compile_backend_cuda_super(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.get_compile_backend() == "super-backend"

    def test_use_custom_allreduce_default(self, platform_env):
        cls = platform_env.module.PlatformFL
        monkeypatch = pytest.MonkeyPatch()
        try:
            assert cls.use_custom_allreduce() is True
        finally:
            monkeypatch.undo()

    def test_use_custom_allreduce_flagcx_off(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "dist_backend", "flagcx", raising=False)
        assert cls.use_custom_allreduce() is False

    def test_use_custom_allreduce_amd_off(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        assert cls.use_custom_allreduce() is False

    def test_support_hybrid_kv_cache_and_opaque(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.support_hybrid_kv_cache() is True
        assert cls.opaque_attention_op() is True


# ---------------------------------------------------------------------------
# Block copy helpers (pure tensor semantics)
# ---------------------------------------------------------------------------


class TestBlockCopies:
    """insert_blocks_to_device / swap_out_blocks_to_host tensor contracts."""

    def _tensors(self):
        src = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
        dst = torch.zeros(2, 3, 4)
        return src, dst

    def test_insert_blocks_to_device(self, platform_env):
        cls = platform_env.module.PlatformFL
        src, dst = self._tensors()
        src_idx = torch.tensor([0])
        dst_idx = torch.tensor([2])
        cls.insert_blocks_to_device(src, dst, src_idx, dst_idx)
        assert torch.equal(dst[:, 2], src[:, 0])

    def test_swap_out_blocks_to_host(self, platform_env):
        cls = platform_env.module.PlatformFL
        src, dst = self._tensors()
        src_idx = torch.tensor([1])
        dst_idx = torch.tensor([0])
        cls.swap_out_blocks_to_host(src, dst, src_idx, dst_idx)
        assert torch.equal(dst[:, 0], src[:, 1])


# ---------------------------------------------------------------------------
# ViT attention backends
# ---------------------------------------------------------------------------


class TestVitAttentionBackends:
    """get_supported_vit_attn_backends / get_vit_attn_backend."""

    def test_supported_default_pair(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.get_supported_vit_attn_backends() == ["torch_sdpa", "flash_attn"]

    def test_supported_amd_defers_to_rocm(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        assert cls.get_supported_vit_attn_backends() == ["rocm-sdpa", "rocm-flash"]

    def test_vit_explicit_backend_returned(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "nvidia", raising=False)
        # patch_mm_encoder_attention is a vllm_fl-internal boundary imported
        # lazily inside the method; stub the whole subpackage chain.
        attention_pkg = _make_module("vllm_fl.attention")
        utils_mod = _make_module("vllm_fl.attention.utils")
        utils_mod.patch_mm_encoder_attention = lambda: None
        attention_pkg.utils = utils_mod
        monkeypatch.setitem(sys.modules, "vllm_fl.attention", attention_pkg)
        monkeypatch.setitem(sys.modules, "vllm_fl.attention.utils", utils_mod)
        assert (
            cls.get_vit_attn_backend(64, torch.bfloat16, "flash_attn") == "flash_attn"
        )

    def test_vit_amd_unsupported_explicit(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        attention_pkg = _make_module("vllm_fl.attention")
        utils_mod = _make_module("vllm_fl.attention.utils")
        utils_mod.patch_mm_encoder_attention = lambda: None
        attention_pkg.utils = utils_mod
        monkeypatch.setitem(sys.modules, "vllm_fl.attention", attention_pkg)
        monkeypatch.setitem(sys.modules, "vllm_fl.attention.utils", utils_mod)
        monkeypatch.setattr(
            cls,
            "get_supported_vit_attn_backends",
            classmethod(lambda c: ["torch_sdpa"]),
        )
        with pytest.raises(AssertionError, match="not supported"):
            cls.get_vit_attn_backend(64, torch.bfloat16, "flash_attn")

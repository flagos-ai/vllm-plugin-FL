# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl/platform.py`` (device introspection).

Companion to ``test_platform_fl_methods.py``: the same ``platform_env``
pattern (stub vllm graph + ``spec_from_file_location`` load) drives the
high-line methods — capability probing, UUID resolution, IR-op priority
defaults and the ROCm/env toggles.
"""

import dataclasses
import importlib.util
import importlib.util as iu
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_PLATFORM_SOURCE = _ROOT / "vllm_fl" / "platform.py"
_PLATFORM_MODULE = "supp_platform_fl_introspect_under_test"

_STUB_ENTRIES = [
    "vllm",
    "vllm.logger",
    "vllm.platforms",
    "vllm.platforms.interface",
    "vllm.platforms.rocm",
    "vllm.v1.attention.backends.registry",
    "vllm.config",
    "vllm.config.compilation",
    "vllm.config.kernel",
    "vllm.ir",
    "vllm.ir.op",
    "vllm.ir.ops",
    "vllm_fl.utils",
]

_ROCM_ENV = "VLLM_ROCM_USE_SKINNY_GEMM"


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

    def exception(self, msg, *args):
        self._log("exception", msg, *args)

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
    """Vendor-neutral DeviceInfo double: probes report nothing."""

    vendor_name = ""
    device_type = ""
    device = SimpleNamespace(__class__=SimpleNamespace(__name__="NoDevice"))
    dispatch_key = "CUDA"
    torch_device_fn = torch.cuda


@dataclasses.dataclass
class _IrOpPriorityConfig:
    """Dataclass stub: ``fields()`` must enumerate the IR-op field names."""

    rms_norm: Any = None
    fused_add_rms_norm: Any = None

    @classmethod
    def with_default(cls, default, **kwargs):
        instance = cls()
        instance.default = list(default)
        for key, value in kwargs.items():
            setattr(instance, key, list(value))
        return instance


class _IrOp:
    def __init__(self, impls):
        self.impls = list(impls)


def _detached_environ(value=None):
    """Plain-dict stand-in for os.environ with the ROCm var set to ``value``.

    ``_disable_rocm_native_kernels`` writes ``os.environ`` directly, so the
    tests swap in a detached copy: the write lands in the copy and the real
    process environment is never touched.
    """
    env = {k: v for k, v in os.environ.items() if k != _ROCM_ENV}
    if value is not None:
        env[_ROCM_ENV] = value
    return env


@pytest.fixture
def platform_env(monkeypatch):
    """Load platform.py against a stub vllm graph; yield (module, logger)."""
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.delenv(_ROCM_ENV, raising=False)
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

        @classmethod
        def visible_device_id_to_physical_device_id(cls, device_id):
            return device_id

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
            return []

        @staticmethod
        def get_vit_attn_backend(head_size, dtype, backend):
            return None

    class _AttentionBackendEnum:
        TORCH_SDPA = "torch_sdpa"
        FLASH_ATTN = "flash_attn"

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
    kernel_mod = _make_module(
        "vllm.config.kernel", IrOpPriorityConfig=_IrOpPriorityConfig
    )
    ir_mod = _make_module("vllm.ir")
    ir_op_mod = _make_module("vllm.ir.op", IrOp=_IrOp)
    ir_ops_mod = _make_module(
        "vllm.ir.ops",
        rms_norm=_IrOp(["native", "vllm_c"]),
        fused_add_rms_norm=_IrOp(["native", "vllm_c"]),
    )
    ir_mod.op = ir_op_mod
    ir_mod.ops = ir_ops_mod

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
        "vllm.config.kernel": kernel_mod,
        "vllm.ir": ir_mod,
        "vllm.ir.op": ir_op_mod,
        "vllm.ir.ops": ir_ops_mod,
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
    yield SimpleNamespace(module=module, logger=logger, stubs=stubs)
    sys.modules.pop(_PLATFORM_MODULE, None)
    for name in _STUB_ENTRIES:
        if saved[name] is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved[name]


class TestSupportsFp8:
    """supports_fp8's nvidia capability gate."""

    def test_nvidia_below_sm89_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "has_device_capability", lambda major: False)
        assert cls.supports_fp8() is False

    def test_nvidia_at_sm89_true(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "has_device_capability", lambda major: major <= 89)
        assert cls.supports_fp8() is True

    def test_non_nvidia_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        assert cls.supports_fp8() is False


class TestGetDeviceUuid:
    """UUID resolution per device family."""

    def test_iluvatar_synthetic(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)
        monkeypatch.setattr(cls, "vendor_name", "iluvatar", raising=False)
        assert cls.get_device_uuid(3) == "ILUVATAR-3"

    def test_npu_visible_devices(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "npu", raising=False)
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "5,6,7")
        assert cls.get_device_uuid(1) == "NPU-6"

    def test_npu_default_index(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "npu", raising=False)
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
        assert cls.get_device_uuid(2) == "NPU-2"

    def test_other_device_type(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "txda", raising=False)
        assert cls.get_device_uuid(0) == "txda-0"

    def test_cuda_pynvml_path(self, platform_env, monkeypatch):
        """The CUDA path resolves through pynvml and the ordinal mapping."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)
        monkeypatch.setattr(cls, "vendor_name", "nvidia", raising=False)
        monkeypatch.setattr(
            cls, "visible_device_id_to_physical_device_id", lambda i: i + 10
        )
        pynvml = _make_module("pynvml")
        events = []
        pynvml.nvmlInit = lambda: events.append("init")
        pynvml.nvmlShutdown = lambda: events.append("shutdown")
        pynvml.nvmlDeviceGetHandleByIndex = lambda idx: ("handle", idx)
        pynvml.nvmlDeviceGetUUID = lambda handle: f"GPU-{handle[1]}"
        monkeypatch.setitem(sys.modules, "pynvml", pynvml)
        assert cls.get_device_uuid(4) == "GPU-14"
        assert events == ["init", "shutdown"]


class TestDeviceProperties:
    """get_device_total_memory / num_compute_units via torch_device_fn."""

    def test_total_memory(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(
                get_device_properties=lambda i: SimpleNamespace(total_memory=1234)
            ),
        )
        assert cls.get_device_total_memory(1) == 1234

    def test_num_compute_units(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(
                get_device_properties=lambda i: SimpleNamespace(
                    multi_processor_count=108
                )
            ),
        )
        assert cls.num_compute_units(0) == 108


class TestUseCustomOpCollectives:
    """use_custom_op_collectives' vendor/backend matrix."""

    def test_txda_true(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "txda", raising=False)
        assert cls.use_custom_op_collectives() is True

    def test_nvidia_true(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.use_custom_op_collectives() is True

    def test_iluvatar_true(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "iluvatar", raising=False)
        assert cls.use_custom_op_collectives() is True

    def test_musa_mccl_true(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "mthreads", raising=False)
        monkeypatch.setattr(cls, "device_type", "musa", raising=False)
        monkeypatch.setattr(cls, "dist_backend", "mccl", raising=False)
        assert cls.use_custom_op_collectives() is True

    def test_musa_nccl_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "mthreads", raising=False)
        monkeypatch.setattr(cls, "device_type", "musa", raising=False)
        monkeypatch.setattr(cls, "dist_backend", "nccl", raising=False)
        assert cls.use_custom_op_collectives() is False

    def test_amd_flagcx_true(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        monkeypatch.setattr(cls, "dist_backend", "flagcx", raising=False)
        assert cls.use_custom_op_collectives() is True

    def test_amd_nccl_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        monkeypatch.setattr(cls, "dist_backend", "nccl", raising=False)
        assert cls.use_custom_op_collectives() is False


class TestGetDeviceCapability:
    """Capability probe per device family."""

    def test_npu_returns_none(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "npu", raising=False)
        assert cls.get_device_capability() is None

    def test_ptpu_returns_none(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "ptpu", raising=False)
        assert cls.get_device_capability() is None

    def test_txda_returns_none(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "txda", raising=False)
        assert cls.get_device_capability() is None

    def test_gcu_without_module(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "gcu", raising=False)
        monkeypatch.delattr(torch, "gcu", raising=False)
        assert cls.get_device_capability() is None

    def test_gcu_with_module(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "gcu", raising=False)
        fake_gcu = SimpleNamespace(get_device_capability=lambda i: (2, 3))
        monkeypatch.setattr(torch, "gcu", fake_gcu, raising=False)
        cap = cls.get_device_capability(0)
        assert (cap.major, cap.minor) == (2, 3)

    def test_cuda_path(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(get_device_capability=lambda i: (9, 0)),
        )
        cap = cls.get_device_capability(0)
        assert (cap.major, cap.minor) == (9, 0)

    def test_musa_path(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "musa", raising=False)
        fake_musa = SimpleNamespace(get_device_capability=lambda i: (2, 1))
        monkeypatch.setattr(torch, "musa", fake_musa, raising=False)
        cap = cls.get_device_capability(0)
        assert (cap.major, cap.minor) == (2, 1)


class TestSupportDeepGemm:
    """Hopper/Blackwell-only DeepGEMM gate."""

    def test_nvidia_sm90(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "is_device_capability", lambda major: major == 90)
        monkeypatch.setattr(cls, "is_device_capability_family", lambda f: False)
        assert cls.support_deep_gemm() is True

    def test_nvidia_family_100(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "is_device_capability", lambda major: False)
        monkeypatch.setattr(cls, "is_device_capability_family", lambda f: f == 100)
        assert cls.support_deep_gemm() is True

    def test_nvidia_older_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "is_device_capability", lambda major: False)
        monkeypatch.setattr(cls, "is_device_capability_family", lambda f: False)
        assert cls.support_deep_gemm() is False

    def test_non_nvidia_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        assert cls.support_deep_gemm() is False


class TestGetDefaultIrOpPriority:
    """IR-op priority defaults per vendor and compile mode."""

    def _cfg(self, backend="inductor", mode="NONE"):
        return SimpleNamespace(
            compilation_config=SimpleNamespace(backend=backend, mode=mode)
        )

    def test_kunlunxin_native_only(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "kunlunxin", raising=False)
        result = cls.get_default_ir_op_priority(self._cfg())
        assert result.default == ["native"]

    def test_npu_native_only(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "npu", raising=False)
        result = cls.get_default_ir_op_priority(self._cfg(mode="NONE"))
        assert result.default == ["native"]

    def test_inductor_active_native_only(self, platform_env):
        cls = platform_env.module.PlatformFL
        result = cls.get_default_ir_op_priority(self._cfg(mode="PIECEWISE"))
        assert result.default == ["native"]

    def test_nvidia_without_vllm_c_native_only(self, platform_env):
        """The stub vllm ships no vllm._C, so nvidia keeps native only."""
        cls = platform_env.module.PlatformFL
        result = cls.get_default_ir_op_priority(self._cfg(mode="NONE"))
        assert result.default == ["native"]

    def test_amd_queries_registered_providers(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        result = cls.get_default_ir_op_priority(self._cfg(mode="NONE"))
        # The stub ir ops all carry a vllm_c impl, so the priority widens.
        assert result.default == ["vllm_c", "native"]

    def test_registered_provider_missing_impl(self, platform_env, monkeypatch):
        """An op without a vllm_c impl collapses the answer to False."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        ir_ops = platform_env.stubs["vllm.ir.ops"]
        saved = ir_ops.fused_add_rms_norm
        ir_ops.fused_add_rms_norm = _IrOp(["native"])
        try:
            assert cls._vllm_c_is_registered_provider() is False
        finally:
            ir_ops.fused_add_rms_norm = saved

    def test_registered_provider_import_failure(self, platform_env):
        """A missing vllm.ir import answers False without raising."""
        cls = platform_env.module.PlatformFL
        # A None sys.modules entry makes ``from vllm.ir import ...`` raise
        # ImportError. Restored manually before the fixture's own teardown
        # (a monkeypatch.setitem here would re-install the fixture's stub
        # after that teardown and leak it into every later real import).
        saved = sys.modules["vllm.ir"]
        sys.modules["vllm.ir"] = None
        try:
            assert cls._vllm_c_is_registered_provider() is False
        finally:
            sys.modules["vllm.ir"] = saved


class TestArchPdl:
    """is_arch_support_pdl's nvidia + major>=9 gate."""

    def test_non_nvidia_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        assert cls.is_arch_support_pdl() is False

    def test_probe_failure_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL

        def boom():
            raise RuntimeError("no device")

        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(current_device=boom),
        )
        assert cls.is_arch_support_pdl() is False

    def test_sm90_true(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(
                current_device=lambda: 0,
                get_device_capability=lambda device: (9, 0),
            ),
        )
        assert cls.is_arch_support_pdl() is True

    def test_sm80_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(
                current_device=lambda: 0,
                get_device_capability=lambda device: (8, 0),
            ),
        )
        assert cls.is_arch_support_pdl() is False


class TestIsFullyConnected:
    """NVLink full-connectivity probe's guarded failure paths."""

    def _pynvml_stub(self, p2p_status):
        pynvml = _make_module("pynvml")
        pynvml.nvmlInit = lambda: None
        pynvml.nvmlDeviceGetHandleByIndex = lambda i: SimpleNamespace(index=i)
        pynvml.NVML_P2P_STATUS_OK = 0
        pynvml.NVML_P2P_CAPS_INDEX_NVLINK = 5
        pynvml.NVMLError = type("NVMLError", (Exception,), {})
        pynvml.nvmlDeviceGetP2PStatus = p2p_status
        return pynvml

    def test_pynvml_missing_false(self, platform_env, monkeypatch):
        """Without pynvml the probe answers False (bare except)."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setitem(sys.modules, "pynvml", None)  # forces ImportError
        assert cls.is_fully_connected([0, 1]) is False

    def test_all_pairs_connected_true(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        pynvml = self._pynvml_stub(lambda a, b, caps: 0)
        monkeypatch.setitem(sys.modules, "pynvml", pynvml)
        assert cls.is_fully_connected([0, 1]) is True

    def test_pair_disconnected_false(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        pynvml = self._pynvml_stub(lambda a, b, caps: -1)
        monkeypatch.setitem(sys.modules, "pynvml", pynvml)
        assert cls.is_fully_connected([0, 1]) is False

    def test_nvlink_error_logged_false(self, platform_env, monkeypatch):
        """An NVMLError during the pair query logs and answers False."""
        cls = platform_env.module.PlatformFL

        def raise_error(a, b, caps):
            raise pynvml.NVMLError("no nvlink")

        pynvml = self._pynvml_stub(raise_error)
        monkeypatch.setitem(sys.modules, "pynvml", pynvml)
        assert cls.is_fully_connected([0, 1]) is False
        assert any("NVLink" in msg for _, msg in platform_env.logger.records)


class TestManualSeedAll:
    """RNG seeding across device families."""

    def test_manual_seed_all_delegates(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        seen = []
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(manual_seed_all=lambda seed: seen.append(seed)),
        )
        cls.manual_seed_all(7)
        assert seen == [7]

    def test_manual_seed_all_fallback(self, platform_env, monkeypatch):
        """Without manual_seed_all, torch.manual_seed is used."""
        cls = platform_env.module.PlatformFL
        seen = []
        monkeypatch.setattr(torch, "manual_seed", lambda seed: seen.append(seed))
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(),  # no manual_seed_all, no device_count
        )
        cls.manual_seed_all(9)
        assert seen == [9]

    def test_manual_seed_all_per_device(self, platform_env, monkeypatch):
        """With device_count + generators each device is seeded."""
        cls = platform_env.module.PlatformFL
        seeds = []

        class _Gen:
            def __init__(self, device_id):
                self.device_id = device_id

            def manual_seed(self, seed):
                seeds.append((self.device_id, seed))

        monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
        monkeypatch.setattr(
            cls,
            "torch_device_fn",
            SimpleNamespace(
                device_count=lambda: 2,
                _get_or_create_default_generator=lambda i: _Gen(i),
            ),
        )
        cls.manual_seed_all(3)
        assert seeds == [(0, 3), (1, 3)]


class TestMiscIntrospection:
    """Remaining small predicates and the ROCm skinny-gemm toggle."""

    def test_check_runner_kv_caches_multi_layer_none(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.check_runner_kv_caches_multi_layer() is None

    def test_is_integrated_gpu_false(self, platform_env):
        cls = platform_env.module.PlatformFL
        assert cls.is_integrated_gpu() is False

    def test_disable_rocm_no_extension_forces_off(self, platform_env, monkeypatch):
        """Without vllm._rocm_C the skinny-gemm default is forced off."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        monkeypatch.setattr(os, "environ", _detached_environ())
        monkeypatch.setattr(iu, "find_spec", lambda name: None)
        cls._disable_rocm_native_kernels()
        assert os.environ[_ROCM_ENV] == "0"
        assert any("SKINNY_GEMM" in msg for _, msg in platform_env.logger.records)

    def test_disable_rocm_extension_present_untouched(self, platform_env, monkeypatch):
        """A present vllm._rocm_C leaves the env var untouched."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        monkeypatch.setattr(os, "environ", _detached_environ())
        monkeypatch.setattr(iu, "find_spec", lambda name: object())
        cls._disable_rocm_native_kernels()
        assert _ROCM_ENV not in os.environ

    def test_disable_rocm_explicit_override(self, platform_env, monkeypatch):
        """An explicit env value wins over the default-off."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        monkeypatch.setattr(os, "environ", _detached_environ("1"))
        monkeypatch.setattr(iu, "find_spec", lambda name: None)
        cls._disable_rocm_native_kernels()
        assert os.environ[_ROCM_ENV] == "1"

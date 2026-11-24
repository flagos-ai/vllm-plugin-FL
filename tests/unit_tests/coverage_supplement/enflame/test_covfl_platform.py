"""Coverage tests for vllm_fl.platform and its small sibling modules.

Real torch is used throughout: sys.modules["torch"] is never replaced and no
fake torch module is built. Vendor/device behaviour is simulated by patching
individual attributes only (torch.musa, torch.gcu, torch.manual_seed,
PlatformFL.torch_device_fn ...). Heavy vLLM modules are imported for real;
only the vllm_fl vendor helper modules touched by a branch are swapped for
in-memory modules for the duration of one test.
"""

from __future__ import annotations

import builtins
import dataclasses
import enum
import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

_SRC = (
    os.environ.get("VLLM_FL_SRC")
    or os.environ.get("ENFLAME_SRC")
    or r"C:\Users\21649\AppData\Local\Temp\opencode\enflame-src-030b"
)
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import vllm_fl
    import vllm_fl.platform as pf
    import vllm_fl.utils as u
except Exception as exc:  # pragma: no cover - unusable environment
    pytest.skip(f"vllm_fl is not importable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _install_mod(monkeypatch, name, **attrs):
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child, mod, raising=False)
    return mod


@contextmanager
def _block_import(name, exc):
    real = builtins.__import__

    def fake(module_name, *args, **kwargs):
        if module_name == name:
            raise exc
        return real(module_name, *args, **kwargs)

    builtins.__import__ = fake
    try:
        yield
    finally:
        builtins.__import__ = real


class FakeCompilationMode(enum.Enum):
    NONE = "none"
    VLLM_COMPILE = "vllm_compile"


class FakeCUDAGraphMode(enum.Enum):
    NONE = "none"
    PIECEWISE = "piecewise"
    FULL = "full"
    FULL_DECODE_ONLY = "full_decode_only"

    def has_full_cudagraphs(self):
        return self in (FakeCUDAGraphMode.FULL, FakeCUDAGraphMode.FULL_DECODE_ONLY)

    def has_piecewise_cudagraphs(self):
        return self in (
            FakeCUDAGraphMode.PIECEWISE,
            FakeCUDAGraphMode.FULL,
            FakeCUDAGraphMode.FULL_DECODE_ONLY,
        )


@pytest.fixture
def fake_enums(monkeypatch):
    vc = pytest.importorskip("vllm.config")
    vcc = pytest.importorskip("vllm.config.compilation")
    monkeypatch.setattr(vc, "CUDAGraphMode", FakeCUDAGraphMode, raising=False)
    monkeypatch.setattr(vcc, "CompilationMode", FakeCompilationMode, raising=False)
    return FakeCompilationMode, FakeCUDAGraphMode


def _make_vllm_config():
    model_config = SimpleNamespace(
        is_hybrid=False,
        is_attention_free=False,
        is_mm_prefix_lm=False,
        use_mla=False,
        is_moe=False,
        disable_cascade_attn=False,
        verify_with_parallel_config=MagicMock(),
    )
    cache_config = SimpleNamespace(block_size=16, mamba_cache_mode=None)
    parallel_config = SimpleNamespace(
        worker_cls=None,
        disable_custom_all_reduce=False,
        all2all_backend="allgather",
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        world_size_across_dp=None,
    )
    scheduler_config = SimpleNamespace(
        is_multimodal_model=False,
        disable_chunked_mm_input=False,
        async_scheduling=False,
    )
    compilation_config = SimpleNamespace(
        backend="inductor",
        mode=FakeCompilationMode.VLLM_COMPILE,
        inductor_compile_config={},
        compile_sizes=None,
        cudagraph_mode=FakeCUDAGraphMode.NONE,
        cudagraph_capture_sizes=[],
        max_cudagraph_capture_size=0,
        use_inductor_graph_partition=False,
        splitting_ops=[],
        cudagraph_mm_encoder=True,
        pass_config=SimpleNamespace(fuse_rope_kvcache_cat_mla=True),
        set_splitting_ops_for_v1=MagicMock(),
    )
    return SimpleNamespace(
        model_config=model_config,
        speculative_config=None,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        compilation_config=compilation_config,
        attention_config=None,
        kernel_config=None,
        use_v2_model_runner=False,
    )


def _platform(monkeypatch, vendor="gcu", device_type="gcu", **extra):
    monkeypatch.setattr(pf.PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(pf.PlatformFL, "device_type", device_type)
    monkeypatch.setattr(
        pf.PlatformFL, "device_name", extra.pop("device_name", device_type)
    )
    for name, value in extra.items():
        monkeypatch.setattr(pf.PlatformFL, name, value)


@pytest.fixture
def gcu_deps(monkeypatch):
    calls = {}

    def _configure(config):
        calls["configure_codegen_cache"] = calls.get("configure_codegen_cache", 0) + 1
        calls["codegen_config"] = config

    def _patch_mm():
        calls["patch_mm_encoder_attention"] = (
            calls.get("patch_mm_encoder_attention", 0) + 1
        )

    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.gcu.compilation_config",
        configure_codegen_cache=_configure,
    )
    _install_mod(
        monkeypatch,
        "vllm_fl.attention.utils",
        patch_mm_encoder_attention=_patch_mm,
    )
    return calls


@pytest.fixture
def npu_deps(monkeypatch):
    calls = {}

    def _prepare(model_config):
        calls["prepared_model_config"] = model_config
        return calls.get("prepare_result", True)

    def _refresh(config):
        calls["refresh_block_size"] = config

    def _configure(config):
        calls["configure_ascend_compilation"] = config

    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.ascend_glm_dsa",
        prepare_glm_dsa_dense_fallback=_prepare,
    )
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patch",
        refresh_block_size=_refresh,
    )
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.compilation",
        _configure_ascend_compilation=_configure,
    )
    return calls


@pytest.fixture
def iluvatar_deps(monkeypatch):
    calls = {}

    def _configure(compilation, model):
        calls["configure_compilation"] = (compilation, model)

    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.iluvatar.compilation",
        configure_compilation=_configure,
    )
    return calls


# ---------------------------------------------------------------------------
# module level helpers
# ---------------------------------------------------------------------------


def test_vllm_c_available(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: sentinel)
    assert pf._vllm_c_available() is True

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert pf._vllm_c_available() is False

    def _missing(name):
        raise ImportError("missing")

    monkeypatch.setattr(importlib.util, "find_spec", _missing)
    assert pf._vllm_c_available() is False

    def _bad(name):
        raise ValueError("bad")

    monkeypatch.setattr(importlib.util, "find_spec", _bad)
    assert pf._vllm_c_available() is False


def test_safe_device_type(monkeypatch):
    assert pf._safe_device_type(SimpleNamespace(device_type="cuda"), "") == "cuda"
    assert pf._safe_device_type(SimpleNamespace(device_type=""), "") == ""

    monkeypatch.setattr(pf, "get_device_type", lambda vendor: "npu")
    assert pf._safe_device_type(SimpleNamespace(device_type=""), "ascend") == "npu"

    def _unknown(vendor):
        raise ValueError(vendor)

    monkeypatch.setattr(pf, "get_device_type", _unknown)
    assert pf._safe_device_type(SimpleNamespace(device_type=""), "nope") == ""


def test_use_flagcx(monkeypatch):
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    assert pf._use_flagcx("nvidia") is False

    monkeypatch.setenv("FLAGCX_PATH", "/tmp/flagcx")
    assert pf._use_flagcx("nvidia") is True
    monkeypatch.delenv("FLAGCX_PATH", raising=False)

    class _Library:
        pass

    _install_mod(monkeypatch, "flagcx.api", FLAGCXLibrary=_Library)
    assert pf._use_flagcx("kunlunxin") is True

    _install_mod(monkeypatch, "flagcx.api", FLAGCXLibrary=5)
    assert pf._use_flagcx("kunlunxin") is False

    monkeypatch.setitem(sys.modules, "flagcx.api", None)
    assert pf._use_flagcx("kunlunxin") is False


# ---------------------------------------------------------------------------
# platform detection / capability
# ---------------------------------------------------------------------------


def test_platform_class_attributes():
    assert issubclass(pf.PlatformFL, pf.Platform)
    assert pf.PlatformFL._enum == pf.PlatformEnum.OOT
    assert isinstance(pf.PlatformFL.ray_device_key, str)
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" in (
        pf.PlatformFL.ray_noset_device_env_vars
    )


@pytest.mark.parametrize(
    "vendor,device_type,expected",
    [
        ("iluvatar", "cuda", False),
        ("nvidia", "musa", True),
        ("hygon", "cuda", False),
        ("gcu", "cuda", True),
        ("enflame", "cuda", True),
        ("nvidia", "cuda", True),
        ("amd", "rocm", False),
    ],
)
def test_is_cuda_alike(monkeypatch, vendor, device_type, expected):
    _platform(monkeypatch, vendor=vendor, device_type=device_type)
    assert pf.PlatformFL.is_cuda_alike() is expected


def test_is_cuda(monkeypatch):
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    assert pf.PlatformFL.is_cuda() is True
    _platform(monkeypatch, vendor="iluvatar", device_type="cuda")
    assert pf.PlatformFL.is_cuda() is False


def test_is_rocm_and_sleep(monkeypatch):
    _platform(monkeypatch, vendor="amd", device_type="rocm")
    assert pf.PlatformFL.is_rocm() is True
    assert pf.PlatformFL.is_sleep_mode_available() is False

    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    assert pf.PlatformFL.is_rocm() is False
    assert pf.PlatformFL.is_sleep_mode_available() is True


def test_is_musa_and_gcu(monkeypatch):
    monkeypatch.delattr(torch, "musa", raising=False)
    monkeypatch.delattr(torch, "gcu", raising=False)
    assert pf.PlatformFL.is_musa() is False
    assert pf.PlatformFL.is_gcu() is False

    monkeypatch.setattr(
        torch, "musa", SimpleNamespace(is_available=lambda: True), raising=False
    )
    monkeypatch.setattr(
        torch, "gcu", SimpleNamespace(is_available=lambda: True), raising=False
    )
    assert pf.PlatformFL.is_musa() is True
    assert pf.PlatformFL.is_gcu() is True

    monkeypatch.setattr(
        torch, "musa", SimpleNamespace(is_available=lambda: False), raising=False
    )
    monkeypatch.setattr(
        torch, "gcu", SimpleNamespace(is_available=lambda: False), raising=False
    )
    assert pf.PlatformFL.is_musa() is False
    assert pf.PlatformFL.is_gcu() is False


def test_supported_dtypes():
    dtypes = pf.PlatformFL.supported_dtypes
    assert torch.bfloat16 in dtypes
    assert torch.float16 in dtypes
    assert torch.float32 in dtypes


def test_check_if_supports_dtype_nvidia_old_gpu(monkeypatch):
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    monkeypatch.setattr(
        pf.PlatformFL,
        "has_device_capability",
        classmethod(lambda cls, capability, device_id=0: False),
    )
    monkeypatch.setattr(
        pf.PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: pf.DeviceCapability(major=7, minor=5)),
    )
    monkeypatch.setattr(
        pf.PlatformFL,
        "get_device_name",
        classmethod(lambda cls, device_id=0: "Tesla-T4"),
    )
    with pytest.raises(ValueError) as excinfo:
        pf.PlatformFL.check_if_supports_dtype(torch.bfloat16)
    text = str(excinfo.value)
    assert "Bfloat16" in text and "7.5" in text

    monkeypatch.setattr(
        pf.PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: None),
    )
    with pytest.raises(ValueError) as excinfo:
        pf.PlatformFL.check_if_supports_dtype(torch.bfloat16)
    assert "does not have a compute capability" in str(excinfo.value)


def test_check_if_supports_dtype_no_raise(monkeypatch):
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    monkeypatch.setattr(
        pf.PlatformFL,
        "has_device_capability",
        classmethod(lambda cls, capability, device_id=0: True),
    )
    assert pf.PlatformFL.check_if_supports_dtype(torch.bfloat16) is None

    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert pf.PlatformFL.check_if_supports_dtype(torch.bfloat16) is None


# ---------------------------------------------------------------------------
# device operations
# ---------------------------------------------------------------------------


def test_get_current_memory_usage(monkeypatch):
    fn = MagicMock()
    fn.max_memory_allocated.return_value = 1234
    monkeypatch.setattr(pf.PlatformFL, "torch_device_fn", fn)
    device = torch.device("cpu")
    assert pf.PlatformFL.get_current_memory_usage(device) == 1234
    fn.empty_cache.assert_called_once_with()
    fn.reset_peak_memory_stats.assert_called_once_with(device)
    fn.max_memory_allocated.assert_called_once_with(device)


def test_set_device(monkeypatch):
    fn = MagicMock()
    monkeypatch.setattr(pf.PlatformFL, "torch_device_fn", fn)
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    device = torch.device("cpu")
    assert pf.PlatformFL.set_device(device) is None
    fn.set_device.assert_called_once_with(device)

    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    assert pf.PlatformFL.set_device(device) is None
    assert fn.set_device.call_count == 2


def test_set_device_nvidia_forces_context(monkeypatch):
    fn = MagicMock()
    monkeypatch.setattr(pf.PlatformFL, "torch_device_fn", fn)
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    zeros = MagicMock(wraps=torch.zeros)
    monkeypatch.setattr(torch, "zeros", zeros)
    pf.PlatformFL.set_device(torch.device("cpu"))
    zeros.assert_called_once()


def test_empty_cache_and_name(monkeypatch):
    fn = MagicMock()
    monkeypatch.setattr(pf.PlatformFL, "torch_device_fn", fn)
    pf.PlatformFL.empty_cache()
    fn.empty_cache.assert_called_once_with()

    monkeypatch.setattr(pf.PlatformFL, "device_name", "gcu-0")
    assert pf.PlatformFL.get_device_name() == "gcu-0"


@pytest.mark.parametrize(
    "device_type,expected",
    [
        ("cuda", True),
        ("xpu", True),
        ("npu", True),
        ("musa", True),
        ("gcu", True),
        ("cpu", False),
        ("txda", False),
    ],
)
def test_is_pin_memory_available(monkeypatch, device_type, expected):
    _platform(monkeypatch, device_type=device_type)
    assert pf.PlatformFL.is_pin_memory_available() is expected


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def test_import_kernels_metax_success(monkeypatch):
    _platform(monkeypatch, vendor="metax", device_type="cuda")
    monkeypatch.setattr(pf.Platform, "import_kernels", classmethod(lambda cls: None))
    _install_mod(monkeypatch, "mcoplib._C")
    _install_mod(monkeypatch, "mcoplib._moe_C")
    _install_mod(monkeypatch, "vllm_fl.dispatch.backends.vendor.metax.patches")
    assert pf.PlatformFL.import_kernels() is None


def test_import_kernels_metax_failures(monkeypatch):
    _platform(monkeypatch, vendor="metax", device_type="cuda")
    monkeypatch.setitem(sys.modules, "mcoplib._C", None)
    monkeypatch.setitem(sys.modules, "mcoplib._moe_C", None)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.metax.patches", None
    )
    assert pf.PlatformFL.import_kernels() is None


def test_import_kernels_else_and_musa(monkeypatch):
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    base = MagicMock()
    monkeypatch.setattr(pf.Platform, "import_kernels", classmethod(base))
    apply_musa = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.musa.patch",
        apply_musa_patches=apply_musa,
    )
    assert pf.PlatformFL.import_kernels() is None
    base.assert_called_once()
    apply_musa.assert_not_called()

    _platform(monkeypatch, vendor="mthreads", device_type="musa")
    assert pf.PlatformFL.import_kernels() is None
    apply_musa.assert_called_once()


def test_import_kernels_musa_failure(monkeypatch):
    _platform(monkeypatch, vendor="mthreads", device_type="musa")
    monkeypatch.setattr(pf.Platform, "import_kernels", classmethod(lambda cls: None))

    def _boom():
        raise RuntimeError("musa patch failed")

    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.musa.patch",
        apply_musa_patches=_boom,
    )
    assert pf.PlatformFL.import_kernels() is None


def test_import_ir_kernels(monkeypatch):
    _install_mod(monkeypatch, "vllm.kernels")
    assert pf.PlatformFL.import_ir_kernels() is None


# ---------------------------------------------------------------------------
# apply_config_platform_defaults
# ---------------------------------------------------------------------------


def test_apply_defaults_kunlunxin(monkeypatch, fake_enums):
    CM, _CG = fake_enums
    _platform(monkeypatch, vendor="kunlunxin", device_type="cuda")
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(mode=CM.VLLM_COMPILE))
    pf.PlatformFL.apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.mode is CM.NONE


def test_apply_defaults_non_metax_returns(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(mode="x"))
    pf.PlatformFL.apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.mode == "x"


def test_apply_defaults_metax_none_compilation(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="metax", device_type="cuda")
    ensure = MagicMock(return_value=[])
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.patches",
        ensure_patches=ensure,
    )
    cfg = SimpleNamespace(compilation_config=None)
    pf.PlatformFL.apply_config_platform_defaults(cfg)
    ensure.assert_called_once()


def test_apply_defaults_metax_piecewise(monkeypatch, fake_enums):
    CM, CG = fake_enums
    _platform(monkeypatch, vendor="metax", device_type="cuda")
    ensure = MagicMock(return_value={"leftover"})
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.patches",
        ensure_patches=ensure,
    )
    compilation = SimpleNamespace(
        mode=CM.VLLM_COMPILE,
        cudagraph_mode=None,
        cudagraph_capture_sizes=[],
        cudagraph_mm_encoder=True,
        pass_config=SimpleNamespace(fuse_rope_kvcache_cat_mla=True),
    )
    cfg = SimpleNamespace(compilation_config=compilation)
    pf.PlatformFL.apply_config_platform_defaults(cfg)
    assert compilation.cudagraph_mode is CG.PIECEWISE
    assert compilation.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert compilation.cudagraph_mm_encoder is False
    assert compilation.pass_config.fuse_rope_kvcache_cat_mla is False


def test_apply_defaults_metax_early_returns(monkeypatch, fake_enums):
    CM, CG = fake_enums
    _platform(monkeypatch, vendor="metax", device_type="cuda")
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.patches",
        ensure_patches=MagicMock(return_value=[]),
    )

    mode_none = SimpleNamespace(mode=CM.NONE, cudagraph_mode=None)
    pf.PlatformFL.apply_config_platform_defaults(
        SimpleNamespace(compilation_config=mode_none)
    )
    assert mode_none.cudagraph_mode is None

    explicit = SimpleNamespace(mode=CM.VLLM_COMPILE, cudagraph_mode=CG.PIECEWISE)
    pf.PlatformFL.apply_config_platform_defaults(
        SimpleNamespace(compilation_config=explicit)
    )
    assert explicit.cudagraph_mode is CG.PIECEWISE


def test_apply_defaults_metax_no_pass_config(monkeypatch, fake_enums):
    CM, CG = fake_enums
    _platform(monkeypatch, vendor="metax", device_type="cuda")
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.patches",
        ensure_patches=MagicMock(return_value=[]),
    )
    compilation = SimpleNamespace(
        mode=CM.VLLM_COMPILE,
        cudagraph_mode=None,
        cudagraph_capture_sizes=[1, 2],
        cudagraph_mm_encoder=True,
        pass_config=None,
    )
    cfg = SimpleNamespace(compilation_config=compilation)
    pf.PlatformFL.apply_config_platform_defaults(cfg)
    assert compilation.cudagraph_mode is CG.PIECEWISE
    assert compilation.cudagraph_capture_sizes == [1, 2]


# ---------------------------------------------------------------------------
# check_and_update_config
# ---------------------------------------------------------------------------


def test_check_update_gcu_hybrid_spec_align_raises(monkeypatch, fake_enums, gcu_deps):
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    cfg = _make_vllm_config()
    cfg.model_config.is_hybrid = True
    cfg.speculative_config = object()
    cfg.cache_config.mamba_cache_mode = "align"
    with pytest.raises(ValueError, match="speculative decoding"):
        pf.PlatformFL.check_and_update_config(cfg)


@pytest.mark.parametrize("vendor", ["gcu", "enflame"])
def test_check_update_gcu_basic(monkeypatch, fake_enums, gcu_deps, vendor):
    _platform(monkeypatch, vendor=vendor, device_type="gcu")
    cfg = _make_vllm_config()
    pf.PlatformFL.check_and_update_config(cfg)
    assert gcu_deps["configure_codegen_cache"] == 1
    assert gcu_deps["patch_mm_encoder_attention"] == 1
    assert cfg.parallel_config.worker_cls == (
        "vllm_fl.dispatch.backends.vendor.gcu.worker.worker.WorkerGCU"
    )
    assert cfg.parallel_config.disable_custom_all_reduce is True
    assert cfg.compilation_config.inductor_compile_config["combo_kernels"] is False
    assert (
        cfg.compilation_config.inductor_compile_config["benchmark_combo_kernel"]
        is False
    )
    assert cfg.compilation_config.compile_sizes == []


def test_check_update_gcu_no_inductor_combo(monkeypatch, fake_enums, gcu_deps):
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    cfg = _make_vllm_config()
    cfg.compilation_config.mode = FakeCompilationMode.NONE
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.compilation_config.inductor_compile_config == {}


def test_check_update_iluvatar(monkeypatch, fake_enums, iluvatar_deps):
    _platform(monkeypatch, vendor="iluvatar", device_type="cuda")
    cfg = _make_vllm_config()
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == (
        "vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker.IluvatarWorkerFL"
    )
    assert iluvatar_deps["configure_compilation"][0] is cfg.compilation_config


def test_check_update_npu(monkeypatch, fake_enums, npu_deps):
    _platform(monkeypatch, vendor="ascend", device_type="npu", device_name="npu")
    cfg = _make_vllm_config()
    cfg.cache_config.block_size = None
    npu_deps["prepare_result"] = True
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 128
    cfg.model_config.verify_with_parallel_config.assert_called_once_with(
        cfg.parallel_config
    )
    assert "refresh_block_size" in npu_deps
    assert "configure_ascend_compilation" in npu_deps
    assert cfg.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"


def test_check_update_npu_prepare_false(monkeypatch, fake_enums, npu_deps):
    _platform(monkeypatch, vendor="ascend", device_type="npu", device_name="npu")
    cfg = _make_vllm_config()
    cfg.cache_config.block_size = 128
    npu_deps["prepare_result"] = False
    pf.PlatformFL.check_and_update_config(cfg)
    cfg.model_config.verify_with_parallel_config.assert_not_called()


def test_check_update_musa_v2_runner_raises(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="mthreads", device_type="musa")
    cfg = _make_vllm_config()
    cfg.use_v2_model_runner = True
    with pytest.raises(ValueError, match="ModelRunnerFL"):
        pf.PlatformFL.check_and_update_config(cfg)


def test_check_update_musa_basic(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="mthreads", device_type="musa")
    monkeypatch.setattr(pf.PlatformFL, "dist_backend", "mccl")
    cfg = _make_vllm_config()
    cfg.cache_config.block_size = None
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.compilation_config.cudagraph_mode = FakeCUDAGraphMode.FULL
    cfg.compilation_config.splitting_ops = ["vllm::attention"]
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 64
    assert cfg.compilation_config.inductor_compile_config["combo_kernels"] is False
    assert cfg.compilation_config.cudagraph_mode is FakeCUDAGraphMode.PIECEWISE
    cfg.compilation_config.set_splitting_ops_for_v1.assert_called_once()
    assert "vllm::all_reduce" in cfg.compilation_config.splitting_ops


def test_check_update_txda_spec_raises(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="tsingmicro", device_type="txda")
    cfg = _make_vllm_config()
    cfg.speculative_config = object()
    with pytest.raises(ValueError, match="speculative decoding"):
        pf.PlatformFL.check_and_update_config(cfg)


def test_check_update_txda_context_parallel_raises(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="tsingmicro", device_type="txda")
    cfg = _make_vllm_config()
    cfg.parallel_config.prefill_context_parallel_size = 2
    with pytest.raises(ValueError, match="context parallelism"):
        pf.PlatformFL.check_and_update_config(cfg)


def test_check_update_txda_world_size_requires_flagcx(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="tsingmicro", device_type="txda")
    monkeypatch.setattr(pf.PlatformFL, "dist_backend", "nccl")
    cfg = _make_vllm_config()
    cfg.parallel_config.world_size_across_dp = 4
    with pytest.raises(ValueError, match="FlagCX"):
        pf.PlatformFL.check_and_update_config(cfg)


def test_check_update_txda_basic(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="tsingmicro", device_type="txda")
    monkeypatch.setattr(pf.PlatformFL, "dist_backend", "flagcx")
    cfg = _make_vllm_config()
    cfg.parallel_config.world_size_across_dp = None
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.compilation_config.mode = FakeCompilationMode.NONE
    cfg.compilation_config.cudagraph_mode = FakeCUDAGraphMode.NONE
    cfg.cache_config.block_size = None
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.parallel_config.disable_custom_all_reduce is True
    assert cfg.cache_config.block_size == 16
    assert cfg.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"


def test_check_update_txda_graph_raises(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="tsingmicro", device_type="txda")
    cfg = _make_vllm_config()
    cfg.compilation_config.cudagraph_mode = FakeCUDAGraphMode.PIECEWISE
    cfg.compilation_config.mode = FakeCompilationMode.NONE
    with pytest.raises(ValueError, match="graph mode"):
        pf.PlatformFL.check_and_update_config(cfg)


def test_check_update_txda_compile_raises(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="tsingmicro", device_type="txda")
    cfg = _make_vllm_config()
    cfg.compilation_config.cudagraph_mode = FakeCUDAGraphMode.NONE
    cfg.compilation_config.mode = FakeCompilationMode.VLLM_COMPILE
    with pytest.raises(ValueError, match="eager execution only"):
        pf.PlatformFL.check_and_update_config(cfg)


def test_check_update_kunlunxin_spec_align_raises(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="kunlunxin", device_type="cuda")
    cfg = _make_vllm_config()
    cfg.speculative_config = object()
    cfg.model_config.is_hybrid = True
    cfg.cache_config.mamba_cache_mode = "align"
    with pytest.raises(ValueError, match="Mamba"):
        pf.PlatformFL.check_and_update_config(cfg)


def test_check_update_kunlunxin_basic(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="kunlunxin", device_type="cuda")
    cfg = _make_vllm_config()
    cfg.cache_config.block_size = None
    cfg.scheduler_config.async_scheduling = None
    cfg.kernel_config = SimpleNamespace(enable_jit_warmup=True)
    cfg.compilation_config.cudagraph_mode = FakeCUDAGraphMode.FULL
    cfg.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8, 16]
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 128
    assert cfg.scheduler_config.async_scheduling is False
    assert cfg.compilation_config.mode is FakeCompilationMode.NONE
    assert cfg.compilation_config.cudagraph_mm_encoder is False
    assert cfg.kernel_config.enable_jit_warmup is False
    assert cfg.compilation_config.cudagraph_mode is FakeCUDAGraphMode.FULL_DECODE_ONLY
    assert cfg.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]
    assert cfg.compilation_config.max_cudagraph_capture_size == 8


def test_check_update_kunlunxin_graph_fallback(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="kunlunxin", device_type="cuda")
    cfg = _make_vllm_config()
    cfg.compilation_config.cudagraph_mode = FakeCUDAGraphMode.FULL
    cfg.compilation_config.cudagraph_capture_sizes = [16, 32]
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.compilation_config.cudagraph_mode is FakeCUDAGraphMode.NONE
    assert cfg.compilation_config.cudagraph_capture_sizes == []
    assert cfg.compilation_config.max_cudagraph_capture_size == 0


def test_check_update_metax(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="metax", device_type="cuda")
    cfg = _make_vllm_config()
    cfg.attention_config = SimpleNamespace(
        use_cudnn_prefill=True,
        use_trtllm_ragged_deepseek_prefill=True,
        use_trtllm_attention=True,
        disable_flashinfer_prefill=False,
    )
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.model_config.disable_cascade_attn is True
    assert cfg.attention_config.use_cudnn_prefill is False
    assert cfg.attention_config.use_trtllm_ragged_deepseek_prefill is False
    assert cfg.attention_config.use_trtllm_attention is False
    assert cfg.attention_config.disable_flashinfer_prefill is True


def test_check_update_mm_prefix_lm(monkeypatch, fake_enums, gcu_deps):
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    cfg = _make_vllm_config()
    cfg.model_config.is_mm_prefix_lm = True
    cfg.scheduler_config.is_multimodal_model = True
    cfg.scheduler_config.disable_chunked_mm_input = False
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.scheduler_config.disable_chunked_mm_input is True


def test_check_update_mla_block_size(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    cfg = _make_vllm_config()
    cfg.model_config.use_mla = True
    cfg.cache_config.block_size = 40
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 64


def test_check_update_deepep_wide(monkeypatch, fake_enums):
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    cfg = _make_vllm_config()
    cfg.parallel_config.all2all_backend = "deepep_high_throughput"
    cfg.parallel_config.data_parallel_size = 2
    cfg.compilation_config.cudagraph_mode = FakeCUDAGraphMode.FULL
    pf.PlatformFL.check_and_update_config(cfg)
    assert cfg.compilation_config.cudagraph_mode is FakeCUDAGraphMode.NONE


# ---------------------------------------------------------------------------
# attention backend selection
# ---------------------------------------------------------------------------


class _ABEMember:
    def __init__(self, name, backend_cls=None):
        self.name = name
        self._backend_cls = backend_cls

    def get_class(self):
        if self._backend_cls is None:
            raise ImportError(self.name)
        return self._backend_cls

    def __repr__(self):
        return f"<ABE {self.name}>"


class _FakeFlashBackendClass:
    @staticmethod
    def supports_head_size(head_size):
        return isinstance(head_size, int) and 0 < head_size <= 256

    @staticmethod
    def supports_dtype(dtype):
        return True


class _FakeABE:
    TORCH_SDPA = _ABEMember("torch_sdpa")
    FLASH_ATTN = _ABEMember("flash_attn", _FakeFlashBackendClass)


class _FakeGCUAttentionBackend:
    @staticmethod
    def supports_head_size(head_size):
        return isinstance(head_size, int) and head_size > 0 and head_size % 8 == 0

    @staticmethod
    def supports_dtype(dtype):
        return dtype in (torch.float16, torch.bfloat16)


def _attn_selector(use_mla=False, use_sparse=False):
    return SimpleNamespace(use_mla=use_mla, use_sparse=use_sparse)


def test_get_attn_backend_cls(monkeypatch):
    dispatch = pytest.importorskip("vllm_fl.dispatch")
    seen = {}

    def _call_op(op_name, **kwargs):
        seen["op"] = op_name
        seen["kwargs"] = kwargs
        return "fake.backend.Cls"

    monkeypatch.setattr(dispatch, "call_op", _call_op)
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    result = pf.PlatformFL.get_attn_backend_cls(
        None, _attn_selector(use_mla=True, use_sparse=False), num_heads=8
    )
    assert result == "fake.backend.Cls"
    assert seen["op"] == "attention_backend"
    assert seen["kwargs"] == {"use_mla": True, "use_sparse": False}


def test_get_attn_backend_cls_amd_sparse(monkeypatch):
    dispatch = pytest.importorskip("vllm_fl.dispatch")
    monkeypatch.setattr(dispatch, "call_op", MagicMock())
    _platform(monkeypatch, vendor="amd", device_type="rocm")
    with pytest.raises(NotImplementedError):
        pf.PlatformFL.get_attn_backend_cls(
            None, _attn_selector(use_mla=True, use_sparse=True)
        )


def test_get_supported_vit_attn_backends_default(monkeypatch):
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    backends = pf.PlatformFL.get_supported_vit_attn_backends()
    assert pf.AttentionBackendEnum.TORCH_SDPA in backends
    assert pf.AttentionBackendEnum.FLASH_ATTN in backends


def test_get_supported_vit_attn_backends_amd(monkeypatch):
    _platform(monkeypatch, vendor="amd", device_type="rocm")
    fake_rocm = _install_mod(
        monkeypatch,
        "vllm.platforms.rocm",
        RocmPlatform=SimpleNamespace(
            get_supported_vit_attn_backends=lambda: ["rocm-vit"]
        ),
    )
    backends = pf.PlatformFL.get_supported_vit_attn_backends()
    assert backends == ["rocm-vit"]
    assert fake_rocm.RocmPlatform is not None

    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", None)
    backends = pf.PlatformFL.get_supported_vit_attn_backends()
    assert backends == [pf.AttentionBackendEnum.TORCH_SDPA]


def test_get_vit_attn_backend_gcu_auto(monkeypatch):
    _install_mod(
        monkeypatch,
        "vllm_fl.attention.utils",
        patch_mm_encoder_attention=MagicMock(),
    )
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.gcu.impl.attention",
        AttentionGCUBackend=_FakeGCUAttentionBackend,
    )
    monkeypatch.setattr(pf, "AttentionBackendEnum", _FakeABE)
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert (
        pf.PlatformFL.get_vit_attn_backend(64, torch.bfloat16)
        is _FakeABE.FLASH_ATTN
    )
    assert (
        pf.PlatformFL.get_vit_attn_backend(0, torch.bfloat16)
        is _FakeABE.TORCH_SDPA
    )
    assert (
        pf.PlatformFL.get_vit_attn_backend(64, torch.int8)
        is _FakeABE.TORCH_SDPA
    )


def test_get_vit_attn_backend_gcu_explicit(monkeypatch):
    _install_mod(
        monkeypatch,
        "vllm_fl.attention.utils",
        patch_mm_encoder_attention=MagicMock(),
    )
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.gcu.impl.attention",
        AttentionGCUBackend=_FakeGCUAttentionBackend,
    )
    monkeypatch.setattr(pf, "AttentionBackendEnum", _FakeABE)
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert (
        pf.PlatformFL.get_vit_attn_backend(
            64, torch.bfloat16, backend=_FakeABE.FLASH_ATTN
        )
        is _FakeABE.FLASH_ATTN
    )
    with pytest.raises(ValueError, match="FlashAttention"):
        pf.PlatformFL.get_vit_attn_backend(
            0, torch.bfloat16, backend=_FakeABE.FLASH_ATTN
        )
    with pytest.raises(AssertionError):
        pf.PlatformFL.get_vit_attn_backend(
            64, torch.bfloat16, backend=object()
        )


def test_get_vit_attn_backend_amd(monkeypatch):
    _install_mod(
        monkeypatch,
        "vllm_fl.attention.utils",
        patch_mm_encoder_attention=MagicMock(),
    )
    monkeypatch.setattr(pf, "AttentionBackendEnum", _FakeABE)
    _platform(monkeypatch, vendor="amd", device_type="rocm")
    _install_mod(
        monkeypatch,
        "vllm.platforms.rocm",
        RocmPlatform=SimpleNamespace(
            get_vit_attn_backend=lambda head_size, dtype, backend: "rocm-backend"
        ),
    )
    assert (
        pf.PlatformFL.get_vit_attn_backend(64, torch.bfloat16) == "rocm-backend"
    )

    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", None)
    assert (
        pf.PlatformFL.get_vit_attn_backend(64, torch.bfloat16)
        is _FakeABE.TORCH_SDPA
    )


def test_get_vit_attn_backend_generic_capability(monkeypatch):
    _install_mod(
        monkeypatch,
        "vllm_fl.attention.utils",
        patch_mm_encoder_attention=MagicMock(),
    )
    monkeypatch.setattr(pf, "AttentionBackendEnum", _FakeABE)
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    monkeypatch.setattr(
        pf.PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: pf.DeviceCapability(major=8, minor=6)),
    )
    assert (
        pf.PlatformFL.get_vit_attn_backend(64, torch.bfloat16)
        is _FakeABE.FLASH_ATTN
    )

    monkeypatch.setattr(
        pf.PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: pf.DeviceCapability(major=7, minor=0)),
    )
    assert (
        pf.PlatformFL.get_vit_attn_backend(64, torch.bfloat16)
        is _FakeABE.TORCH_SDPA
    )


# ---------------------------------------------------------------------------
# wrappers / communicators / config helpers
# ---------------------------------------------------------------------------


def test_get_vit_attn_backend_generic_flash_import_error(monkeypatch):
    _install_mod(
        monkeypatch,
        "vllm_fl.attention.utils",
        patch_mm_encoder_attention=MagicMock(),
    )
    monkeypatch.setattr(pf, "AttentionBackendEnum", _FakeABE)
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    monkeypatch.setattr(
        pf.PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: pf.DeviceCapability(major=8, minor=0)),
    )
    monkeypatch.setattr(
        _FakeABE, "FLASH_ATTN", _ABEMember("flash_attn_no_class")
    )
    assert (
        pf.PlatformFL.get_vit_attn_backend(64, torch.bfloat16)
        is _FakeABE.TORCH_SDPA
    )


def test_get_default_ir_op_priority_gcu_device_native(monkeypatch, fake_enums):
    CM, _CG = fake_enums
    _install_kernel_module(monkeypatch, _FakeIrOpPriorityConfig)
    _FakeIrOpPriorityConfig.captured = []
    _platform(monkeypatch, vendor="nvidia", device_type="gcu")
    monkeypatch.setattr(
        pf, "_vllm_c_available", lambda: (_ for _ in ()).throw(AssertionError)
    )
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(backend="eager", mode=CM.NONE)
    )
    pf.PlatformFL.get_default_ir_op_priority(cfg)
    assert _FakeIrOpPriorityConfig.captured[-1][0] == ["native"]


def test_get_punica_wrapper():
    assert pf.PlatformFL.get_punica_wrapper() == (
        "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
    )


def test_get_device_communicator_cls(monkeypatch):
    _platform(monkeypatch, vendor="iluvatar", device_type="cuda")
    monkeypatch.setattr(pf.PlatformFL, "dist_backend", "flagcx")
    assert pf.PlatformFL.get_device_communicator_cls() == (
        "vllm_fl.dispatch.backends.vendor.iluvatar.communicator."
        "IluvatarCommunicatorFL"
    )

    _platform(monkeypatch, vendor="amd", device_type="rocm")
    assert pf.PlatformFL.get_device_communicator_cls() == (
        "vllm_fl.distributed.cuda_flagcx_communicator.CudaFlagcxCommunicatorFL"
    )

    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    assert pf.PlatformFL.get_device_communicator_cls() == (
        "vllm_fl.distributed.communicator.CommunicatorFL"
    )

    monkeypatch.setattr(pf.PlatformFL, "dist_backend", "nccl")
    assert pf.PlatformFL.get_device_communicator_cls() == (
        "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"
    )


def test_get_static_graph_wrapper_cls(monkeypatch):
    _platform(monkeypatch, vendor="mthreads", device_type="musa")
    assert pf.PlatformFL.get_static_graph_wrapper_cls() == (
        "vllm_fl.dispatch.backends.vendor.musa.graph.MUSAGraphWrapper"
    )

    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert pf.PlatformFL.get_static_graph_wrapper_cls() == (
        "vllm_fl.compilation.graph.GraphWrapper"
    )

    _platform(monkeypatch, vendor="amd", device_type="rocm")
    vc = pytest.importorskip("vllm.config")
    monkeypatch.setattr(
        vc,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(use_v2_model_runner=True),
        raising=False,
    )
    assert pf.PlatformFL.get_static_graph_wrapper_cls() == (
        "vllm.compilation.cuda_graph.CUDAGraphWrapper"
    )

    monkeypatch.setattr(
        vc, "get_current_vllm_config_or_none", lambda: None, raising=False
    )
    assert pf.PlatformFL.get_static_graph_wrapper_cls() == (
        "vllm_fl.compilation.graph.GraphWrapper"
    )


def test_get_compile_backend(monkeypatch):
    monkeypatch.setattr(
        pf.Platform, "get_compile_backend", classmethod(lambda cls: "base-backend")
    )
    _platform(monkeypatch, device_type="npu")
    assert pf.PlatformFL.get_compile_backend() == "eager"
    _platform(monkeypatch, device_type="musa")
    assert pf.PlatformFL.get_compile_backend() == "eager"
    _platform(monkeypatch, device_type="cuda")
    assert pf.PlatformFL.get_compile_backend() == "base-backend"


@pytest.mark.parametrize(
    "vendor,expected",
    [
        ("nvidia", True),
        ("ascend", True),
        ("metax", True),
        ("hygon", True),
        ("mthreads", True),
        ("iluvatar", True),
        ("thead", True),
        ("gcu", True),
        ("enflame", True),
        ("kunlunxin", True),
        ("amd", True),
        ("other", False),
    ],
)
def test_support_static_graph_mode(monkeypatch, vendor, expected):
    _platform(monkeypatch, vendor=vendor)
    assert pf.PlatformFL.support_static_graph_mode() is expected


def test_insert_and_swap_blocks():
    src = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    dst = torch.full((4, 2), -1.0)
    src_idx = torch.tensor([0, 2])
    dst_idx = torch.tensor([1, 0])
    pf.PlatformFL.insert_blocks_to_device(src, dst, src_idx, dst_idx)
    assert torch.equal(dst[:, 1], src[:, 0])
    assert torch.equal(dst[:, 0], src[:, 2])

    host = torch.full((4, 2), -1.0)
    pf.PlatformFL.swap_out_blocks_to_host(src, host, src_idx, dst_idx)
    assert torch.equal(host[:, 1], src[:, 0])
    assert torch.equal(host[:, 0], src[:, 2])


def test_align_hybrid_block_size(monkeypatch):
    monkeypatch.setattr(
        pf.Platform,
        "_align_hybrid_block_size",
        classmethod(lambda cls, cfg, backend: "base"),
    )
    calls = []
    _install_mod(
        monkeypatch,
        "vllm_fl.kv_cache.ascend.hybrid",
        uses_dense_hybrid_cache=lambda model_config: True,
        align_hybrid_block_size=lambda cfg: calls.append(cfg),
    )
    _platform(monkeypatch, vendor="ascend", device_type="npu")
    cfg = SimpleNamespace(model_config=object())
    pf.PlatformFL._align_hybrid_block_size(cfg, object())
    assert calls == [cfg]

    _install_mod(
        monkeypatch,
        "vllm_fl.kv_cache.ascend.hybrid",
        uses_dense_hybrid_cache=lambda model_config: False,
        align_hybrid_block_size=MagicMock(),
    )
    assert pf.PlatformFL._align_hybrid_block_size(cfg, object()) == "base"

    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert pf.PlatformFL._align_hybrid_block_size(cfg, object()) == "base"


def test_misc_support_flags(monkeypatch):
    assert pf.PlatformFL.support_hybrid_kv_cache() is True
    assert pf.PlatformFL.opaque_attention_op() is True
    assert pf.PlatformFL.check_runner_kv_caches_multi_layer() is None
    assert pf.PlatformFL.is_integrated_gpu() is False


@pytest.mark.parametrize(
    "vendor,dist_backend,expected",
    [
        ("hygon", "eccl", False),
        ("amd", "rccl", False),
        ("gcu", "flagcx", False),
        ("nvidia", "nccl", True),
        ("gcu", "eccl", True),
    ],
)
def test_use_custom_allreduce(monkeypatch, vendor, dist_backend, expected):
    _platform(monkeypatch, vendor=vendor)
    monkeypatch.setattr(pf.PlatformFL, "dist_backend", dist_backend)
    assert pf.PlatformFL.use_custom_allreduce() is expected


# ---------------------------------------------------------------------------
# pre-registration and vendor hooks
# ---------------------------------------------------------------------------


def test_pre_register_npu(monkeypatch):
    _platform(monkeypatch, vendor="ascend", device_type="npu", device_name="npu")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    _install_mod(monkeypatch, "vllm_fl.dispatch.backends.vendor.ascend")
    assert pf.PlatformFL.pre_register_and_update() is None
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"


def test_pre_register_npu_unset(monkeypatch):
    _platform(monkeypatch, vendor="ascend", device_type="npu", device_name="npu")
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    _install_mod(monkeypatch, "vllm_fl.dispatch.backends.vendor.ascend")
    pf.PlatformFL.pre_register_and_update()
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"


def test_pre_register_iluvatar(monkeypatch):
    _platform(monkeypatch, vendor="iluvatar", device_type="cuda", device_name="cuda")
    chained = MagicMock()
    moe = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar",
        patch_triton_chained_or_for_iluvatar=chained,
    )
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.iluvatar.moe_router",
        patch_moe_router=moe,
    )
    assert pf.PlatformFL.pre_register_and_update() is None
    chained.assert_called_once()
    moe.assert_called_once()


def test_pre_register_amd(monkeypatch):
    _platform(monkeypatch, vendor="amd", device_type="rocm", device_name="cuda")
    disable = MagicMock()
    monkeypatch.setattr(pf.PlatformFL, "_disable_rocm_native_kernels", disable)
    assert pf.PlatformFL.pre_register_and_update() is None
    disable.assert_called_once()


def test_disable_rocm_native_kernels(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "1")
    pf.PlatformFL._disable_rocm_native_kernels()
    assert os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] == "1"

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.delenv("VLLM_ROCM_USE_SKINNY_GEMM", raising=False)
    pf.PlatformFL._disable_rocm_native_kernels()
    assert os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] == "0"


def test_supports_fp8(monkeypatch):
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    monkeypatch.setattr(
        pf.PlatformFL,
        "has_device_capability",
        classmethod(lambda cls, capability, device_id=0: True),
    )
    assert pf.PlatformFL.supports_fp8() is True

    monkeypatch.setattr(
        pf.PlatformFL,
        "has_device_capability",
        classmethod(lambda cls, capability, device_id=0: False),
    )
    assert pf.PlatformFL.supports_fp8() is False

    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert pf.PlatformFL.supports_fp8() is False


# ---------------------------------------------------------------------------
# device identity / memory / capability
# ---------------------------------------------------------------------------


def test_get_device_uuid(monkeypatch):
    _platform(monkeypatch, vendor="iluvatar", device_type="cuda")
    assert pf.PlatformFL.get_device_uuid(3) == "ILUVATAR-3"

    pynvml = _install_mod(
        monkeypatch,
        "pynvml",
        nvmlInit=MagicMock(),
        nvmlShutdown=MagicMock(),
        nvmlDeviceGetHandleByIndex=MagicMock(return_value="handle"),
        nvmlDeviceGetUUID=MagicMock(return_value="GPU-uuid"),
    )
    monkeypatch.setattr(
        pf.PlatformFL,
        "visible_device_id_to_physical_device_id",
        classmethod(lambda cls, device_id: device_id + 1),
    )
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    assert pf.PlatformFL.get_device_uuid(2) == "GPU-uuid"
    pynvml.nvmlInit.assert_called_once()
    pynvml.nvmlShutdown.assert_called_once()

    _platform(monkeypatch, vendor="ascend", device_type="npu")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3,4")
    assert pf.PlatformFL.get_device_uuid(1) == "NPU-4"
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    assert pf.PlatformFL.get_device_uuid(1) == "NPU-1"

    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert pf.PlatformFL.get_device_uuid(7) == "gcu-7"


def test_device_memory_and_units(monkeypatch):
    fn = MagicMock()
    fn.get_device_properties.return_value = SimpleNamespace(
        total_memory=1024, multi_processor_count=64
    )
    monkeypatch.setattr(pf.PlatformFL, "torch_device_fn", fn)
    assert pf.PlatformFL.get_device_total_memory(1) == 1024
    assert pf.PlatformFL.num_compute_units(1) == 64
    fn.get_device_properties.assert_called_with(1)


def test_get_device_capability_npu_like(monkeypatch):
    _platform(monkeypatch, device_type="npu")
    assert pf.PlatformFL.get_device_capability() is None
    _platform(monkeypatch, device_type="ptpu")
    assert pf.PlatformFL.get_device_capability() is None
    _platform(monkeypatch, device_type="txda")
    assert pf.PlatformFL.get_device_capability() is None


def test_get_device_capability_musa_gcu_default(monkeypatch):
    _platform(monkeypatch, device_type="musa")
    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(get_device_capability=lambda device_id: (5, 1)),
        raising=False,
    )
    cap = pf.PlatformFL.get_device_capability(0)
    assert (cap.major, cap.minor) == (5, 1)

    _platform(monkeypatch, device_type="gcu")
    monkeypatch.delattr(torch, "gcu", raising=False)
    assert pf.PlatformFL.get_device_capability() is None

    monkeypatch.setattr(
        torch,
        "gcu",
        SimpleNamespace(get_device_capability=lambda device_id: (7, 2)),
        raising=False,
    )
    cap = pf.PlatformFL.get_device_capability(0)
    assert (cap.major, cap.minor) == (7, 2)

    _platform(monkeypatch, device_type="cuda")
    fn = MagicMock()
    fn.get_device_capability.return_value = (8, 6)
    monkeypatch.setattr(pf.PlatformFL, "torch_device_fn", fn)
    cap = pf.PlatformFL.get_device_capability(0)
    assert (cap.major, cap.minor) == (8, 6)


def test_support_deep_gemm(monkeypatch):
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    monkeypatch.setattr(
        pf.PlatformFL,
        "is_device_capability",
        classmethod(lambda cls, capability, device_id=0: True),
    )
    assert pf.PlatformFL.support_deep_gemm() is True

    monkeypatch.setattr(
        pf.PlatformFL,
        "is_device_capability",
        classmethod(lambda cls, capability, device_id=0: False),
    )
    monkeypatch.setattr(
        pf.PlatformFL,
        "is_device_capability_family",
        classmethod(lambda cls, capability, device_id=0: capability == 100),
    )
    assert pf.PlatformFL.support_deep_gemm() is True

    monkeypatch.setattr(
        pf.PlatformFL,
        "is_device_capability_family",
        classmethod(lambda cls, capability, device_id=0: False),
    )
    assert pf.PlatformFL.support_deep_gemm() is False

    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert pf.PlatformFL.support_deep_gemm() is False


def test_use_custom_op_collectives(monkeypatch):
    _platform(monkeypatch, vendor="tsingmicro", device_type="txda")
    monkeypatch.setattr(pf.PlatformFL, "dist_backend", "tccl")
    assert pf.PlatformFL.use_custom_op_collectives() is True

    _platform(monkeypatch, vendor="enflame", device_type="gcu")
    assert pf.PlatformFL.use_custom_op_collectives() is True

    _platform(monkeypatch, vendor="mthreads", device_type="musa")
    monkeypatch.setattr(pf.PlatformFL, "dist_backend", "mccl")
    assert pf.PlatformFL.use_custom_op_collectives() is True
    monkeypatch.setattr(pf.PlatformFL, "dist_backend", "other")
    assert pf.PlatformFL.use_custom_op_collectives() is False

    _platform(monkeypatch, vendor="amd", device_type="rocm")
    monkeypatch.setattr(pf.PlatformFL, "dist_backend", "flagcx")
    assert pf.PlatformFL.use_custom_op_collectives() is True

    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert pf.PlatformFL.use_custom_op_collectives() is False


# ---------------------------------------------------------------------------
# IR op priority
# ---------------------------------------------------------------------------


class _FakeIrOpPriorityConfig:
    captured = []

    @classmethod
    def with_default(cls, default, **kwargs):
        cls.captured.append((list(default), kwargs))
        return ("priority", tuple(default), kwargs)


def _install_kernel_module(monkeypatch, config_cls):
    mod = _install_mod(
        monkeypatch, "vllm.config.kernel", IrOpPriorityConfig=config_cls
    )
    return mod


def test_get_default_ir_op_priority_kunlunxin(monkeypatch, fake_enums):
    _install_kernel_module(monkeypatch, _FakeIrOpPriorityConfig)
    _FakeIrOpPriorityConfig.captured = []
    _platform(monkeypatch, vendor="kunlunxin", device_type="cuda")
    cfg = SimpleNamespace(compilation_config=SimpleNamespace())
    assert pf.PlatformFL.get_default_ir_op_priority(cfg) is not None
    assert _FakeIrOpPriorityConfig.captured[-1][0] == ["native"]


def test_get_default_ir_op_priority_inductor(monkeypatch, fake_enums):
    CM, _CG = fake_enums
    _install_kernel_module(monkeypatch, _FakeIrOpPriorityConfig)
    _FakeIrOpPriorityConfig.captured = []
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(backend="inductor", mode=CM.VLLM_COMPILE)
    )
    pf.PlatformFL.get_default_ir_op_priority(cfg)
    assert _FakeIrOpPriorityConfig.captured[-1][0] == ["native"]


def test_get_default_ir_op_priority_vllm_c(monkeypatch, fake_enums):
    CM, _CG = fake_enums
    _install_kernel_module(monkeypatch, _FakeIrOpPriorityConfig)
    _FakeIrOpPriorityConfig.captured = []
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    monkeypatch.setattr(pf, "_vllm_c_available", lambda: True)
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(backend="eager", mode=CM.NONE)
    )
    pf.PlatformFL.get_default_ir_op_priority(cfg)
    assert _FakeIrOpPriorityConfig.captured[-1][0] == ["vllm_c", "native"]


def test_get_default_ir_op_priority_no_vllm_c(monkeypatch, fake_enums):
    CM, _CG = fake_enums
    _install_kernel_module(monkeypatch, _FakeIrOpPriorityConfig)
    _FakeIrOpPriorityConfig.captured = []
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    monkeypatch.setattr(pf, "_vllm_c_available", lambda: False)
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(backend="eager", mode=CM.NONE)
    )
    pf.PlatformFL.get_default_ir_op_priority(cfg)
    assert _FakeIrOpPriorityConfig.captured[-1][0] == ["native"]


def test_get_default_ir_op_priority_amd_registered(monkeypatch, fake_enums):
    CM, _CG = fake_enums
    _install_kernel_module(monkeypatch, _FakeIrOpPriorityConfig)
    _FakeIrOpPriorityConfig.captured = []
    _platform(monkeypatch, vendor="amd", device_type="rocm")
    monkeypatch.setattr(
        pf.PlatformFL, "_vllm_c_is_registered_provider", lambda: True
    )
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(backend="eager", mode=CM.NONE)
    )
    pf.PlatformFL.get_default_ir_op_priority(cfg)
    assert _FakeIrOpPriorityConfig.captured[-1][0] == ["vllm_c", "native"]


class _FakeIrOp:
    def __init__(self, impls=None):
        self.impls = {} if impls is None else impls


@dataclasses.dataclass
class _IrCfgTwo:
    rms_norm: object = None
    gelu_and_mul: object = None


@dataclasses.dataclass
class _IrCfgEmpty:
    pass


def _install_ir_modules(monkeypatch, cfg_cls, ops_attrs):
    _install_mod(monkeypatch, "vllm.config.kernel", IrOpPriorityConfig=cfg_cls)
    op_mod = _install_mod(monkeypatch, "vllm.ir.op", IrOp=_FakeIrOp)
    ops_mod = _install_mod(monkeypatch, "vllm.ir.ops", **ops_attrs)
    _install_mod(monkeypatch, "vllm.ir", op=op_mod, ops=ops_mod)
    return ops_mod


def test_vllm_c_is_registered_provider_true(monkeypatch):
    _install_ir_modules(
        monkeypatch,
        _IrCfgTwo,
        {
            "rms_norm": _FakeIrOp({"vllm_c": object()}),
            "gelu_and_mul": _FakeIrOp({"vllm_c": object()}),
        },
    )
    assert pf.PlatformFL._vllm_c_is_registered_provider() is True


def test_vllm_c_is_registered_provider_missing_impl(monkeypatch):
    _install_ir_modules(
        monkeypatch,
        _IrCfgTwo,
        {
            "rms_norm": _FakeIrOp({"vllm_c": object()}),
            "gelu_and_mul": _FakeIrOp({"native": object()}),
        },
    )
    assert pf.PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_is_registered_provider_not_ir_op(monkeypatch):
    _install_ir_modules(
        monkeypatch,
        _IrCfgTwo,
        {"rms_norm": object(), "gelu_and_mul": _FakeIrOp({"vllm_c": object()})},
    )
    assert pf.PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_is_registered_provider_impls_raises(monkeypatch):
    class _Broken(_FakeIrOp):
        @property
        def impls(self):
            raise RuntimeError("boom")

    _install_ir_modules(
        monkeypatch,
        _IrCfgTwo,
        {"rms_norm": _Broken(), "gelu_and_mul": _FakeIrOp({"vllm_c": object()})},
    )
    assert pf.PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_is_registered_provider_fields_type_error(monkeypatch):
    class _NotDataclass:
        pass

    _install_ir_modules(monkeypatch, _NotDataclass, {})
    assert pf.PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_is_registered_provider_no_fields(monkeypatch):
    _install_ir_modules(monkeypatch, _IrCfgEmpty, {})
    assert pf.PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_is_registered_provider_import_error(monkeypatch):
    with _block_import("vllm.ir", ImportError("no ir")):
        assert pf.PlatformFL._vllm_c_is_registered_provider() is False


# ---------------------------------------------------------------------------
# PDL / NVLink / seeding
# ---------------------------------------------------------------------------


def test_is_arch_support_pdl(monkeypatch):
    _platform(monkeypatch, vendor="gcu", device_type="gcu")
    assert pf.PlatformFL.is_arch_support_pdl() is False

    fn = MagicMock()
    fn.current_device.return_value = 0
    fn.get_device_capability.return_value = (9, 0)
    monkeypatch.setattr(pf.PlatformFL, "torch_device_fn", fn)
    _platform(monkeypatch, vendor="nvidia", device_type="cuda")
    assert pf.PlatformFL.is_arch_support_pdl() is True

    fn.get_device_capability.return_value = (8, 0)
    assert pf.PlatformFL.is_arch_support_pdl() is False

    fn.current_device.side_effect = RuntimeError("no device")
    assert pf.PlatformFL.is_arch_support_pdl() is False


class _NVMLError(Exception):
    pass


def _pynvml_module(status):
    return dict(
        NVMLError=_NVMLError,
        NVML_P2P_CAPS_INDEX_NVLINK=1,
        NVML_P2P_STATUS_OK=0,
        nvmlInit=MagicMock(),
        nvmlDeviceGetHandleByIndex=MagicMock(side_effect=lambda index: index),
        nvmlDeviceGetP2PStatus=MagicMock(return_value=status),
    )


def test_is_fully_connected_ok(monkeypatch):
    _install_mod(monkeypatch, "pynvml", **_pynvml_module(0))
    assert pf.PlatformFL.is_fully_connected([0, 1]) is True


def test_is_fully_connected_not_ok(monkeypatch):
    _install_mod(monkeypatch, "pynvml", **_pynvml_module(1))
    assert pf.PlatformFL.is_fully_connected([0, 1, 2]) is False


def test_is_fully_connected_nvml_error(monkeypatch):
    mod = _pynvml_module(0)
    mod["nvmlDeviceGetP2PStatus"] = MagicMock(side_effect=_NVMLError("x"))
    _install_mod(monkeypatch, "pynvml", **mod)
    assert pf.PlatformFL.is_fully_connected([0, 1]) is False


def test_is_fully_connected_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert pf.PlatformFL.is_fully_connected([0, 1]) is False


def test_manual_seed_all_direct(monkeypatch):
    fn = SimpleNamespace(manual_seed_all=MagicMock())
    monkeypatch.setattr(pf.PlatformFL, "torch_device_fn", fn)
    pf.PlatformFL.manual_seed_all(123)
    fn.manual_seed_all.assert_called_once_with(123)


def test_manual_seed_all_fallback(monkeypatch):
    generators = []

    class _Fn:
        def device_count(self):
            return 2

        def _get_or_create_default_generator(self, device_id):
            generator = SimpleNamespace(manual_seed=MagicMock())
            generators.append((device_id, generator))
            return generator

    monkeypatch.setattr(pf.PlatformFL, "torch_device_fn", _Fn())
    manual_seed = MagicMock()
    monkeypatch.setattr(torch, "manual_seed", manual_seed)
    pf.PlatformFL.manual_seed_all(456)
    manual_seed.assert_called_once_with(456)
    assert [device_id for device_id, _ in generators] == [0, 1]
    for _, generator in generators:
        generator.manual_seed.assert_called_once_with(456)


# ---------------------------------------------------------------------------
# vllm_fl.utils
# ---------------------------------------------------------------------------


def test_env_vendor_name(monkeypatch):
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    assert u._env_vendor_name() == ""
    monkeypatch.setenv("GEMS_VENDOR", "  GCU ")
    assert u._env_vendor_name() == "gcu"


def test_get_vendor_device_field(monkeypatch):
    assert u.get_device_type("gcu") == "gcu"
    assert u.get_device_name("nvidia") == "nvidia"

    with pytest.raises(ValueError):
        u._get_vendor_device_field("", "device_type")
    with pytest.raises(ValueError):
        u._get_vendor_device_field(123, "device_type")
    with pytest.raises(ValueError):
        u._get_vendor_device_field("does-not-exist", "device_type")

    monkeypatch.setitem(
        u.VENDOR_DEVICE_MAP, "broken", {"device_type": "", "device_name": "x"}
    )
    with pytest.raises(ValueError):
        u._get_vendor_device_field("broken", "device_type")
    with pytest.raises(ValueError):
        u._get_vendor_device_field("broken", "missing")


def test_get_device_control_env_var(monkeypatch):
    monkeypatch.delenv("FLAGOS_DEVICE_CONTROL_ENV_VAR", raising=False)
    assert u.get_device_control_env_var("nvidia") == "CUDA_VISIBLE_DEVICES"
    assert u.get_device_control_env_var("enflame") == "TOPS_VISIBLE_DEVICES"
    assert (
        u.get_device_control_env_var("unknown")
        == u._DEVICE_CONTROL_ENV_VAR_PLACEHOLDER
    )
    monkeypatch.setenv("FLAGOS_DEVICE_CONTROL_ENV_VAR", "MY_VISIBLE_DEVICES")
    assert u.get_device_control_env_var("nvidia") == "MY_VISIBLE_DEVICES"


def test_use_flaggems(monkeypatch):
    monkeypatch.delenv("VLLM_FL_PREFER_ENABLED", raising=False)
    monkeypatch.delenv("VLLM_FL_PREFER", raising=False)
    monkeypatch.delenv("USE_FLAGGEMS", raising=False)
    assert u.use_flaggems() is True
    assert u.use_flaggems(default=False) is False

    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "0")
    assert u.use_flaggems() is False

    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "1")
    monkeypatch.setenv("VLLM_FL_PREFER", "vendor")
    assert u.use_flaggems() is False

    monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
    monkeypatch.setenv("USE_FLAGGEMS", "true")
    assert u.use_flaggems(default=False) is True
    monkeypatch.setenv("USE_FLAGGEMS", "false")
    assert u.use_flaggems(default=True) is False


def _clear_flagos_env(monkeypatch):
    for name in (
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    ):
        monkeypatch.delenv(name, raising=False)


def test_flagos_whitelist_blacklist_env(monkeypatch):
    _clear_flagos_env(monkeypatch)
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", " a, b ,, ")
    assert u.get_flag_gems_whitelist_blacklist() == (["a", "b"], None)

    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST")
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "c,d")
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "e,c,f")
    assert u.get_flag_gems_whitelist_blacklist() == (None, ["c", "d", "e", "f"])

    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "x")
    with pytest.raises(ValueError):
        u.get_flag_gems_whitelist_blacklist()


def test_flagos_whitelist_blacklist_config(monkeypatch):
    _clear_flagos_env(monkeypatch)
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=lambda: ["w1", "w2"],
        get_flagos_blacklist=lambda: [],
    )
    assert u.get_flag_gems_whitelist_blacklist() == (["w1", "w2"], None)

    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=lambda: [],
        get_flagos_blacklist=lambda: ["b1"],
    )
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "b2")
    assert u.get_flag_gems_whitelist_blacklist() == (None, ["b1", "b2"])

    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=lambda: ["w"],
        get_flagos_blacklist=lambda: ["b"],
    )
    with pytest.raises(ValueError):
        u.get_flag_gems_whitelist_blacklist()

    def _explode():
        raise RuntimeError("config unavailable")

    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_flagos_whitelist=_explode,
        get_flagos_blacklist=lambda: [],
    )
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", raising=False)
    assert u.get_flag_gems_whitelist_blacklist() == (None, None)


def test_use_flaggems_op(monkeypatch):
    monkeypatch.setattr(u, "use_flaggems", lambda default=True: False)
    assert u.use_flaggems_op("x") is False

    monkeypatch.setattr(u, "use_flaggems", lambda default=True: True)
    monkeypatch.setattr(
        u, "get_flag_gems_whitelist_blacklist", lambda: (["a"], None)
    )
    assert u.use_flaggems_op("a") is True
    assert u.use_flaggems_op("b") is False

    monkeypatch.setattr(
        u, "get_flag_gems_whitelist_blacklist", lambda: (None, ["b"])
    )
    assert u.use_flaggems_op("a") is True
    assert u.use_flaggems_op("b") is False

    monkeypatch.setattr(
        u, "get_flag_gems_whitelist_blacklist", lambda: (None, None)
    )
    assert u.use_flaggems_op("anything") is True


def test_load_op_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setattr(u, "_OP_CONFIG", None)
    monkeypatch.delenv("VLLM_FL_OP_CONFIG", raising=False)
    u._load_op_config_from_env()
    assert u.get_op_config() is None

    monkeypatch.setenv("VLLM_FL_OP_CONFIG", "   ")
    u._load_op_config_from_env()
    assert u.get_op_config() is None

    missing = tmp_path / "missing.json"
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(missing))
    with pytest.raises(ValueError):
        u._load_op_config_from_env()

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(invalid))
    with pytest.raises(ValueError):
        u._load_op_config_from_env()

    not_dict = tmp_path / "list.json"
    not_dict.write_text("[1, 2]", encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(not_dict))
    with pytest.raises(ValueError):
        u._load_op_config_from_env()

    bad_types = tmp_path / "bad.json"
    bad_types.write_text('{"a": 1}', encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(bad_types))
    with pytest.raises(ValueError):
        u._load_op_config_from_env()

    good = tmp_path / "good.json"
    good.write_text('{"op": "backend"}', encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_OP_CONFIG", str(good))
    u._load_op_config_from_env()
    assert u.get_op_config() == {"op": "backend"}


def test_device_info(monkeypatch):
    backend = SimpleNamespace(
        set_torch_backend_device_fn=MagicMock(),
        gen_torch_device_object=lambda: "torch-device-fn",
        get_torch_backend_device_fn=lambda: "torch-backend-device",
    )
    device = SimpleNamespace(
        vendor_name="gcu", dispatch_key="gcu", name="gcu"
    )
    _install_mod(monkeypatch, "flag_gems.runtime", backend=backend, device=device)

    info = u.DeviceInfo()
    assert info.vendor_name == "gcu"
    assert info.device_type == "gcu"
    assert info.dispatch_key == "gcu"
    assert info.torch_device_fn == "torch-device-fn"
    assert info.torch_backend_device == "torch-backend-device"
    backend.set_torch_backend_device_fn.assert_called_once_with("gcu")
    assert info.get_supported_device() is True

    device.vendor_name = "unknown-vendor"
    with pytest.raises(NotImplementedError):
        info.get_supported_device()


def test_get_flaggems_all_ops(monkeypatch):
    _install_mod(
        monkeypatch,
        "flag_gems",
        _FULL_CONFIG=[("a", object()), ("b", object(), object())],
    )
    assert u.get_flaggems_all_ops() == ["a", "b"]

    _install_mod(monkeypatch, "flag_gems", _FULL_CONFIG=None)
    assert u.get_flaggems_all_ops() == []


def test_oot_whitelist_blacklist(monkeypatch):
    for name in (
        "VLLM_FL_OOT_WHITELIST",
        "VLLM_FL_OOT_BLACKLIST",
    ):
        monkeypatch.delenv(name, raising=False)
    assert u.get_oot_whitelist() is None
    assert u.get_oot_blacklist() is None

    monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", " a, b ")
    assert u.get_oot_whitelist() == ["a", "b"]
    assert u.get_oot_blacklist() is None

    monkeypatch.delenv("VLLM_FL_OOT_WHITELIST")
    monkeypatch.setenv("VLLM_FL_OOT_BLACKLIST", "c,d")
    assert u.get_oot_blacklist() == ["c", "d"]

    monkeypatch.delenv("VLLM_FL_OOT_BLACKLIST")
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.config",
        get_oot_blacklist=lambda: ["config"],
    )
    assert u.get_oot_blacklist() == ["config"]


def test_is_oot_enabled(monkeypatch):
    monkeypatch.delenv("VLLM_FL_PREFER_ENABLED", raising=False)
    monkeypatch.delenv("VLLM_FL_OOT_ENABLED", raising=False)
    assert u.is_oot_enabled() is True

    monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "false")
    assert u.is_oot_enabled() is False

    monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "1")
    assert u.is_oot_enabled() is True

    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "0")
    assert u.is_oot_enabled() is False


# ---------------------------------------------------------------------------
# vllm_fl.envs
# ---------------------------------------------------------------------------


def test_envs_lazy_access(monkeypatch):
    import vllm_fl.envs as envs

    monkeypatch.delenv("VLLM_FL_PREFER_ENABLED", raising=False)
    assert envs.VLLM_FL_PREFER_ENABLED is True
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "0")
    assert envs.VLLM_FL_PREFER_ENABLED is False

    monkeypatch.delenv("FLAGGEMS_ENABLE_OPLIST_PATH", raising=False)
    assert envs.FLAGGEMS_ENABLE_OPLIST_PATH == "/tmp/flaggems_enable_oplist.txt"
    monkeypatch.setenv("FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/custom.txt")
    assert envs.FLAGGEMS_ENABLE_OPLIST_PATH == "/tmp/custom.txt"

    monkeypatch.delenv("VLLM_FL_PREFER_ENABLED", raising=False)
    monkeypatch.delenv("VLLM_FL_PREFER", raising=False)
    monkeypatch.setenv("USE_FLAGGEMS", "true")
    assert envs.USE_FLAGGEMS is True

    assert set(envs.__dir__()) == set(envs.fl_vllm_environment_variables)
    assert envs.is_set("FLAGGEMS_ENABLE_OPLIST_PATH") is True
    assert envs.is_set("VLLM_FL_PREFER_ENABLED") is False
    with pytest.raises(AttributeError):
        envs.is_set("NOT_A_FL_VAR")
    with pytest.raises(AttributeError):
        envs.NOT_A_FL_VAR


# ---------------------------------------------------------------------------
# vllm_fl.version
# ---------------------------------------------------------------------------


def test_version_module_pkg_version(monkeypatch):
    import vllm_fl.version as ver

    monkeypatch.setattr(ver.metadata, "version", lambda name: "1.2.3")
    assert ver._pkg_version() == "1.2.3"

    def _no_metadata(name):
        raise RuntimeError("not installed")

    monkeypatch.setattr(ver.metadata, "version", _no_metadata)
    _install_mod(monkeypatch, "vllm_fl._version", __version__="9.9.9")
    assert ver._pkg_version() == "9.9.9"

    monkeypatch.delattr(vllm_fl, "_version", raising=False)
    monkeypatch.setitem(sys.modules, "vllm_fl._version", None)
    assert ver._pkg_version() == "0.0.0+unknown"
    assert isinstance(ver.__version__, str)
    assert isinstance(ver.git_version, str)
    assert set(ver.git_info) == {"id", "date"}


def test_version_git_helpers(monkeypatch):
    import vllm_fl.version as ver

    monkeypatch.setattr(
        ver,
        "subprocess",
        SimpleNamespace(
            check_output=lambda *a, **k: "abc123\n",
            DEVNULL=None,
        ),
    )
    assert ver._git_head_from_repo() == "abc123"
    monkeypatch.setattr(
        ver,
        "subprocess",
        SimpleNamespace(
            check_output=lambda *a, **k: "2024-01-02T03:04:05+00:00\n",
            DEVNULL=None,
        ),
    )
    assert ver._git_commit_date_from_repo() == "2024-01-02"

    def _empty(*a, **k):
        return "\n"

    monkeypatch.setattr(
        ver, "subprocess", SimpleNamespace(check_output=_empty, DEVNULL=None)
    )
    assert ver._git_head_from_repo() is None
    assert ver._git_commit_date_from_repo() is None

    def _raise(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(
        ver, "subprocess", SimpleNamespace(check_output=_raise, DEVNULL=None)
    )
    assert ver._git_head_from_repo() is None
    assert ver._git_commit_date_from_repo() is None


def test_version_load_scm(monkeypatch):
    import vllm_fl.version as ver

    _install_mod(
        monkeypatch,
        "vllm_fl._version",
        git_version="deadbeef",
        git_date="2024-05-06",
    )
    assert ver._load_scm() == ("deadbeef", "2024-05-06")

    _install_mod(
        monkeypatch,
        "vllm_fl._version",
        git_version="Unknown",
        git_date="",
    )
    assert ver._load_scm() == (None, None)

    monkeypatch.delattr(vllm_fl, "_version", raising=False)
    monkeypatch.setitem(sys.modules, "vllm_fl._version", None)
    assert ver._load_scm() == (None, None)


# ---------------------------------------------------------------------------
# vllm_fl.nvidia_platform
# ---------------------------------------------------------------------------


@pytest.fixture
def npf():
    mod = pytest.importorskip("vllm_fl.nvidia_platform")
    original = mod.NvidiaPlatformFL.dist_backend
    yield mod
    mod.NvidiaPlatformFL.dist_backend = original


def test_nvidia_refresh_dist_backend(monkeypatch, npf):
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    assert npf.NvidiaPlatformFL._refresh_dist_backend() is False
    assert npf.NvidiaPlatformFL.dist_backend == npf.CudaPlatform.dist_backend

    monkeypatch.setenv("FLAGCX_PATH", "/tmp/flagcx")
    assert npf.NvidiaPlatformFL._refresh_dist_backend() is True
    assert npf.NvidiaPlatformFL.dist_backend == "flagcx"


def test_nvidia_check_and_update_flagcx(monkeypatch, npf):
    monkeypatch.setenv("FLAGCX_PATH", "/tmp/flagcx")
    monkeypatch.setattr(
        npf.CudaPlatform,
        "check_and_update_config",
        classmethod(lambda cls, cfg: None),
    )
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(disable_custom_all_reduce=False, worker_cls=None),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(fuse_allreduce_rms=True)
        ),
    )
    npf.NvidiaPlatformFL.check_and_update_config(cfg)
    assert cfg.parallel_config.disable_custom_all_reduce is True
    assert cfg.compilation_config.pass_config.fuse_allreduce_rms is False
    assert cfg.parallel_config.worker_cls == (
        "vllm_fl.worker.nvidia_worker.NvidiaWorkerFL"
    )


def test_nvidia_check_and_update_no_flagcx(monkeypatch, npf):
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setattr(
        npf.CudaPlatform,
        "check_and_update_config",
        classmethod(lambda cls, cfg: None),
    )
    pass_config = SimpleNamespace(fuse_allreduce_rms=True)
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(disable_custom_all_reduce=False, worker_cls=None),
        compilation_config=SimpleNamespace(pass_config=pass_config),
    )
    npf.NvidiaPlatformFL.check_and_update_config(cfg)
    assert cfg.parallel_config.disable_custom_all_reduce is False
    assert pass_config.fuse_allreduce_rms is True


def test_nvidia_communicator_and_allreduce(monkeypatch, npf):
    monkeypatch.setenv("FLAGCX_PATH", "/tmp/flagcx")
    assert npf.NvidiaPlatformFL.get_device_communicator_cls() == (
        "vllm_fl.distributed.nvidia_communicator.NvidiaCommunicatorFL"
    )
    assert npf.NvidiaPlatformFL.use_custom_allreduce() is False

    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setattr(
        npf.CudaPlatform,
        "get_device_communicator_cls",
        classmethod(lambda cls: "cuda-comm"),
    )
    monkeypatch.setattr(
        npf.CudaPlatform, "use_custom_allreduce", classmethod(lambda cls: True)
    )
    assert npf.NvidiaPlatformFL.get_device_communicator_cls() == "cuda-comm"
    assert npf.NvidiaPlatformFL.use_custom_allreduce() is True


def test_nvidia_get_attn_backend_cls(monkeypatch, npf):
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "yes")

    class _Backend:
        def attention_backend(self, use_mla, use_sparse):
            return f"fg-{use_mla}-{use_sparse}"

    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.flaggems.flaggems",
        FlagGemsBackend=_Backend,
    )
    selector = SimpleNamespace(use_mla=True, use_sparse=False)
    assert (
        npf.NvidiaPlatformFL.get_attn_backend_cls(None, selector, 8)
        == "fg-True-False"
    )

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "0")
    monkeypatch.setattr(
        npf.CudaPlatform,
        "get_attn_backend_cls",
        classmethod(lambda cls, backend, selector, num_heads=None: "cuda-attn"),
    )
    assert npf.NvidiaPlatformFL.get_attn_backend_cls(None, selector, 8) == "cuda-attn"


# ---------------------------------------------------------------------------
# vllm_fl.__init__ helpers
# ---------------------------------------------------------------------------


def test_triton_import_compat_skips_without_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "triton.language", None)
    assert vllm_fl._patch_flag_gems_triton_import_compat() is None


def test_triton_import_compat_adds_stub(monkeypatch):
    tl = ModuleType("triton.language")
    triton = ModuleType("triton")
    triton.__path__ = []
    triton.language = tl
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)
    monkeypatch.setitem(sys.modules, "triton.knobs", ModuleType("triton.knobs"))
    assert vllm_fl._patch_flag_gems_triton_import_compat() is None

    fake_tl = ModuleType("triton.language")
    fake_triton = ModuleType("triton")
    fake_triton.__path__ = []
    fake_triton.language = fake_tl
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_tl)
    monkeypatch.delitem(sys.modules, "triton.knobs", raising=False)

    real_import = importlib.import_module

    def _missing(name, *args, **kwargs):
        if name == "triton.knobs":
            raise ModuleNotFoundError("no knobs", name="triton.knobs")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _missing)
    try:
        vllm_fl._patch_flag_gems_triton_import_compat()
        assert hasattr(fake_tl, "map_elementwise")
        fake_tl.map_elementwise()
    except NotImplementedError:
        pass
    finally:
        sys.modules.pop("triton.knobs", None)

    def _other(name, *args, **kwargs):
        if name == "triton.knobs":
            raise ModuleNotFoundError("other", name="triton.other")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _other)
    with pytest.raises(ModuleNotFoundError):
        vllm_fl._patch_flag_gems_triton_import_compat()


def test_arm_cpu_platform(monkeypatch):
    import platform as std_platform

    monkeypatch.setattr(std_platform, "machine", lambda: "x86_64")
    assert vllm_fl._arm_cpu_platform() is None

    monkeypatch.setattr(std_platform, "machine", lambda: "aarch64")
    vp = pytest.importorskip("vllm.platforms")
    monkeypatch.setattr(
        vp, "cpu_platform_plugin", lambda: "CPU-platform", raising=False
    )
    assert vllm_fl._arm_cpu_platform() == "CPU-platform"


def test_dunder_getattr_distributed(monkeypatch):
    saved = vllm_fl.__dict__.get("distributed")
    vllm_fl.__dict__.pop("distributed", None)
    fake = ModuleType("vllm_fl.distributed")
    monkeypatch.setitem(sys.modules, "vllm_fl.distributed", fake)
    try:
        assert vllm_fl.distributed is fake
    finally:
        vllm_fl.__dict__.pop("distributed", None)
        if saved is not None:
            vllm_fl.distributed = saved

    with pytest.raises(AttributeError):
        vllm_fl.not_a_real_attribute


def test_register_flagcx_connector(monkeypatch):
    registered = {}

    class _Factory:
        _registry = {}

        @classmethod
        def register_connector(cls, alias, module, attr):
            registered[alias] = (module, attr)
            cls._registry[alias] = True

    _install_mod(
        monkeypatch,
        "vllm.distributed.kv_transfer.kv_connector.factory",
        KVConnectorFactory=_Factory,
    )
    vllm_fl._register_flagcx_connector()
    assert set(registered) == {"FlagCXConnector", "FlagcxConnector"}
    vllm_fl._register_flagcx_connector()
    assert len(registered) == 2


def test_patch_flash_attn_import_already_loaded(monkeypatch):
    existing = ModuleType("vllm.vllm_flash_attn")
    existing.marker = object()
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", existing)
    vllm_fl._patch_flash_attn_import()
    assert sys.modules["vllm.vllm_flash_attn"] is existing


def test_patch_flash_attn_import_installs_stub(monkeypatch):
    monkeypatch.delitem(sys.modules, "vllm.vllm_flash_attn", raising=False)
    monkeypatch.setitem(
        sys.modules, "vllm.vllm_flash_attn.flash_attn_interface", object()
    )
    try:
        with _block_import(
            "vllm.vllm_flash_attn", ImportError("no flash attn")
        ):
            vllm_fl._patch_flash_attn_import()
        stub = sys.modules["vllm.vllm_flash_attn"]
        assert stub.FA2_AVAILABLE is False
        assert stub.FA3_AVAILABLE is False
        assert stub.flash_attn_varlen_func is None
        assert stub.is_fa_version_supported() is False
        assert "flash_attn C extensions" in stub.fa_version_unsupported_reason()
        assert "vllm.vllm_flash_attn.flash_attn_interface" not in sys.modules
    finally:
        sys.modules.pop("vllm.vllm_flash_attn", None)


def test_is_metax_backend(monkeypatch):
    _install_mod(
        monkeypatch,
        "vllm_fl.ops._C_ops_registry",
        _is_metax_backend=lambda: True,
    )
    assert vllm_fl._is_metax_backend() is True


def test_patch_custom_ops_metax(monkeypatch):
    monkeypatch.setattr(vllm_fl, "_is_metax_backend", lambda: True)
    vllm_fl._patch_custom_ops()


def test_patch_custom_ops_extension_present(monkeypatch):
    monkeypatch.setattr(vllm_fl, "_is_metax_backend", lambda: False)
    _install_mod(monkeypatch, "vllm._C")
    vllm_fl._patch_custom_ops()


def test_patch_custom_ops_registers_schemas(monkeypatch):
    monkeypatch.setattr(vllm_fl, "_is_metax_backend", lambda: False)
    monkeypatch.setitem(sys.modules, "vllm._C", None)
    monkeypatch.setitem(sys.modules, "vllm._C_stable_libtorch", None)
    monkeypatch.setitem(sys.modules, "vllm_fl._C", None)
    register = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.ops._C_ops_registry",
        register_op_schemas=register,
    )
    vllm_fl._patch_custom_ops()
    register.assert_called_once()


def test_init_vendor_device_mthreads(monkeypatch):
    monkeypatch.setattr(vllm_fl, "_install_metax_compat", lambda: None)
    monkeypatch.setattr(
        u, "DeviceInfo", lambda: SimpleNamespace(vendor_name="mthreads")
    )
    patch_accel = MagicMock(return_value=True)
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.musa.patch",
        patch_accelerator_missing_attrs=patch_accel,
    )
    vllm_fl._init_vendor_device()
    patch_accel.assert_called_once()

    patch_accel.return_value = False
    with pytest.raises(RuntimeError):
        vllm_fl._init_vendor_device()


def test_init_vendor_device_kunlunxin_and_amd(monkeypatch):
    monkeypatch.setattr(vllm_fl, "_install_metax_compat", lambda: None)
    monkeypatch.setattr(
        u, "DeviceInfo", lambda: SimpleNamespace(vendor_name="kunlunxin")
    )
    patch_xpu = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils",
        _patch_xpu_get_device=patch_xpu,
    )
    vllm_fl._init_vendor_device()
    patch_xpu.assert_called_once()

    monkeypatch.setattr(u, "DeviceInfo", lambda: SimpleNamespace(vendor_name="amd"))
    apply_target = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.patches.flagtree_target_info",
        apply_flagtree_target_info=apply_target,
    )
    vllm_fl._init_vendor_device()
    apply_target.assert_called_once()


def test_init_vendor_device_other(monkeypatch):
    monkeypatch.setattr(vllm_fl, "_install_metax_compat", lambda: None)
    monkeypatch.setattr(u, "DeviceInfo", lambda: SimpleNamespace(vendor_name="gcu"))
    assert vllm_fl._init_vendor_device() is None


def test_install_metax_compat(monkeypatch):
    install = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.shims",
        install=install,
        metax_active=lambda: True,
    )
    vllm_fl._install_metax_compat()
    install.assert_called_once()

    install.reset_mock()
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.shims",
        install=install,
        metax_active=lambda: False,
    )
    vllm_fl._install_metax_compat()
    install.assert_not_called()

    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.metax.shims", None
    )
    vllm_fl._install_metax_compat()


def test_install_kunlunxin_fla_alias(monkeypatch):
    fix_vars = MagicMock()
    install = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils",
        _fix_platform_vars=fix_vars,
    )
    _install_mod(
        monkeypatch, "vllm_fl.patches.fla_module_relocation", install=install
    )
    vllm_fl._install_kunlunxin_fla_alias()
    fix_vars.assert_called_once()
    install.assert_called_once()

    assert vllm_fl._kunlunxin_fla_alias_lock.acquire(blocking=False)
    try:
        vllm_fl._install_kunlunxin_fla_alias()
    finally:
        vllm_fl._kunlunxin_fla_alias_lock.release()

    def _boom():
        raise RuntimeError("alias failed")

    _install_mod(
        monkeypatch, "vllm_fl.patches.fla_module_relocation", install=_boom
    )
    vllm_fl._install_kunlunxin_fla_alias()
    assert vllm_fl._kunlunxin_fla_alias_lock.acquire(blocking=False)
    vllm_fl._kunlunxin_fla_alias_lock.release()


# ---------------------------------------------------------------------------
# vllm_fl.__init__ register hooks
# ---------------------------------------------------------------------------


def _prep_register(monkeypatch, vendor, arm_platform=None):
    monkeypatch.setattr(vllm_fl, "_init_vendor_device", MagicMock())
    monkeypatch.setattr(vllm_fl, "_patch_custom_ops", MagicMock())
    monkeypatch.setattr(vllm_fl, "_patch_flash_attn_import", MagicMock())
    monkeypatch.setattr(vllm_fl, "_get_op_config", MagicMock())
    monkeypatch.setattr(vllm_fl, "_install_kunlunxin_fla_alias", MagicMock())
    monkeypatch.setattr(vllm_fl, "_arm_cpu_platform", lambda: arm_platform)
    monkeypatch.setattr(u, "DeviceInfo", lambda: SimpleNamespace(vendor_name=vendor))
    for name in (
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_MOE_SKIP_PADDING",
    ):
        monkeypatch.delenv(name, raising=False)


def test_register_arm_cpu(monkeypatch):
    _prep_register(monkeypatch, "gcu", arm_platform="CPU-platform")
    assert vllm_fl.register() == "CPU-platform"
    vllm_fl._patch_custom_ops.assert_not_called()


def test_register_nvidia(monkeypatch):
    _prep_register(monkeypatch, "nvidia")
    assert vllm_fl.register() == "vllm_fl.nvidia_platform.NvidiaPlatformFL"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    vllm_fl._patch_custom_ops.assert_not_called()
    vllm_fl._patch_flash_attn_import.assert_not_called()
    vllm_fl._get_op_config.assert_called_once()


def test_register_kunlunxin(monkeypatch):
    _prep_register(monkeypatch, "kunlunxin")
    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_MOE_SKIP_PADDING"] == "0"
    vllm_fl._patch_custom_ops.assert_called_once()
    vllm_fl._patch_flash_attn_import.assert_called_once()
    vllm_fl._install_kunlunxin_fla_alias.assert_called_once()


def test_register_mthreads_and_other(monkeypatch):
    _prep_register(monkeypatch, "mthreads")
    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"

    _prep_register(monkeypatch, "gcu")
    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ


def test_register_keeps_existing_multiproc_method(monkeypatch):
    _prep_register(monkeypatch, "gcu")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    vllm_fl.register()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


def test_register_quant_linear(monkeypatch):
    vp = pytest.importorskip("vllm.platforms")
    add_oot = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.quantization.quant_linear",
        add_oot_quant_kernel=add_oot,
    )
    monkeypatch.setattr(vp, "current_platform", SimpleNamespace(device_type="gcu"))
    vllm_fl.register_quant_linear()
    add_oot.assert_not_called()

    monkeypatch.setattr(vp, "current_platform", SimpleNamespace(device_type="cuda"))
    vllm_fl.register_quant_linear()
    add_oot.assert_called_once()


def test_register_router(monkeypatch):
    vp = pytest.importorskip("vllm.platforms")
    replace = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.ops.fused_moe.router",
        replace_router_with_fl=replace,
    )
    monkeypatch.setattr(u, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(vp, "current_platform", SimpleNamespace(device_type="cuda"))
    vllm_fl.register_router()
    replace.assert_called_once()

    monkeypatch.setattr(u, "is_oot_enabled", lambda: False)
    replace.reset_mock()
    vllm_fl.register_router()
    replace.assert_not_called()

    monkeypatch.setattr(vp, "current_platform", SimpleNamespace(device_type="musa"))
    vllm_fl.register_router()
    replace.assert_not_called()


def test_register_gdn_packed_decode_patch(monkeypatch):
    _install_mod(
        monkeypatch,
        "vllm_fl.patches.gdn_packed_decode",
        patch_vllm_packed_gdn_beta=lambda: True,
    )
    assert vllm_fl._register_gdn_packed_decode_patch() is True

    _install_mod(monkeypatch, "vllm_fl.patches.gdn_packed_decode")
    assert vllm_fl._register_gdn_packed_decode_patch() is False

    monkeypatch.setitem(sys.modules, "vllm_fl.patches.gdn_packed_decode", None)
    assert vllm_fl._register_gdn_packed_decode_patch() is False


def _prep_register_model(monkeypatch, device_type="cuda", vendor="gcu"):
    vp = pytest.importorskip("vllm.platforms")
    monkeypatch.setattr(
        vp, "current_platform", SimpleNamespace(device_type=device_type)
    )
    monkeypatch.setattr(vllm_fl, "_arm_cpu_platform", lambda: None)
    monkeypatch.setattr(u, "DeviceInfo", lambda: SimpleNamespace(vendor_name=vendor))
    monkeypatch.setattr(vllm_fl, "_register_flagcx_connector", MagicMock())
    monkeypatch.setattr(vllm_fl, "register_quant_linear", MagicMock())
    monkeypatch.setattr(vllm_fl, "register_router", MagicMock())
    monkeypatch.setattr(vllm_fl, "_register_gdn_packed_decode_patch", MagicMock())
    return vp


def test_register_model_arm_cpu_flag_gems_arm(monkeypatch):
    _prep_register_model(monkeypatch, device_type="cpu", vendor="gcu")
    monkeypatch.setattr(vllm_fl, "_arm_cpu_platform", lambda: "CPU-platform")
    apply_patch = MagicMock()
    install_w4a8 = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.patches.arm_cpu_gdn",
        apply_arm_cpu_gdn_state_indices_patch=apply_patch,
    )
    _install_mod(monkeypatch, "flag_gems", vendor_name="arm")
    _install_mod(
        monkeypatch,
        "vllm_fl.quantization.arm_cpu_w4a8",
        install_arm_cpu_packed_w4a8=install_w4a8,
    )
    vllm_fl.register_model()
    apply_patch.assert_called_once()
    install_w4a8.assert_called_once()


def test_register_model_arm_cpu_wrong_vendor(monkeypatch):
    _prep_register_model(monkeypatch, device_type="cpu", vendor="gcu")
    monkeypatch.setattr(vllm_fl, "_arm_cpu_platform", lambda: "CPU-platform")
    _install_mod(
        monkeypatch,
        "vllm_fl.patches.arm_cpu_gdn",
        apply_arm_cpu_gdn_state_indices_patch=MagicMock(),
    )
    _install_mod(monkeypatch, "flag_gems", vendor_name="not-arm")
    install_w4a8 = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.quantization.arm_cpu_w4a8",
        install_arm_cpu_packed_w4a8=install_w4a8,
    )
    vllm_fl.register_model()
    install_w4a8.assert_not_called()


def test_register_model_arm_cpu_no_flag_gems(monkeypatch):
    _prep_register_model(monkeypatch, device_type="cpu", vendor="gcu")
    monkeypatch.setattr(vllm_fl, "_arm_cpu_platform", lambda: "CPU-platform")
    _install_mod(
        monkeypatch,
        "vllm_fl.patches.arm_cpu_gdn",
        apply_arm_cpu_gdn_state_indices_patch=MagicMock(),
    )
    with _block_import(
        "flag_gems", ModuleNotFoundError("no flag_gems", name="flag_gems")
    ):
        vllm_fl.register_model()
    vllm_fl.register_quant_linear.assert_not_called()


def test_register_model_gcu(monkeypatch):
    _prep_register_model(monkeypatch, device_type="cuda", vendor="gcu")
    vllm_fl.register_model()
    vllm_fl._register_flagcx_connector.assert_called_once()
    vllm_fl.register_quant_linear.assert_called_once()
    vllm_fl.register_router.assert_called_once()
    vllm_fl._register_gdn_packed_decode_patch.assert_called_once()


def test_register_model_nvidia(monkeypatch):
    _prep_register_model(monkeypatch, device_type="cuda", vendor="nvidia")
    audio = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.patches.minicpmo_audio",
        patch_minicpmo_audio_budget=audio,
    )
    vllm_fl.register_model()
    audio.assert_called_once()
    vllm_fl._register_gdn_packed_decode_patch.assert_not_called()


def test_register_model_iluvatar(monkeypatch):
    _prep_register_model(monkeypatch, device_type="cuda", vendor="iluvatar")
    cleanup = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.iluvatar.shutdown",
        install_worker_cleanup=cleanup,
    )
    vllm_fl.register_model()
    cleanup.assert_called_once()


def test_register_model_ascend(monkeypatch):
    _prep_register_model(monkeypatch, device_type="npu", vendor="ascend")
    registry = MagicMock()
    _install_mod(
        monkeypatch,
        "vllm.model_executor.models",
        ModelRegistry=registry,
    )
    _install_mod(monkeypatch, "vllm_fl.quantization.modelslim_w8a8")
    vllm_fl.register_model()
    assert registry.register_model.call_count == 2
    registered = [c.args[0] for c in registry.register_model.call_args_list]
    assert "DeepseekV4ForCausalLM" in registered
    assert "GlmMoeDsaForCausalLM" in registered











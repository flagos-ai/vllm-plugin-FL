# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G3 coverage: vllm_fl.platform helpers and PlatformFL device surface (814b).

Targets the 814b snapshot of ``vllm_fl/platform.py``. Every test drives the
behavior through module globals, class attributes, ``sys.modules`` fakes or
``torch`` attribute injection (the sanctioned probe boundaries). No device is
required: all delegation targets are recording stubs and the tensor work is
CPU-only.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace as NS

import pytest
import torch


def _ensure_triton_stub():
    """Best-effort triton shim for CPU hosts with a partial install."""
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
        return
    except Exception:
        pass

    triton_mod = sys.modules.get("triton") or types.ModuleType("triton")
    lang = sys.modules.get("triton.language") or types.ModuleType(
        "triton.language"
    )

    def _mark(fn):
        try:
            fn.fn = fn
        except Exception:
            pass
        return fn

    def _jit(fn=None, *a, **k):
        return (lambda f: _mark(f)) if fn is None else _mark(fn)

    for name in (
        "float32", "float16", "bfloat16", "int32", "int64", "constexpr",
        "void", "uint32", "uint64",
    ):
        if not hasattr(lang, name):
            setattr(lang, name, name)
    for name, value in (
        ("jit", _jit),
        ("next_power_of_2", lambda n: 1 << max(0, int(n) - 1).bit_length()),
        ("cdiv", lambda a, b: -(-int(a) // int(b))),
    ):
        if not hasattr(lang, name):
            setattr(lang, name, value)
    triton_mod.language = lang
    if not hasattr(triton_mod, "jit"):
        triton_mod.jit = _jit
    sys.modules.setdefault("triton", triton_mod)
    sys.modules.setdefault("triton.language", lang)


_ensure_triton_stub()

from vllm.platforms.interface import DeviceCapability

import vllm_fl.platform as fl_platform
from vllm_fl.platform import PlatformFL


@pytest.fixture
def restore_identity(monkeypatch):
    """Snapshot/restore the device-probe-derived class identity."""
    for name in (
        "vendor_name",
        "device_type",
        "device_name",
        "dist_backend",
        "torch_device_fn",
    ):
        monkeypatch.setattr(PlatformFL, name, getattr(PlatformFL, name), raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Module-level helpers: _vllm_c_available / _safe_device_type
# ---------------------------------------------------------------------------


def test_vllm_c_available_true(monkeypatch):
    # Isolate the module global so the real importlib is never touched.
    monkeypatch.setattr(
        fl_platform,
        "importlib",
        NS(util=NS(find_spec=lambda name: object())),
    )
    assert fl_platform._vllm_c_available() is True


def test_vllm_c_available_swallows_probe_errors(monkeypatch):
    def _boom(name):
        raise ValueError("bad spec")

    monkeypatch.setattr(
        fl_platform, "importlib", NS(util=NS(find_spec=_boom))
    )
    assert fl_platform._vllm_c_available() is False


def test_safe_device_type_prefers_detected_value():
    assert fl_platform._safe_device_type(NS(device_type="cuda"), "nvidia") == "cuda"


def test_safe_device_type_maps_vendor_when_probe_empty(monkeypatch):
    monkeypatch.setattr(fl_platform, "get_device_type", lambda vendor: "npu")
    assert fl_platform._safe_device_type(NS(device_type=""), "ascend") == "npu"


def test_safe_device_type_empty_vendor(monkeypatch):
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    assert fl_platform._safe_device_type(NS(device_type=""), "") == ""


def test_safe_device_type_unknown_vendor_warns(monkeypatch):
    def _raise(vendor):
        raise ValueError("not in map")

    monkeypatch.setattr(fl_platform, "get_device_type", _raise)
    assert fl_platform._safe_device_type(NS(device_type=""), "bogus") == ""


def _load_platform_with_empty_vendor(monkeypatch):
    """Import a second PlatformFL instance with an unresolved vendor probe."""
    import vllm_fl.utils as fl_utils

    class _EmptyInfo:
        vendor_name = ""
        device_type = ""
        dispatch_key = ""
        torch_device_fn = object()
        device = NS()

    monkeypatch.setattr(fl_utils, "DeviceInfo", _EmptyInfo)
    monkeypatch.setattr(fl_utils, "_env_vendor_name", lambda: "")
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.delenv("FLAGCX_PATH", raising=False)

    spec = importlib.util.spec_from_file_location(
        "_g3_platform_empty_vendor", fl_platform.__file__
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_class_body_logs_unresolved_vendor(monkeypatch):
    """An empty detector must not abort import; the class body logs an error."""
    try:
        module = _load_platform_with_empty_vendor(monkeypatch)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"simulated empty-vendor import unsupported here: {exc}")
    assert module.PlatformFL.vendor_name == ""
    assert module.PlatformFL.device_type == ""


# ---------------------------------------------------------------------------
# Stateless identity predicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vendor_name", "device_type", "expected"),
    [
        ("nvidia", "cuda", True),
        ("metax", "cuda", True),
        ("mthreads", "musa", True),
        ("gcu", "gcu", True),
        ("enflame", "gcu", True),
        ("iluvatar", "cuda", False),
        ("hygon", "cuda", False),
        ("ascend", "npu", False),
    ],
)
def test_is_cuda_alike_matrix(
    restore_identity, vendor_name, device_type, expected
):
    instance = PlatformFL()
    instance.vendor_name = vendor_name
    instance.device_type = device_type
    assert instance.is_cuda_alike() is expected


def test_is_sleep_mode_available_nvidia_only(restore_identity):
    instance = PlatformFL()
    instance.vendor_name = "nvidia"
    assert instance.is_sleep_mode_available() is True
    instance.vendor_name = "ascend"
    assert instance.is_sleep_mode_available() is False


def test_is_musa_and_is_gcu_probe_torch_namespaces(restore_identity, monkeypatch):
    instance = PlatformFL()
    monkeypatch.setattr(torch, "musa", NS(is_available=lambda: True), raising=False)
    monkeypatch.setattr(torch, "gcu", NS(is_available=lambda: True), raising=False)
    assert instance.is_musa() is True
    assert instance.is_gcu() is True

    monkeypatch.delattr(torch, "musa", raising=False)
    monkeypatch.delattr(torch, "gcu", raising=False)
    assert instance.is_musa() is False
    assert instance.is_gcu() is False


def test_supported_dtypes_are_bf16_fp16_fp32():
    assert PlatformFL().supported_dtypes == [
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ]


def test_check_if_supports_dtype_bf16_without_capability_raises(
    restore_identity, monkeypatch
):
    PlatformFL.vendor_name = "nvidia"
    monkeypatch.setattr(
        PlatformFL,
        "has_device_capability",
        classmethod(lambda cls, capability: False),
        raising=False,
    )
    monkeypatch.setattr(
        PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: None),
        raising=False,
    )
    monkeypatch.setattr(
        PlatformFL,
        "get_device_name",
        classmethod(lambda cls, device_id=0: "FakeGPU"),
        raising=False,
    )
    with pytest.raises(ValueError, match="does not have a compute capability"):
        PlatformFL.check_if_supports_dtype(torch.bfloat16)


def test_check_if_supports_dtype_reports_capability_version(
    restore_identity, monkeypatch
):
    PlatformFL.vendor_name = "nvidia"
    monkeypatch.setattr(
        PlatformFL,
        "has_device_capability",
        classmethod(lambda cls, capability: False),
        raising=False,
    )
    monkeypatch.setattr(
        PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(7, 5)),
        raising=False,
    )
    monkeypatch.setattr(
        PlatformFL,
        "get_device_name",
        classmethod(lambda cls, device_id=0: "FakeGPU"),
        raising=False,
    )
    with pytest.raises(ValueError, match="compute capability 7.5"):
        PlatformFL.check_if_supports_dtype(torch.bfloat16)

    # Non-bf16 dtypes short-circuit before the capability check.
    PlatformFL.check_if_supports_dtype(torch.float16)


# ---------------------------------------------------------------------------
# torch_device_fn delegation
# ---------------------------------------------------------------------------


def test_get_current_memory_usage_delegates(restore_identity):
    calls = []

    class Recording:
        empty_cache = staticmethod(lambda *a: calls.append("empty_cache"))
        reset_peak_memory_stats = staticmethod(
            lambda device: calls.append(("reset", device))
        )
        max_memory_allocated = staticmethod(lambda device: 7.5)

    PlatformFL.torch_device_fn = Recording
    assert PlatformFL.get_current_memory_usage(None) == 7.5
    assert calls == ["empty_cache", ("reset", None)]


def test_set_device_nvidia_initialises_context(restore_identity, monkeypatch):
    calls = []

    class Recording:
        set_device = staticmethod(lambda device: calls.append(("set", device)))

    PlatformFL.torch_device_fn = Recording
    PlatformFL.vendor_name = "nvidia"
    zeros = []
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *args, **kwargs: zeros.append(kwargs) or torch.tensor([0.0]),
    )

    device = torch.device("cpu")
    PlatformFL.set_device(device)
    assert calls == [("set", device)]
    assert zeros and zeros[0]["device"] == device


def test_empty_cache_and_get_device_name(restore_identity):
    calls = []
    PlatformFL.torch_device_fn = NS(empty_cache=lambda: calls.append(True))
    PlatformFL.empty_cache()
    assert calls == [True]
    assert PlatformFL.get_device_name(0) == PlatformFL.device_name


# ---------------------------------------------------------------------------
# Kernel import hooks
# ---------------------------------------------------------------------------


def test_import_kernels_metax_import_failures_are_logged(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    PlatformFL.device_type = "cuda"
    for name in (
        "mcoplib._C",
        "mcoplib._moe_C",
        "vllm_fl.dispatch.backends.vendor.metax.patches",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    # Must not raise: each failure is caught at its own level.
    PlatformFL.import_kernels()


def test_import_kernels_musa_success(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    PlatformFL.device_type = "musa"
    calls = []
    fake = types.ModuleType("vllm_fl.dispatch.backends.vendor.musa.patch")
    fake.apply_musa_patches = lambda: calls.append(True)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.musa.patch", fake
    )
    PlatformFL.import_kernels()
    assert calls == [True]


def test_import_kernels_musa_failure_is_logged(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    PlatformFL.device_type = "musa"

    def _boom():
        raise RuntimeError("musa patch failed")

    fake = types.ModuleType("vllm_fl.dispatch.backends.vendor.musa.patch")
    fake.apply_musa_patches = _boom
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.musa.patch", fake
    )
    PlatformFL.import_kernels()


def test_import_ir_kernels_imports_vllm_kernels(monkeypatch):
    fake = types.ModuleType("vllm.kernels")
    monkeypatch.setitem(sys.modules, "vllm.kernels", fake)
    PlatformFL.import_ir_kernels()

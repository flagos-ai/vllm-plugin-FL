# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for PlatformFL attention/ViT backend selection."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_fl.platform as platform_mod
from vllm_fl.platform import PlatformFL

_counter = {"n": 0}


def _sub(**attrs):
    _counter["n"] += 1
    return type(f"AttnPlatform{_counter['n']}", (PlatformFL,), attrs)


def _install_fake_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child, module, raising=False)
    return module


def test_get_attn_backend_cls_dispatches(monkeypatch):
    call_op = Mock(return_value="dispatched-backend")
    _install_fake_module(monkeypatch, "vllm_fl.dispatch", call_op=call_op)
    selector = SimpleNamespace(use_mla=True, use_sparse=True)
    cls = _sub(vendor_name="mthreads")
    assert cls.get_attn_backend_cls(None, selector) == "dispatched-backend"
    call_op.assert_called_once_with(
        "attention_backend", use_mla=True, use_sparse=True
    )


def test_get_attn_backend_cls_amd_rejects_sparse():
    selector = SimpleNamespace(use_mla=True, use_sparse=True)
    with pytest.raises(NotImplementedError, match="Sparse MLA is unsupported"):
        _sub(vendor_name="amd").get_attn_backend_cls(None, selector)


def test_supported_vit_backends_default():
    assert _sub(vendor_name="nvidia").get_supported_vit_attn_backends() == [
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FLASH_ATTN,
    ]


def test_supported_vit_backends_amd_defers_to_rocm(monkeypatch):
    class _FakeRocm:
        @classmethod
        def get_supported_vit_attn_backends(cls):
            return ["rocm-backend"]

    _install_fake_module(
        monkeypatch,
        "vllm.platforms.rocm",
        RocmPlatform=_FakeRocm,
    )
    assert _sub(vendor_name="amd").get_supported_vit_attn_backends() == [
        "rocm-backend"
    ]


def test_supported_vit_backends_amd_without_rocm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", None)
    assert _sub(vendor_name="amd").get_supported_vit_attn_backends() == [
        AttentionBackendEnum.TORCH_SDPA
    ]


def test_get_vit_attn_backend_respects_explicit_supported_backend(monkeypatch):
    patch = Mock()
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=patch
    )
    cls = _sub(vendor_name="mthreads")
    result = cls.get_vit_attn_backend(64, "float16", AttentionBackendEnum.TORCH_SDPA)
    assert result == AttentionBackendEnum.TORCH_SDPA
    patch.assert_called_once_with()


def test_get_vit_attn_backend_rejects_unsupported_explicit_backend(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    with pytest.raises(AssertionError):
        _sub(vendor_name="mthreads").get_vit_attn_backend(64, "float16", "bogus")


def _gcu_backend(supports_head=True, supports_dtype=True):
    return SimpleNamespace(
        supports_head_size=lambda head_size: supports_head,
        supports_dtype=lambda dtype: supports_dtype,
    )


def test_get_vit_attn_backend_gcu_explicit_flash_attn_valid(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.gcu.impl.attention",
        AttentionGCUBackend=_gcu_backend(),
    )
    cls = _sub(vendor_name="gcu")
    assert cls.get_vit_attn_backend(64, "float16", AttentionBackendEnum.FLASH_ATTN) == (
        AttentionBackendEnum.FLASH_ATTN
    )


def test_get_vit_attn_backend_gcu_explicit_flash_attn_invalid(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.gcu.impl.attention",
        AttentionGCUBackend=_gcu_backend(supports_head=False),
    )
    with pytest.raises(ValueError, match="requires FP16/BF16"):
        _sub(vendor_name="enflame").get_vit_attn_backend(
            0, "float32", AttentionBackendEnum.FLASH_ATTN
        )


def test_get_vit_attn_backend_gcu_auto_prefers_flash_attn(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.gcu.impl.attention",
        AttentionGCUBackend=_gcu_backend(),
    )
    assert _sub(vendor_name="gcu").get_vit_attn_backend(64, "float16") == (
        AttentionBackendEnum.FLASH_ATTN
    )


def test_get_vit_attn_backend_gcu_auto_falls_back_to_sdpa(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    _install_fake_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.gcu.impl.attention",
        AttentionGCUBackend=_gcu_backend(supports_dtype=False),
    )
    assert _sub(vendor_name="enflame").get_vit_attn_backend(64, "float32") == (
        AttentionBackendEnum.TORCH_SDPA
    )


def test_get_vit_attn_backend_amd_defers_to_rocm(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    rocm = Mock(return_value="rocm-vit")

    class _FakeRocm:
        @classmethod
        def get_vit_attn_backend(cls, head_size, dtype, backend):
            return rocm(head_size, dtype, backend)

    _install_fake_module(
        monkeypatch,
        "vllm.platforms.rocm",
        RocmPlatform=_FakeRocm,
    )
    cls = _sub(vendor_name="amd")
    assert cls.get_vit_attn_backend(64, "float16") == "rocm-vit"
    rocm.assert_called_once_with(64, "float16", None)


def test_get_vit_attn_backend_amd_without_rocm(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", None)
    assert _sub(vendor_name="amd").get_vit_attn_backend(64, "float16") == (
        AttentionBackendEnum.TORCH_SDPA
    )


class _FakeFlashBackend:
    def __init__(self, head_ok=True, dtype_ok=True, raises=False):
        self._head_ok = head_ok
        self._dtype_ok = dtype_ok
        self._raises = raises

    def supports_head_size(self, head_size):
        return self._head_ok

    def supports_dtype(self, dtype):
        return self._dtype_ok


def _fake_enum(flash_backend):
    def get_class():
        if isinstance(flash_backend, Exception):
            raise flash_backend
        return flash_backend

    flash = SimpleNamespace(get_class=get_class)
    return SimpleNamespace(TORCH_SDPA="sdpa", FLASH_ATTN=flash)


def test_get_vit_attn_backend_capability_chooses_flash(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    fake_enum = _fake_enum(_FakeFlashBackend())
    monkeypatch.setattr(platform_mod, "AttentionBackendEnum", fake_enum)
    cls = _sub(
        vendor_name="mthreads",
        get_device_capability=classmethod(
            lambda cls, device_id=0: SimpleNamespace(major=8)
        ),
    )
    assert cls.get_vit_attn_backend(64, "float16") is fake_enum.FLASH_ATTN


def test_get_vit_attn_backend_capability_too_low(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    get_class = Mock()
    monkeypatch.setattr(
        platform_mod,
        "AttentionBackendEnum",
        SimpleNamespace(
            TORCH_SDPA="sdpa", FLASH_ATTN=SimpleNamespace(get_class=get_class)
        ),
    )
    cls = _sub(
        vendor_name="mthreads",
        get_device_capability=classmethod(
            lambda cls, device_id=0: SimpleNamespace(major=7)
        ),
    )
    assert cls.get_vit_attn_backend(64, "float16") == "sdpa"
    get_class.assert_not_called()


def test_get_vit_attn_backend_capability_unknown(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    monkeypatch.setattr(
        platform_mod,
        "AttentionBackendEnum",
        SimpleNamespace(TORCH_SDPA="sdpa", FLASH_ATTN=SimpleNamespace(get_class=Mock())),
    )
    cls = _sub(
        vendor_name="mthreads",
        get_device_capability=classmethod(lambda cls, device_id=0: None),
    )
    assert cls.get_vit_attn_backend(64, "float16") == "sdpa"


def test_get_vit_attn_backend_flash_import_error(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    monkeypatch.setattr(
        platform_mod,
        "AttentionBackendEnum",
        _fake_enum(ImportError("no flash")),
    )
    cls = _sub(
        vendor_name="mthreads",
        get_device_capability=classmethod(
            lambda cls, device_id=0: SimpleNamespace(major=9)
        ),
    )
    assert cls.get_vit_attn_backend(64, "float16") == "sdpa"


def test_get_vit_attn_backend_flash_unsupported_head(monkeypatch):
    _install_fake_module(
        monkeypatch, "vllm_fl.attention.utils", patch_mm_encoder_attention=Mock()
    )
    monkeypatch.setattr(
        platform_mod,
        "AttentionBackendEnum",
        _fake_enum(_FakeFlashBackend(head_ok=False)),
    )
    cls = _sub(
        vendor_name="mthreads",
        get_device_capability=classmethod(
            lambda cls, device_id=0: SimpleNamespace(major=8)
        ),
    )
    assert cls.get_vit_attn_backend(64, "float16") == "sdpa"

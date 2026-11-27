# Copyright 2025 BAAI. All rights reserved.

"""TheadBackend class contract for the T-Head (PPU) dispatch backend.

The backend's availability probe is a five-branch ladder (no CUDA, matching
vendor, PPU_SDK fallback env, neither, import failure) memoized in a *class*
attribute, so every test resets ``TheadBackend._available`` before and after.
The attention backend path must fall back to the vLLM FLASH_ATTN registry
entry for MLA/sparse so unsupported attention modes never reach the PPU FA3
bridge.

All tests are CPU-only: CUDA and the platform vendor are stubbed.
"""

from __future__ import annotations

import contextlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.supplementary]


@pytest.fixture(autouse=True)
def reset_availability_memo():
    """Reset the class-level availability memo around every test."""
    from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

    TheadBackend._available = None
    yield
    TheadBackend._available = None


def _cuda_present(monkeypatch, device_count=4):
    """Make ``torch.cuda`` report a healthy CUDA device stack."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: device_count)


@contextlib.contextmanager
def _vendor(vendor_name):
    """Stub ``vllm.platforms`` for the probe's inner from-import.

    ``thead.py`` imports ``from vllm.platforms import current_platform``
    *inside* ``is_available()``. Patching the module attribute is not enough
    on hosts where vLLM's lazy platform resolution itself raises (an ``empty``
    wheel has no CUDA extension to resolve), so the whole module is swapped
    for a stub and restored afterwards (save/install/restore in ``finally``;
    no stub ever leaks into ``sys.modules``).
    """
    saved = sys.modules.get("vllm.platforms")
    stub = ModuleType("vllm.platforms")
    stub.current_platform = SimpleNamespace(vendor_name=vendor_name)
    sys.modules["vllm.platforms"] = stub
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("vllm.platforms", None)
        else:
            sys.modules["vllm.platforms"] = saved


class TestIsAvailable:
    """The five-branch availability ladder of ``TheadBackend.is_available``."""

    @pytest.mark.platforms("thead")
    def test_no_cuda_reports_unavailable(self, monkeypatch):
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        assert TheadBackend().is_available() is False
        assert TheadBackend._available is False

    @pytest.mark.platforms("thead")
    def test_cuda_with_thead_platform_vendor(self, monkeypatch):
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        _cuda_present(monkeypatch)
        monkeypatch.delenv("PPU_SDK", raising=False)
        with _vendor("thead"):
            assert TheadBackend().is_available() is True
            assert TheadBackend._available is True

    @pytest.mark.platforms("thead")
    def test_other_vendor_with_ppu_sdk_env_fallback(self, monkeypatch):
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        _cuda_present(monkeypatch)
        monkeypatch.setenv("PPU_SDK", "/opt/ppu/sdk")
        with _vendor("kunlunxin"):
            assert TheadBackend().is_available() is True
            assert TheadBackend._available is True

    @pytest.mark.platforms("thead")
    def test_other_vendor_without_env_reports_unavailable(self, monkeypatch):
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        _cuda_present(monkeypatch)
        monkeypatch.delenv("PPU_SDK", raising=False)
        with _vendor("kunlunxin"):
            assert TheadBackend().is_available() is False
            assert TheadBackend._available is False

    @pytest.mark.platforms("thead")
    def test_platform_import_failure_reports_unavailable(self, monkeypatch):
        """A broken ``vllm.platforms`` import degrades to False, not a raise.

        Setting ``sys.modules["vllm.platforms"] = None`` makes
        ``from vllm.platforms import current_platform`` raise ImportError,
        which the probe's except-path must swallow.
        """
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        _cuda_present(monkeypatch)
        saved = sys.modules.get("vllm.platforms")
        sys.modules["vllm.platforms"] = None
        try:
            assert TheadBackend().is_available() is False
            assert TheadBackend._available is False
        finally:
            if saved is None:
                sys.modules.pop("vllm.platforms", None)
            else:
                sys.modules["vllm.platforms"] = saved

    @pytest.mark.platforms("thead")
    def test_platform_without_vendor_name_uses_env_fallback(self, monkeypatch):
        """A platform lacking ``vendor_name`` falls through to the env check."""
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        _cuda_present(monkeypatch)
        monkeypatch.setenv("PPU_SDK", "set")
        saved = sys.modules.get("vllm.platforms")
        stub = ModuleType("vllm.platforms")
        stub.current_platform = SimpleNamespace()  # no vendor_name at all
        sys.modules["vllm.platforms"] = stub
        try:
            assert TheadBackend().is_available() is True
        finally:
            if saved is None:
                sys.modules.pop("vllm.platforms", None)
            else:
                sys.modules["vllm.platforms"] = saved


class TestAvailabilityMemoization:
    """The probe result is memoized in a class attribute (probed once)."""

    @pytest.mark.platforms("thead")
    def test_probe_runs_once_and_caches_result(self, monkeypatch):
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        calls = []

        def counting_is_available():
            calls.append(1)
            return True

        monkeypatch.setattr("torch.cuda.is_available", counting_is_available)
        monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
        with _vendor("thead"):
            backend = TheadBackend()
            assert backend.is_available() is True
            assert backend.is_available() is True
        assert len(calls) == 1, "availability probe must be memoized per class"

    @pytest.mark.platforms("thead")
    def test_cached_result_survives_vendor_change(self, monkeypatch):
        """After memoization the probe short-circuits before touching CUDA."""
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        _cuda_present(monkeypatch)
        with _vendor("thead"):
            assert TheadBackend().is_available() is True

        # Simulate a probe that would flip the answer: CUDA disappears and
        # the vendor changes. The memo must win over a re-probe.
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        with _vendor("kunlunxin"):
            monkeypatch.delenv("PPU_SDK", raising=False)
            assert TheadBackend().is_available() is True


class TestAttentionBackendPath:
    """``attention_backend`` returns the FA3 bridge or the vLLM fallback."""

    BRIDGE_PATH = (
        "vllm_fl.dispatch.backends.vendor.thead.impl.attention."
        "TheadFlashAttentionBackend"
    )

    @pytest.mark.platforms("thead")
    def test_default_returns_fa3_bridge_path(self):
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        assert TheadBackend().attention_backend() == self.BRIDGE_PATH

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"use_mla": True},
            {"use_sparse": True},
            {"use_mla": True, "use_sparse": True},
        ],
        ids=["mla", "sparse", "mla_and_sparse"],
    )
    @pytest.mark.platforms("thead")
    def test_mla_or_sparse_falls_back_to_vllm_flash_attn(self, kwargs):
        pytest.importorskip(
            "vllm.v1.attention.backends.registry",
            reason="vLLM attention registry required for the fallback path",
        )
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        expected = AttentionBackendEnum.FLASH_ATTN.get_path()
        assert expected != self.BRIDGE_PATH, "fallback must differ from the bridge"
        result = TheadBackend().attention_backend(**kwargs)
        assert result == expected

    @pytest.mark.platforms("thead")
    def test_explicit_defaults_match_default_call(self):
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        backend = TheadBackend()
        assert backend.attention_backend() == backend.attention_backend(
            use_mla=False, use_sparse=False
        )


class TestBackendIdentity:
    """Name and vendor properties pin the backend's identity."""

    @pytest.mark.platforms("thead")
    def test_name_and_vendor(self):
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        backend = TheadBackend()
        assert backend.name == "thead"
        assert backend.vendor == "thead"

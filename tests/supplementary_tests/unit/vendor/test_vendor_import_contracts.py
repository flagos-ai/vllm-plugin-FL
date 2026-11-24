# Copyright 2025 BAAI. All rights reserved.

"""Vendor backend import contracts for the four supplementary platforms.

These tests pin the import-time surface of each vendor backend package:
the backend class, its name/vendor properties, and (where applicable)
the register_ops module's ability to build OpImpl registrations against a
fresh registry. They are CPU-only (no vendor runtime is touched) and
platform-gated so each vendor's contract runs on its own platform only.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.supplementary

VENDOR_MODULES = {
    "kunlunxin": "vllm_fl.dispatch.backends.vendor.kunlunxin",
    "hygon": "vllm_fl.dispatch.backends.vendor.hygon",
    "iluvatar": "vllm_fl.dispatch.backends.vendor.iluvatar",
    "thead": "vllm_fl.dispatch.backends.vendor.thead",
}


def _import_vendor_module(platform: str):
    return importlib.import_module(VENDOR_MODULES[platform])


class TestVendorBackendClass:
    """Every vendor package re-exports its backend class."""

    @pytest.mark.platforms("kunlunxin")
    def test_kunlunxin_backend_export(self):
        mod = _import_vendor_module("kunlunxin")
        assert hasattr(mod, "KunlunxinBackend")

    @pytest.mark.platforms("hygon")
    def test_hygon_patch_entry_point(self):
        # hygon's public surface is the patch entry, not a backend class
        from vllm_fl.dispatch.backends.vendor.hygon import patch

        assert hasattr(patch, "apply_hygon_patches")

    @pytest.mark.platforms("iluvatar")
    def test_iluvatar_backend_export(self):
        mod = _import_vendor_module("iluvatar")
        assert hasattr(mod, "IluvatarBackend")

    @pytest.mark.platforms("thead")
    def test_thead_backend_export(self):
        mod = _import_vendor_module("thead")
        assert hasattr(mod, "TheadBackend")


class TestRegisterOpsContract:
    """register_ops modules register vendor OpImpls into a fresh registry.

    Uses the real registry type from vllm_fl.dispatch; each call must
    produce at least one VENDOR-kind impl carrying the vendor's name.
    """

    @staticmethod
    def _fresh_registry():
        from vllm_fl.dispatch.registry import OpRegistry

        return OpRegistry()

    @pytest.mark.platforms("kunlunxin")
    def test_kunlunxin_registers_vendor_ops(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin import register_ops

        registry = self._fresh_registry()
        register_ops.register_builtins(registry)
        snapshot = registry.snapshot()
        impls = [i for impls in snapshot.impls_by_op.values() for i in impls]
        assert impls, "kunlunxin register_builtins registered no impls"
        assert all(i.impl_id.startswith("vendor.kunlunxin") for i in impls)

    @pytest.mark.platforms("iluvatar")
    def test_iluvatar_registers_vendor_ops(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar import register_ops

        registry = self._fresh_registry()
        register_ops.register_builtins(registry)
        snapshot = registry.snapshot()
        impls = [i for impls in snapshot.impls_by_op.values() for i in impls]
        assert impls, "iluvatar register_builtins registered no impls"
        assert all(i.impl_id.startswith("vendor.iluvatar") for i in impls)

    @pytest.mark.platforms("thead")
    def test_thead_registers_vendor_ops(self):
        from vllm_fl.dispatch.backends.vendor.thead import register_ops

        registry = self._fresh_registry()
        register_ops.register_builtins(registry)
        snapshot = registry.snapshot()
        impls = [i for impls in snapshot.impls_by_op.values() for i in impls]
        assert impls, "thead register_builtins registered no impls"
        assert all(i.impl_id.startswith("vendor.thead") for i in impls)

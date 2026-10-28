# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for flagcx communicator module.

Integration tests for actual distributed operations should be in functional_tests/.
"""

import importlib
import sys
import types


def _install_fake_pre_api_flagcx(monkeypatch):
    """Replace ``flagcx`` in ``sys.modules`` with a package that has no
    ``flagcx.api`` submodule, simulating an installed pre-API wheel."""
    fake = types.ModuleType("flagcx")
    fake.__path__ = []  # a package whose api.py does not exist
    fake.__spec__ = types.SimpleNamespace(name="flagcx", submodule_search_locations=[])
    monkeypatch.setitem(sys.modules, "flagcx", fake)
    monkeypatch.delitem(sys.modules, "flagcx.api", raising=False)


def test_pre_api_package_without_flagcx_path_keeps_native_backend(monkeypatch):
    """Regression: an installed pre-API flagcx package must not enable the
    FlagCX backend when FLAGCX_PATH is unset."""
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    _install_fake_pre_api_flagcx(monkeypatch)

    from vllm_fl.platform import _flagcx_backend_enabled

    assert _flagcx_backend_enabled() is False


def test_pre_api_package_without_flagcx_path_disables_communicator(monkeypatch):
    """Regression: the communicator module must resolve FLAGCXLibrary=None
    (and _flagcx_available=False) for a pre-API package with no source tree."""
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    _install_fake_pre_api_flagcx(monkeypatch)

    module_name = "vllm_fl.distributed.device_communicators.flagcx"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    fc = importlib.import_module(module_name)

    assert fc._flagcx_available is False
    assert fc.FLAGCXLibrary is None

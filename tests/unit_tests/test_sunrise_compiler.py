# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for the Sunrise FlagTree libdevice namespace bridge."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def compiler_compat(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "vllm_fl/patches/sunrise_compiler.py"
    spec = importlib.util.spec_from_file_location("_sunrise_compiler_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vendor_package = ModuleType("triton.language.extra.ptpu")
    vendor_math = ModuleType("triton.language.extra.ptpu.libdevice")
    extra = ModuleType("triton.language.extra")
    for name in ("pow", "erf", "exp", "tanh"):
        setattr(vendor_math, name, Mock(name="ptpu_" + name))
    modules = {
        vendor_package.__name__: vendor_package,
        vendor_math.__name__: vendor_math,
        extra.__name__: extra,
    }
    for name in (module._LEGACY_PACKAGE, module._LEGACY_LIBDEVICE):
        # Record restoration even though the product hook creates the aliases.
        monkeypatch.setitem(sys.modules, name, None)
        sys.modules.pop(name)

    def import_module(name):
        if name in modules:
            return modules[name]
        if name in sys.modules:
            return sys.modules[name]
        raise ModuleNotFoundError(name=module._LEGACY_PACKAGE)

    module.importlib = SimpleNamespace(import_module=Mock(side_effect=import_module))
    return SimpleNamespace(
        module=module,
        vendor_package=vendor_package,
        vendor_math=vendor_math,
        extra=extra,
        modules=modules,
    )


def test_historical_namespace_resolves_to_real_ptpu_module(compiler_compat):
    state = compiler_compat
    assert state.module.install_sunrise_libdevice_compat() is state.vendor_math
    assert sys.modules[state.module._LEGACY_LIBDEVICE] is state.vendor_math
    assert sys.modules[state.module._LEGACY_PACKAGE] is state.vendor_package
    assert state.extra.tang is state.vendor_package
    assert state.vendor_package.libdevice is state.vendor_math
    pow_fn = state.vendor_math.pow
    assert state.module.install_sunrise_libdevice_compat() is state.vendor_math
    assert state.vendor_math.pow is pow_fn
    pow_fn.assert_not_called()


def test_existing_vendor_math_is_never_replaced(compiler_compat):
    state = compiler_compat
    original = ModuleType(state.module._LEGACY_LIBDEVICE)
    state.modules[original.__name__] = original
    assert state.module.install_sunrise_libdevice_compat() is original
    assert len(state.module.importlib.import_module.call_args_list) == 1
    assert not hasattr(state.extra, "tang")


def test_missing_executable_pow_fails_without_installing_aliases(compiler_compat):
    state = compiler_compat
    del state.vendor_math.pow
    with pytest.raises(RuntimeError, match="no executable pow"):
        state.module.install_sunrise_libdevice_compat()
    assert state.module._LEGACY_LIBDEVICE not in sys.modules
    assert not hasattr(state.extra, "tang")


def test_unrelated_import_failures_are_not_masked(compiler_compat):
    state = compiler_compat
    state.module.importlib.import_module.side_effect = ModuleNotFoundError(
        name="vendor_required_dependency"
    )
    with pytest.raises(ModuleNotFoundError, match="") as exc:
        state.module.install_sunrise_libdevice_compat()
    assert exc.value.name == "vendor_required_dependency"
    assert state.module._LEGACY_LIBDEVICE not in sys.modules


def test_sunrise_math_is_installed_before_device_detection(monkeypatch):
    import vllm_fl
    import vllm_fl.utils

    hook = Mock()
    module = ModuleType("vllm_fl.patches.sunrise_compiler")
    module.install_sunrise_libdevice_compat = hook
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("GEMS_VENDOR", "sunrise")
    monkeypatch.setattr(vllm_fl, "_install_metax_compat", lambda: None)

    def detect_device():
        hook.assert_called_once_with()
        return SimpleNamespace(vendor_name="sunrise")

    monkeypatch.setattr(vllm_fl.utils, "DeviceInfo", detect_device)
    vllm_fl._init_vendor_device()
    assert hook.call_count == 2

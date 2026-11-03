# Copyright (c) 2026 BAAI. All rights reserved.

"""Legacy compiler fixes must not replace the modern MACA runtime APIs."""

import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_METAX = Path(__file__).resolve().parents[3] / "vllm_fl/dispatch/backends/vendor/metax"
_TRITON_SHIMS = {
    "_fl_triton_experimental_stub",
    "_fl_metax_annassign_compat",
    "_fl_metax_boolchain_compat",
    "_fl_metax_kths_compat",
    "_fl_metax_load_ptr_compat",
}
_TORCH_SHIMS = {"_fl_metax_functorch_compat", "_fl_metax_torch_accel"}
_VENDOR_SHIMS = {"_fl_metax_moe_is_monolithic_compat", "_fl_metax_sampler_compat"}


def _load(monkeypatch, name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
        submodule_search_locations=[str(path.parent)] if package else None,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _runtime(monkeypatch, torch_version, triton_version):
    for name, version in (("torch", torch_version), ("triton", triton_version)):
        module = types.ModuleType(name)
        module.__version__ = version
        monkeypatch.setitem(sys.modules, name, module)
    vendor = types.ModuleType("metax_runtime_test")
    vendor.__path__ = [str(_METAX)]
    monkeypatch.setitem(sys.modules, vendor.__name__, vendor)
    return vendor.__name__


@pytest.mark.parametrize(
    "torch_version,triton_version,expected",
    [
        ("2.8.0+metax3.7.0.7", "3.0.0+metax3.7.0.7", (True, True)),
        ("2.10.0+metax3.9.0.8", "3.6.0+metax3.9.0.8", (False, False)),
        ("2.8.0+metax", "3.6.0+metax", (True, False)),
        ("2.10.0+metax", "3.0.0+metax", (False, True)),
        ("unknown", "unknown", (False, False)),
    ],
)
def test_runtime_versions_keep_torch_and_triton_gates_independent(
    monkeypatch, torch_version, triton_version, expected
):
    parent = _runtime(monkeypatch, torch_version, triton_version)
    compat = _load(monkeypatch, parent + "._compat", _METAX / "_compat.py")
    assert (compat.legacy_torch_runtime(), compat.legacy_triton_runtime()) == expected


@pytest.mark.parametrize(
    "legacy_torch,legacy_triton",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_boot_installer_only_imports_fixes_for_affected_runtime(
    monkeypatch, legacy_torch, legacy_triton
):
    parent = _runtime(
        monkeypatch,
        "2.8.0+metax" if legacy_torch else "2.10.0+metax",
        "3.0.0+metax" if legacy_triton else "3.6.0+metax",
    )
    _load(monkeypatch, parent + "._compat", _METAX / "_compat.py")
    name = parent + ".shims"
    gate = types.ModuleType(name + "._fl_metax_gate")
    gate.metax_active = lambda: True
    monkeypatch.setitem(sys.modules, gate.__name__, gate)
    for shim in _TRITON_SHIMS | _TORCH_SHIMS | _VENDOR_SHIMS:
        monkeypatch.setitem(sys.modules, name + "." + shim, types.ModuleType(shim))
    shims = _load(monkeypatch, name, _METAX / "shims/__init__.py", package=True)
    monkeypatch.setenv("FL_BOOT_SHIMS_PATCH", "1")
    imported = []
    original_import = builtins.__import__

    def capture_import(module, globals=None, locals=None, fromlist=(), level=0):
        if globals and globals.get("__package__") == name and level == 1:
            imported.extend(fromlist)
        return original_import(module, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", capture_import)
    assert shims.install()
    expected = _VENDOR_SHIMS.copy()
    if legacy_torch:
        expected |= _TORCH_SHIMS
    if legacy_triton:
        expected |= _TRITON_SHIMS
        assert imported[0] == "_fl_triton_experimental_stub"
    assert set(imported) == expected


@pytest.mark.parametrize("legacy_torch", [True, False])
def test_patch_loader_leaves_modern_torch_private_apis_alone(monkeypatch, legacy_torch):
    parent = _runtime(
        monkeypatch,
        "2.8.0+metax" if legacy_torch else "2.10.0+metax",
        "3.6.0+metax",
    )
    _load(monkeypatch, parent + "._compat", _METAX / "_compat.py")
    vllm = types.ModuleType("vllm")
    utils = types.ModuleType("vllm.utils")
    iu = types.ModuleType("vllm.utils.import_utils")
    iu.has_triton_kernels = lambda: True
    vllm.utils = utils
    utils.import_utils = iu
    for module in (vllm, utils, iu):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    imported = []

    def capture_import(name):
        imported.append(name.rsplit(".", 1)[-1])
        return types.ModuleType(name)

    monkeypatch.setattr("importlib.import_module", capture_import)
    patches = _load(
        monkeypatch, parent + ".patches", _METAX / "patches/__init__.py", package=True
    )
    assert patches.PATCH_ERRORS == {}
    for name in ("fix_standalone_compile", "functorch_config_patch"):
        assert (name in imported) is legacy_torch
    # The upstream MetaX 0.32 platform still disables third-party triton_kernels
    # on Triton 3.6; that vendor restriction is independent of the AST fixes.
    assert iu.has_triton_kernels() is False


def test_modern_triton_never_receives_experimental_placeholders(monkeypatch):
    parent = _runtime(monkeypatch, "2.10.0+metax", "3.6.0+metax")
    _load(monkeypatch, parent + "._compat", _METAX / "_compat.py")
    package = types.ModuleType(parent + ".shims")
    package.__path__ = [str(_METAX / "shims")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    stub = _load(
        monkeypatch,
        package.__name__ + "._fl_triton_experimental_stub",
        _METAX / "shims/_fl_triton_experimental_stub.py",
    )
    monkeypatch.setattr(stub, "_vendor_artifacts", lambda: True)
    monkeypatch.setattr(stub, "_needs_stub", lambda: True)
    monkeypatch.setenv("FL_TRITON_EXPERIMENTAL_STUB", "1")
    monkeypatch.delitem(sys.modules, "triton.experimental", raising=False)
    assert stub.install() is False
    assert "triton.experimental" not in sys.modules
    assert stub._AGG_HOOK is None

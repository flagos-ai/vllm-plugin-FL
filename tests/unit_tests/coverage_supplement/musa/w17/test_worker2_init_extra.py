# SPDX-License-Identifier: Apache-2.0
"""Real-import contracts for remaining vllm_fl package-init branches.

Uses the real ``vllm_fl`` module.  The FlagGems/Triton import shim, the Kunlunxin
FLA alias lock, and the ARM flag_gems-missing path are driven with stub modules.
"""

import sys
import types
from types import SimpleNamespace

import pytest


def _ensure_pkg(monkeypatch, name):
    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    return module


def _install(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_triton_compat_adds_missing_builtin_and_knobs(monkeypatch):
    import vllm_fl as m

    triton = types.ModuleType("triton")
    triton.__path__ = []
    language = types.ModuleType("triton.language")
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setattr(triton, "language", language, raising=False)
    monkeypatch.delitem(sys.modules, "triton.knobs", raising=False)
    monkeypatch.delattr(triton, "knobs", raising=False)
    monkeypatch.delattr(language, "map_elementwise", raising=False)

    try:
        m._patch_flag_gems_triton_import_compat()

        mapping = language.map_elementwise
        assert mapping.__name__ == "map_elementwise"
        assert mapping.__module__ == "triton.language"
        assert mapping.__triton_builtin__ is True
        with pytest.raises(NotImplementedError):
            mapping(1, 2)

        knobs = sys.modules["triton.knobs"]
        assert knobs.autotuning.adjust_block_size is True
        assert triton.knobs is knobs
    finally:
        sys.modules.pop("triton.knobs", None)


def test_triton_compat_keeps_present_builtin(monkeypatch):
    import vllm_fl as m

    triton = types.ModuleType("triton")
    triton.__path__ = []
    language = types.ModuleType("triton.language")
    sentinel = object()
    language.map_elementwise = sentinel
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setattr(triton, "language", language, raising=False)

    importlib_stub = SimpleNamespace(import_module=lambda name: types.ModuleType(name))
    monkeypatch.setattr(m, "importlib", importlib_stub)

    m._patch_flag_gems_triton_import_compat()
    assert language.map_elementwise is sentinel


def test_triton_compat_propagates_unrelated_module_error(monkeypatch):
    import vllm_fl as m

    triton = types.ModuleType("triton")
    triton.__path__ = []
    language = types.ModuleType("triton.language")
    language.map_elementwise = object()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setattr(triton, "language", language, raising=False)

    def _raise(name):
        error = ModuleNotFoundError(name)
        error.name = "triton.somewhere_else"
        raise error

    monkeypatch.setattr(m, "importlib", SimpleNamespace(import_module=_raise))
    with pytest.raises(ModuleNotFoundError):
        m._patch_flag_gems_triton_import_compat()


def test_triton_compat_returns_when_triton_missing(monkeypatch):
    import vllm_fl as m

    monkeypatch.setitem(sys.modules, "triton", None)
    m._patch_flag_gems_triton_import_compat()


def test_kunlunxin_alias_skips_when_lock_held(monkeypatch):
    import vllm_fl as m

    called = []
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils",
        _fix_platform_vars=lambda: called.append("fix"),
        _patch_xpu_get_device=lambda: called.append("xpu"),
    )
    _install(
        monkeypatch,
        "vllm_fl.patches.fla_module_relocation",
        install=lambda: called.append("fla"),
    )

    assert m._kunlunxin_fla_alias_lock.acquire(blocking=False)
    try:
        m._install_kunlunxin_fla_alias()
    finally:
        m._kunlunxin_fla_alias_lock.release()
    assert called == []


def _fail_flag_gems(monkeypatch, name):
    import builtins

    real_import = builtins.__import__

    def _import(mod_name, *args, **kwargs):
        if mod_name == "flag_gems":
            error = ModuleNotFoundError("No module named 'flag_gems'")
            error.name = name
            raise error
        return real_import(mod_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


def test_register_model_arm_cpu_without_flag_gems_returns(monkeypatch):
    import vllm_fl as m

    calls = []
    monkeypatch.setattr(m, "_arm_cpu_platform", lambda: "cpu-platform")
    _install(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="cpu"),
    )
    _install(
        monkeypatch,
        "vllm_fl.patches.arm_cpu_gdn",
        apply_arm_cpu_gdn_state_indices_patch=lambda: calls.append("gdn"),
    )
    _fail_flag_gems(monkeypatch, "flag_gems")

    assert m.register_model() is None
    assert calls == ["gdn"]


def test_register_model_arm_cpu_propagates_unrelated_error(monkeypatch):
    import vllm_fl as m

    monkeypatch.setattr(m, "_arm_cpu_platform", lambda: "cpu-platform")
    _install(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="cpu"),
    )
    _install(
        monkeypatch,
        "vllm_fl.patches.arm_cpu_gdn",
        apply_arm_cpu_gdn_state_indices_patch=lambda: None,
    )
    _fail_flag_gems(monkeypatch, "other_dependency")
    with pytest.raises(ModuleNotFoundError):
        m.register_model()


def test_init_vendor_device_musa_failure_raises(monkeypatch):
    import vllm_fl as m
    import vllm_fl.utils as utils_mod

    monkeypatch.setattr(m, "_install_metax_compat", lambda: None)
    monkeypatch.setattr(
        utils_mod, "DeviceInfo", lambda: SimpleNamespace(vendor_name="mthreads")
    )
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.musa.patch",
        patch_accelerator_missing_attrs=lambda: False,
    )
    with pytest.raises(RuntimeError):
        m._init_vendor_device()

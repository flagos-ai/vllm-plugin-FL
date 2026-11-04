# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Setup failure/retry contracts, independent of GPU numerical tests."""

import ast
import copy
import importlib.util
import sys
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "vllm_fl/patches/_qwen38_installation.py"
spec = importlib.util.spec_from_file_location("qwen_install_transaction", HELPER)
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)


def declaration(name):
    path = (
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/launch_profile.py"
    )
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    ns = {"copy": copy.copy}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def test_partial_attribute_install_is_rolled_back_and_retry_succeeds():
    first = SimpleNamespace(value="old")

    class FailingSetter:
        value = "old"

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if value == "fail":
                raise RuntimeError("injected setter failure")

    last = FailingSetter()
    with pytest.raises(RuntimeError, match="injected"):
        install.apply_attributes(
            [(first, "value", "new"), (first, "added", 1), (last, "value", "fail")]
        )
    assert first.value == last.value == "old"
    assert not hasattr(first, "added")
    install.apply_attributes([(first, "value", "new"), (last, "value", "new")])
    assert first.value == last.value == "new"


def test_registry_sequence_and_staticmethod_descriptor_are_restored():
    mapping, sequence = {"old": 1}, ["old"]

    class Config:
        @staticmethod
        def override(value):
            return value

    original = Config.__dict__["override"]
    with pytest.raises(RuntimeError), install.InstallationTransaction() as transaction:
        transaction.mapping(mapping)
        transaction.sequence(sequence)
        transaction.attribute(Config, "override")
        mapping["new"] = 2
        sequence.append("new")
        Config.override = staticmethod(lambda value: None)
        raise RuntimeError("injected after registry update")
    assert mapping == {"old": 1} and sequence == ["old"]
    assert Config.__dict__["override"] is original
    assert Config.override(4) == 4


def test_preset_overlay_never_writes_pinned_rows_to_native_cache():
    Cache = declaration("_PresetConfigCache")
    native = {"existing": 1}
    cache = Cache(native, {"pinned": 2})
    assert native == {"existing": 1}
    assert cache["pinned"] == 2 and cache["existing"] == 1
    cache["pinned"] = 999
    assert cache["pinned"] == 2 and "pinned" not in native
    cache["new"] = 3
    assert native["new"] == 3
    with pytest.raises(KeyError):
        _ = cache["missing"]


def test_fresh_entry_does_not_mutate_original_dispatch_caches():
    clone = declaration("_copy_entry")
    fn = object()
    original = SimpleNamespace(
        fn=fn, kernel_cache=[{"compiled": object()}], _cpu_cache={"cpu": object()}
    )
    replacement = clone(original, object())
    assert original.fn is fn and original.kernel_cache[0] and original._cpu_cache
    assert replacement.kernel_cache == [{}] and replacement._cpu_cache == {}
    assert replacement.kernel_cache is not original.kernel_cache


def test_scoped_rollback_preserves_other_model_registration():
    registry = {"other": object()}
    unrelated = object()
    ops = ["other_op"]
    with pytest.raises(RuntimeError), install.InstallationTransaction() as transaction:
        transaction.mapping(registry, ("qwen",))
        transaction.sequence_additions(ops, ("qwen_op",))
        registry["qwen"] = object()
        registry["new_other"] = unrelated
        ops.extend(["qwen_op", "new_other_op"])
        raise RuntimeError("failure after another model registered")
    assert "qwen" not in registry and registry["new_other"] is unrelated
    assert ops == ["other_op", "new_other_op"]


def sampler_module(native, monkeypatch):
    path = ROOT / "vllm_fl/dispatch/backends/vendor/metax/patches/topk_topp_sampler.py"
    tree = ast.parse(path.read_text())
    body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.Assign, ast.AnnAssign))
    ]
    namespace = {"ContextVar": ContextVar, "wraps": wraps, "topk_topp_sampler": native}
    monkeypatch.setitem(sys.modules, "vllm_fl.patches._qwen38_installation", install)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_sampling_group_install_is_idempotent(monkeypatch):
    native = SimpleNamespace(
        apply_top_k_top_p=lambda *a: None,
        random_sample=lambda *a: None,
        sample_with_exponential_noise=lambda *a: None,
    )
    original = native.random_sample
    module = sampler_module(native, monkeypatch)
    assert native.random_sample is original
    module["install_metax_sampling_patches"]()
    first = native.random_sample
    module["install_metax_sampling_patches"]()
    assert native.random_sample is first and first.__wrapped__ is original


def test_sampling_group_rolls_back_partial_publication(monkeypatch):
    class Native(SimpleNamespace):
        fail = False

        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if (
                self.fail
                and name == "sample_with_exponential_noise"
                and getattr(value, "_fl_metax_sampling_patch", False)
            ):
                self.fail = False
                raise RuntimeError("injected sampler setup failure")

    native = Native(
        apply_top_k_top_p=lambda *a: None,
        random_sample=lambda *a: None,
        sample_with_exponential_noise=lambda *a: None,
    )
    originals = dict(native.__dict__)
    module = sampler_module(native, monkeypatch)
    native.fail = True
    with pytest.raises(RuntimeError, match="injected sampler"):
        module["install_metax_sampling_patches"]()
    assert all(getattr(native, key) is value for key, value in originals.items())
    module["install_metax_sampling_patches"]()
    assert native.random_sample is module["_random_sample"]

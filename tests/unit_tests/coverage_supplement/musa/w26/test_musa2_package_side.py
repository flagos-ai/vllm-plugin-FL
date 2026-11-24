# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU controls for the MUSA package re-export module and impl package marker."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _repo_root() -> Path:
    import os

    candidates = [Path("/tmp/flsrc")]
    if os.environ.get("VLLM_FL_ROOT"):
        candidates.insert(0, Path(os.environ["VLLM_FL_ROOT"]))
    for base in [*candidates, Path.cwd(), *Path(__file__).resolve().parents]:
        if not base.is_dir():
            continue
        if (base / "vllm_fl/dispatch/backends/vendor/musa/__init__.py").is_file():
            return base
    raise RuntimeError("cannot locate vllm_fl source tree")


ROOT = _repo_root()


def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_package_init_reexports_backend(monkeypatch):
    package_name = "vllm_fl.dispatch.backends.vendor.musa"
    musa_module = _module(package_name + ".musa")

    class MusaBackend:
        pass

    musa_module.MusaBackend = MusaBackend
    monkeypatch.setitem(sys.modules, package_name + ".musa", musa_module)

    source = ROOT / "vllm_fl/dispatch/backends/vendor/musa/__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name, source, submodule_search_locations=[]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)
    spec.loader.exec_module(module)

    assert module.MusaBackend is MusaBackend
    assert module.__all__ == ["MusaBackend"]


def test_impl_package_marker_documents_operators():
    source = ROOT / "vllm_fl/dispatch/backends/vendor/musa/impl/__init__.py"
    spec = importlib.util.spec_from_file_location("w26_musa_impl_init", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "MUSA operator implementations" in (module.__doc__ or "")

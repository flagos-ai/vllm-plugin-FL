# Copyright 2026 FlagOS Contributors

import builtins
import sys
from importlib.metadata import distribution
from types import SimpleNamespace

import pytest

from vllm_fl.patches import flaggems_vllm_metax as compat


def test_release_exports_without_optional_topk():
    source = distribution("flaggems-vllm").locate_file(
        "flaggems_vllm/runtime/backend/_metax/ops/__init__.py"
    )
    before = source.read_bytes()
    code = compat._OpsLoader(compat._OPS_MODULE, str(source)).get_code(
        compat._OPS_MODULE
    )
    imported = []

    def import_op(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        if name.endswith(".persistent_topk"):
            raise ModuleNotFoundError(name="triton.experimental")
        return SimpleNamespace(**dict.fromkeys(fromlist, lambda: None))

    namespace = {"__builtins__": dict(vars(builtins), __import__=import_op)}
    exec(code, namespace)
    assert "persistent_topk" not in namespace
    assert "persistent_topk" not in namespace["__all__"]
    assert len(imported) == 8
    assert callable(namespace["fused_experts_impl"])
    assert callable(namespace["scaled_int8_quant"])
    assert source.read_bytes() == before


@pytest.mark.parametrize("switch,tle", [("0", None), ("1", object())])
def test_compatible_runtimes_are_unchanged(monkeypatch, switch, tle):
    before = list(sys.meta_path)
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", switch)
    monkeypatch.setattr(compat.importlib.util, "find_spec", lambda name: tle)
    compat.prepare_flaggems_vllm_metax()
    assert sys.meta_path == before


def test_missing_tle_installs_one_scoped_finder(monkeypatch):
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))

    def missing(name):
        raise ModuleNotFoundError(name="triton.experimental")

    monkeypatch.setattr(compat.importlib.util, "find_spec", missing)
    compat.prepare_flaggems_vllm_metax()
    compat.prepare_flaggems_vllm_metax()
    finders = [f for f in sys.meta_path if isinstance(f, compat._OpsFinder)]
    assert len(finders) == 1
    assert finders[0].find_spec("some.unrelated.module", None) is None


def test_foreign_release_is_rejected(monkeypatch):
    loader = compat._OpsLoader(compat._OPS_MODULE, "unused.py")
    monkeypatch.setattr(loader, "get_source", lambda name: "foreign source")
    with pytest.raises(ImportError, match="commit b5c2c224"):
        loader.get_code(compat._OPS_MODULE)


def test_unrelated_import_error_is_not_hidden(monkeypatch):
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")

    def broken(name):
        raise ModuleNotFoundError(name="unrelated_dependency")

    monkeypatch.setattr(compat.importlib.util, "find_spec", broken)
    with pytest.raises(ModuleNotFoundError) as error:
        compat.prepare_flaggems_vllm_metax()
    assert error.value.name == "unrelated_dependency"

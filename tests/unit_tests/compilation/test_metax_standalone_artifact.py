# Copyright (c) 2026 BAAI. All rights reserved.

"""Artifact capability selection must happen before compilation has side effects."""

import importlib
import inspect
from abc import ABC, abstractmethod
from unittest.mock import Mock

import pytest

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "metax", reason="MetaX patch")


@pytest.fixture
def modules():
    patch = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.metax.patches.fix_standalone_compile"
    )
    native = importlib.import_module("torch._inductor.standalone_compile")
    return patch, native


def test_native_concrete_artifact(modules):
    patch, native = modules
    expected = (
        native.CacheCompiledArtifact
        if inspect.isabstract(native.CompiledArtifact)
        else native.CompiledArtifact
    )
    assert patch._get_compiled_artifact_type() is expected


def test_legacy_concrete_base_without_cache_class(modules, monkeypatch):
    patch, native = modules

    class LegacyArtifact:
        pass

    monkeypatch.setattr(patch, "CompiledArtifact", LegacyArtifact)
    monkeypatch.delattr(native, "CacheCompiledArtifact", raising=False)
    assert patch._get_compiled_artifact_type() is LegacyArtifact


@pytest.mark.parametrize("missing", [True, False])
def test_invalid_artifact_fails_before_compilation(modules, monkeypatch, missing):
    patch, native = modules

    class AbstractArtifact(ABC):
        @abstractmethod
        def save(self):
            pass

    monkeypatch.setattr(patch, "CompiledArtifact", AbstractArtifact)
    if missing:
        monkeypatch.delattr(native, "CacheCompiledArtifact", raising=False)
    else:
        monkeypatch.setattr(
            native, "CacheCompiledArtifact", AbstractArtifact, raising=False
        )
    compiler = Mock(side_effect=AssertionError("Compilation must not start"))
    compiler_module = importlib.import_module("torch._inductor.compile_fx")
    monkeypatch.setattr(compiler_module, "compile_fx", compiler)
    with pytest.raises(
        (ImportError, TypeError), match="CacheCompiledArtifact|concrete"
    ):
        patch.standalone_compile_impl(None, [], dynamic_shapes="from_graph", options={})
    compiler.assert_not_called()

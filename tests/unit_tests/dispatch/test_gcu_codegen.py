# Copyright (c) 2026 BAAI. All rights reserved.
"""Regression: generated i64 pointer ABIs must not become i32 pointers."""
import inspect
from types import SimpleNamespace

import pytest

from vllm_fl.dispatch.backends.vendor.gcu.impl import inductor


def test_codegen_delegates_without_rewriting_dtypes(monkeypatch):
    from torch._inductor.graph import GraphLowering

    seen = []

    def native(self):
        seen.append(self)
        return self.source

    class Helper:
        @staticmethod
        def compile_to_module(graph):
            return graph.source.replace("int64", "int32").replace("i64", "i32")

    monkeypatch.setattr(GraphLowering, "_compile_to_module", native)
    provider = SimpleNamespace(GCUCompileHelper=Helper)
    source = "signature={'index': '*i64'}; dtype=torch.int64; tl.float64"
    graph = SimpleNamespace(source=source)
    assert Helper.compile_to_module(graph) != source
    inductor._install_gcu_dtype_preserving_codegen(provider)
    assert Helper.compile_to_module(graph) == source
    assert seen == [graph]
    assert Helper.compile_to_module is native
    inductor._install_gcu_dtype_preserving_codegen(provider)
    assert Helper.compile_to_module is native
    assert tuple(inspect.signature(Helper.compile_to_module).parameters) == ("self",)


def test_codegen_rejects_unknown_vendor_abi():
    class Helper:
        @staticmethod
        def compile_to_module(graph, options):
            return None

    original = Helper.compile_to_module
    with pytest.raises(RuntimeError, match="Unknown TorchGCU"):
        inductor._install_gcu_dtype_preserving_codegen(
            SimpleNamespace(GCUCompileHelper=Helper)
        )
    assert Helper.compile_to_module is original

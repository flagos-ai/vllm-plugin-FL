# Copyright (c) 2026 BAAI. All rights reserved.

"""NPU empty wheels cannot select CUDA-only vllm_c IR providers."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("backend,mode", [("eager", 0), ("eager", 1), ("inductor", 1)])
def test_npu_ir_defaults_select_native_for_every_operator(monkeypatch, backend, mode):
    source = Path(__file__).resolve().parents[2] / "vllm_fl/platform.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    platform = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    method = next(
        node
        for node in platform.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_default_ir_op_priority"
    )
    method.decorator_list = []

    class PriorityConfig:
        @classmethod
        def with_default(cls, default, **overrides):
            # vLLM 0.32's NPU registry provides native implementations for
            # these IR ops, but does not register the vllm_c providers.
            operators = ("rms_norm", "fused_add_rms_norm", "gelu_and_mul_sparse")
            priorities = {name: overrides.get(name, default) for name in operators}
            for priority in priorities.values():
                assert priority == ["native"]
            return priorities

    modules = {
        "vllm.config.compilation": {
            "CompilationMode": SimpleNamespace(NONE=0),
        },
        "vllm.config.kernel": {"IrOpPriorityConfig": PriorityConfig},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    result = namespace["get_default_ir_op_priority"](
        SimpleNamespace(device_type="npu"),
        SimpleNamespace(compilation_config=SimpleNamespace(backend=backend, mode=mode)),
    )
    assert set(result) == {"rms_norm", "fused_add_rms_norm", "gelu_and_mul_sparse"}

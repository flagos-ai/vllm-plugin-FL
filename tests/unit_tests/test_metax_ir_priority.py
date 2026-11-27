# Copyright (c) 2026 BAAI. All rights reserved.
"""Empty-wheel schemas must not select unregistered IR implementations."""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("inductor", [False, True])
def test_metax_ir_defaults_only_select_registered_implementations(
    monkeypatch, inductor
):
    @dataclass
    class Priority:
        rms_norm: list
        fused_add_rms_norm: list
        gelu_and_mul_sparse: list

    registry = {
        "rms_norm": SimpleNamespace(impls={"native": object()}),
        "fused_add_rms_norm": SimpleNamespace(
            impls={"native": object(), "vllm_c": object()}
        ),
        "gelu_and_mul_sparse": SimpleNamespace(impls={"native": object()}),
    }
    modules = {
        name: ModuleType(name)
        for name in (
            "vllm",
            "vllm.config",
            "vllm.config.compilation",
            "vllm.config.kernel",
            "vllm.ir",
            "vllm.ir.ops",
            "vllm.ir.op",
        )
    }
    modules["vllm.config.compilation"].CompilationMode = SimpleNamespace(NONE=0)
    modules["vllm.config.kernel"].IrOpPriorityConfig = Priority
    modules["vllm.ir.op"].IrOp = SimpleNamespace(registry=registry)
    for name, module in modules.items():
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[2] / "vllm_fl/platform.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    platform = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    function = next(
        node
        for node in platform.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_default_ir_op_priority"
    )
    function.decorator_list = []
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    priorities = namespace[function.name](
        SimpleNamespace(vendor_name="metax"),
        SimpleNamespace(
            compilation_config=SimpleNamespace(
                backend="inductor" if inductor else "eager", mode=1 if inductor else 0
            )
        ),
    )
    assert priorities.rms_norm == ["native"]
    assert priorities.gelu_and_mul_sparse == ["native"]
    assert priorities.fused_add_rms_norm == (
        ["native"] if inductor else ["vllm_c", "native"]
    )

# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for TXDA defaults against vLLM's registered IR providers."""

import ast
import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class CompilationMode(IntEnum):
    NONE = 0
    VLLM_COMPILE = 3


class Priorities:
    op_names = ("rms_norm", "fused_add_rms_norm", "gelu_and_mul_sparse", "future_op")

    @classmethod
    def with_default(cls, default, **overrides):
        return SimpleNamespace(
            **{name: list(overrides.get(name, default)) for name in cls.op_names}
        )


@pytest.fixture
def get_priorities(monkeypatch):
    compilation = ModuleType("vllm.config.compilation")
    compilation.CompilationMode = CompilationMode
    kernel = ModuleType("vllm.config.kernel")
    kernel.IrOpPriorityConfig = Priorities
    monkeypatch.setitem(sys.modules, compilation.__name__, compilation)
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    path = Path(__file__).resolve().parents[2] / "vllm_fl/platform.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name].__func__


@pytest.mark.parametrize(
    "backend,mode",
    [
        ("eager", CompilationMode.NONE),
        ("inductor", CompilationMode.NONE),
        ("inductor", CompilationMode.VLLM_COMPILE),
        ("other", CompilationMode.VLLM_COMPILE),
    ],
)
def test_txda_ir_defaults_use_only_registered_native_provider(
    get_priorities, backend, mode
):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(backend=backend, mode=mode)
    )
    priorities = get_priorities(SimpleNamespace(device_type="txda"), config)
    # This includes 0.32's new activation and future IR fields, so a CUDA-only
    # provider cannot silently become the default for newly added operations.
    assert vars(priorities) == {name: ["native"] for name in Priorities.op_names}


def test_txda_default_selection_does_not_replace_explicit_user_priority(get_priorities):
    explicit = SimpleNamespace(
        rms_norm=["user_provider"], fused_add_rms_norm=["native"]
    )
    config = SimpleNamespace(kernel_config=SimpleNamespace(ir_op_priority=explicit))
    priorities = get_priorities(SimpleNamespace(device_type="txda"), config)
    assert config.kernel_config.ir_op_priority is explicit
    assert explicit.rms_norm == ["user_provider"]
    assert explicit.fused_add_rms_norm == ["native"]
    assert priorities.rms_norm == ["native"]


@pytest.mark.parametrize("device_type", ["cuda", "npu", "musa", "ptpu", "gcu"])
@pytest.mark.parametrize("compiled", [False, True])
def test_other_backend_ir_defaults_remain_unchanged(
    get_priorities, device_type, compiled
):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            backend="inductor",
            mode=CompilationMode.VLLM_COMPILE if compiled else CompilationMode.NONE,
        )
    )
    priorities = get_priorities(SimpleNamespace(device_type=device_type), config)
    expected = ["native"] if compiled else ["vllm_c", "native"]
    assert vars(priorities) == {name: expected for name in Priorities.op_names}

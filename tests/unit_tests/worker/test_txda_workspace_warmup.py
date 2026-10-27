# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for eager TXDA warmup and shared workspace sizing."""

import __future__

import ast
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class _CompilationMode(IntEnum):
    NONE = 0
    VLLM_COMPILE = 3


class _GraphMode(IntEnum):
    NONE = 0
    PIECEWISE = 1


class _CompileRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __contains__(self, size):
        return self.start <= size <= self.end


class _Workspace:
    def __init__(self):
        self.capacity = 0
        self.locked = False

    def reserve(self, tokens):
        # Reproduce the observed scale: small captures reserve 0.50 MiB,
        # whereas 177 prefill tokens require about 1.69 MiB.
        required = max(512 * 1024, tokens * 10_000)
        if self.locked and required > self.capacity:
            raise RuntimeError("Workspace is locked and cannot grow")
        self.capacity = max(self.capacity, required)


class _Runner:
    def __init__(self, graph_mode, capture_sizes):
        self.graph_mode = graph_mode
        self.capture_sizes = capture_sizes
        self.workspace = _Workspace()
        self.events = []
        self.dummy_calls = []
        self.lora_config = object()

    def _dummy_run(self, size, **kwargs):
        self.events.append(("dummy", size))
        self.dummy_calls.append((size, kwargs))
        self.workspace.reserve(size)

    def maybe_remove_all_loras(self, config):
        assert config is self.lora_config
        self.events.append(("remove_loras",))

    def capture_model(self):
        self.events.append(("capture",))
        if self.graph_mode == _GraphMode.NONE:
            return 0
        for size in self.capture_sizes:
            self.workspace.reserve(size)
        self.workspace.locked = True
        self.events.append(("lock", self.workspace.capacity))
        return 0


@pytest.fixture(scope="module")
def warmup_code():
    source = Path(__file__).resolve().parents[3] / "vllm_fl/worker/worker.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    )
    method = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "compile_or_warm_up_model"
    )
    capture_statements = [
        index
        for index, statement in enumerate(method.body)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "capture_model"
            for node in ast.walk(statement)
        )
    ]
    assert len(capture_statements) == 1
    # Keep every real statement through graph capture. Later sampler warmup,
    # memory reporting, and timing results are outside this workspace contract.
    method.body = method.body[: capture_statements[0] + 1]
    return compile(
        ast.Module(body=[method], type_ignores=[]),
        str(source),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )


def _run_warmup(
    code,
    *,
    device="txda",
    graph_mode=_GraphMode.PIECEWISE,
    enforce_eager=False,
    compile_sizes=None,
    profiled_tokens=None,
):
    capture_sizes = [1, 2, 4, 8]
    compilation = SimpleNamespace(
        mode=_CompilationMode.VLLM_COMPILE,
        cudagraph_mode=graph_mode,
        compile_sizes=compile_sizes,
        cudagraph_capture_sizes=capture_sizes,
        get_compile_ranges=lambda: [_CompileRange(1, 512)],
    )
    if enforce_eager:
        compilation.mode = _CompilationMode.NONE
    runner = _Runner(graph_mode, capture_sizes)
    if profiled_tokens is not None:
        runner.workspace.reserve(profiled_tokens)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(compilation_config=compilation),
        compilation_config=compilation,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        model_runner=runner,
    )
    kernel_warmup = Mock(side_effect=lambda _: runner.events.append(("kernels",)))
    namespace = {
        "instrument": lambda **_: lambda method: method,
        "CompilationMode": _CompilationMode,
        "CUDAGraphMode": _GraphMode,
        "current_platform": SimpleNamespace(device_type=device),
        "logger": Mock(),
        "kernel_warmup": kernel_warmup,
    }
    exec(code, namespace)
    namespace["compile_or_warm_up_model"](worker)
    return runner, kernel_warmup


def test_txda_eager_does_not_add_graph_workspace_warmup(warmup_code):
    runner, kernel_warmup = _run_warmup(
        warmup_code, graph_mode=_GraphMode.NONE, enforce_eager=True
    )
    assert runner.dummy_calls == []
    assert runner.events == [("remove_loras",)]
    assert not runner.workspace.locked
    runner.workspace.reserve(177)
    kernel_warmup.assert_not_called()


@pytest.mark.parametrize("device", ["cuda", "npu", "xpu"])
def test_other_platforms_keep_compile_warmups_and_profiled_capacity(
    warmup_code, device
):
    runner, kernel_warmup = _run_warmup(
        warmup_code,
        device=device,
        compile_sizes=[8, 64, 256],
        profiled_tokens=512,
    )
    assert runner.dummy_calls == [
        (256, {"skip_eplb": True, "remove_lora": False}),
        (64, {"skip_eplb": True, "remove_lora": False}),
    ]
    assert runner.events == [
        ("dummy", 256),
        ("dummy", 64),
        ("remove_loras",),
        ("kernels",),
        ("capture",),
        ("lock", 512 * 10_000),
    ]
    runner.workspace.reserve(177)
    kernel_warmup.assert_called_once()

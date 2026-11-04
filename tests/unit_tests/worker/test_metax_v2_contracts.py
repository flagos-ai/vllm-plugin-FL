"""CPU numerical / selection contracts for opt-in MetaX upstream V2.

Production helper and upstream buffer class bodies run without GPU imports.
Native graph execution is covered separately by real MetaX GPU-marked tests.
"""

import __future__

import ast
import builtins
import dataclasses
import logging
import os
import weakref
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = Path(os.environ.get("VLLM_UPSTREAM_SOURCE_ROOT", ""))


def _upstream(relative):
    if str(UPSTREAM) != ".":
        path = UPSTREAM / relative
        assert path.is_file(), path
        return path
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None:
        pytest.skip("Requires official vLLM 0.32 source or installed package")
    return Path(spec.origin).parent / relative.removeprefix("vllm/")


def _exec_nodes(path, names, namespace):
    nodes = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if getattr(n, "name", None) in names
    ]
    assert {n.name for n in nodes} == names
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(native=False, queries=[], imports=[], weak_calls=[])
    state.platform = SimpleNamespace(vendor_name="metax", device_type="cpu")
    state.buffers = ModuleType("buffer_utils_contract")
    state.original_uva = lambda: True
    state.weak = ModuleType("torch_utils_contract")
    state.weak.weak_ref_tensor = object()

    def metax_weak(value):
        state.weak_calls.append(value)
        return value

    state.metax_weak = metax_weak
    state.weak_before = state.weak.weak_ref_tensor

    def native_query(name, key):
        state.queries.append((name, key))
        assert name == "_C::get_cuda_view_from_cpu_tensor" and key == "CPU"
        if isinstance(state.native, Exception):
            raise state.native
        return state.native

    def selected_import(name, globals=None, locals=None, fromlist=(), level=0):
        state.imports.append(name)
        if name == "vllm.platforms":
            return SimpleNamespace(current_platform=state.platform)
        if name == "vllm.v1.worker.gpu":
            return SimpleNamespace(buffer_utils=state.buffers)
        if name == "vllm.utils":
            return SimpleNamespace(torch_utils=state.weak)
        if name == "vllm_fl.compilation.graph":
            return SimpleNamespace(_weak_ref_metax_tensor=state.metax_weak)
        return builtins.__import__(name, globals, locals, fromlist, level)

    scope = dict(
        __builtins__={**vars(builtins), "__import__": selected_import},
        os=os,
        torch=SimpleNamespace(
            _C=SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=native_query)
        ),
        current_platform=state.platform,
        logger=SimpleNamespace(info_once=lambda *a: None),
    )
    _exec_nodes(
        ROOT / "vllm_fl/worker/metax_v2.py",
        {"_has_native_mapped_cpu_view", "_no_uva", "prepare_metax_v2_runtime"},
        scope,
    )
    state.prepare = scope["prepare_metax_v2_runtime"]
    state.query = scope["_has_native_mapped_cpu_view"]
    state.buffers.__dict__.update(
        __builtins__={**vars(builtins), "__import__": selected_import},
        torch=torch,
        np=np,
        Sequence=Sequence,
        Iterable=Iterable,
        _DEFAULT_MAX_CONCURRENCY=2,
        is_uva_available=state.original_uva,
        logger=SimpleNamespace(warning_once=lambda *a: None),
    )
    _exec_nodes(
        _upstream("vllm/v1/worker/gpu/buffer_utils.py"),
        {"UvaBuffer", "NonUvaBuffer", "UvaBufferPool", "UvaBackedTensor"},
        state.buffers.__dict__,
    )
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    return state


@pytest.mark.parametrize(
    "vendor,requested,use_v2",
    [
        ("nvidia", "1", True),
        ("ascend", "1", True),
        ("mthreads", "1", True),
        ("metax", None, True),
        ("metax", "0", True),
        ("metax", "1", False),
    ],
)
def test_out_of_scope_prepare_has_no_import_or_mutation(
    runtime, monkeypatch, vendor, requested, use_v2
):
    runtime.platform.vendor_name = vendor
    if requested is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER")
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", requested)
    assert runtime.prepare(SimpleNamespace(use_v2_model_runner=use_v2)) is False
    assert runtime.imports == [] and runtime.queries == []
    assert runtime.buffers.is_uva_available is runtime.original_uva
    assert runtime.weak.weak_ref_tensor is runtime.weak_before


@pytest.mark.parametrize("native", [False, True, RuntimeError("no schema")])
def test_native_detection_uses_cpu_kernel_and_preserves_native_predicate(
    runtime, native
):
    runtime.native = native
    assert runtime.prepare(SimpleNamespace(use_v2_model_runner=True))
    assert runtime.queries == [("_C::get_cuda_view_from_cpu_tensor", "CPU")]
    assert runtime.buffers.is_uva_available() is (native is True)
    if native is True:
        assert runtime.buffers.is_uva_available is runtime.original_uva
    assert runtime.weak.weak_ref_tensor is runtime.metax_weak
    assert runtime.platform.device_type == "cpu"


def test_native_detection_does_not_hide_unexpected_errors(runtime):
    error = ValueError("dispatcher bug")
    runtime.native = error
    with pytest.raises(ValueError) as caught:
        runtime.prepare(SimpleNamespace(use_v2_model_runner=True))
    assert caught.value is error


def test_prepare_idempotence_preserves_upstream_classes(runtime):
    original_classes = (
        runtime.buffers.UvaBuffer,
        runtime.buffers.NonUvaBuffer,
        runtime.buffers.UvaBufferPool,
    )
    runtime.prepare(SimpleNamespace(use_v2_model_runner=True))
    installed = runtime.buffers.is_uva_available
    runtime.prepare(SimpleNamespace(use_v2_model_runner=True))
    assert runtime.buffers.is_uva_available is installed
    assert (
        runtime.buffers.UvaBuffer,
        runtime.buffers.NonUvaBuffer,
        runtime.buffers.UvaBufferPool,
    ) == original_classes


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_real_upstream_non_uva_pool_copies_and_reuses_only_retired_slots(
    runtime, dtype
):
    runtime.prepare(SimpleNamespace(use_v2_model_runner=True))
    pool = runtime.buffers.UvaBufferPool(4, dtype, max_concurrency=2)
    assert pool._buffer_cls is runtime.buffers.NonUvaBuffer
    first = pool.copy_to_uva([1, 2, 3])
    second = pool.copy_to_uva([4, 5, 6])
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, torch.tensor([1, 2, 3], dtype=dtype))
    torch.testing.assert_close(second, torch.tensor([4, 5, 6], dtype=dtype))
    # The first slot is now explicitly retired before wrapping around.
    reused = pool.copy_to_uva([7, 8, 9])
    assert reused.data_ptr() == first.data_ptr()
    torch.testing.assert_close(reused, torch.tensor([7, 8, 9], dtype=dtype))
    torch.testing.assert_close(second, torch.tensor([4, 5, 6], dtype=dtype))
    grown = pool.copy_to_gpu(torch.arange(7, dtype=dtype))
    torch.testing.assert_close(grown, torch.arange(7, dtype=dtype))
    assert pool._uva_bufs[pool._curr].cpu.shape[0] == 8


def test_real_upstream_uva_backed_tensor_updates_data_without_aliasing_cpu(runtime):
    runtime.prepare(SimpleNamespace(use_v2_model_runner=True))
    value = runtime.buffers.UvaBackedTensor(4, torch.int32, max_concurrency=2)
    assert value.gpu.data_ptr() != value.cpu.data_ptr()
    value.cpu.copy_(torch.tensor([3, 1, 4, 2], dtype=torch.int32))
    actual = value.copy_to_uva(3)
    torch.testing.assert_close(actual, torch.tensor([3, 1, 4], dtype=torch.int32))
    value.cpu.fill_(-1)
    torch.testing.assert_close(actual, torch.tensor([3, 1, 4], dtype=torch.int32))


@pytest.mark.parametrize(
    "vendor,device,requested,expected",
    [
        ("metax", "cuda", "1", "vllm.compilation.cuda_graph.CUDAGraphWrapper"),
        ("metax", "cuda", "0", "vllm_fl.compilation.graph.GraphWrapper"),
        ("metax", "cuda", None, "vllm_fl.compilation.graph.GraphWrapper"),
        ("nvidia", "cuda", "1", "vllm_fl.compilation.graph.GraphWrapper"),
        ("ascend", "npu", "1", "vllm_fl.compilation.graph.GraphWrapper"),
        (
            "mthreads",
            "musa",
            "1",
            "vllm_fl.dispatch.backends.vendor.musa.graph.MUSAGraphWrapper",
        ),
    ],
)
def test_platform_wrapper_policy_is_scoped_to_explicit_metax_v2(
    monkeypatch, vendor, device, requested, expected
):
    tree = ast.parse((ROOT / "vllm_fl/platform.py").read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlatformFL"
    )
    cls.bases, cls.decorator_list = [], []
    cls.body = [
        n
        for n in cls.body
        if getattr(n, "name", None) == "get_static_graph_wrapper_cls"
    ]
    scope = {"os": os}
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            "selector",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    platform = scope["PlatformFL"]
    platform.vendor_name, platform.device_type = vendor, device
    if requested is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", requested)
    assert platform.get_static_graph_wrapper_cls() == expected


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def test_worker_prepares_compatibility_before_official_v2_constructor(runtime):
    tree = ast.parse((ROOT / "vllm_fl/worker/worker.py").read_text(encoding="utf-8"))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    )
    init = next(
        node for node in worker.body if getattr(node, "name", None) == "init_device"
    )
    branch = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.use_v2_model_runner"
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module == "vllm.v1.worker.gpu.model_runner"
            for child in node.body
        )
    )
    calls = []
    logged_types = []

    def official(config, device):
        assert not runtime.buffers.is_uva_available()
        calls.append("official-v2")
        return object()

    def selected_import(name, *args, **kwargs):
        if name == "vllm_fl.worker.metax_v2":
            return SimpleNamespace(prepare_metax_v2_runtime=runtime.prepare)
        if name == "vllm.v1.worker.gpu.model_runner":
            return SimpleNamespace(GPUModelRunner=official)
        raise AssertionError(name)

    runner = SimpleNamespace(
        use_v2_model_runner=True,
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        device=torch.device("cpu"),
    )
    scope = {
        "self": runner,
        "current_platform": runtime.platform,
        "logger": SimpleNamespace(
            info_once=lambda message, *args: logged_types.append(args)
        ),
        "__builtins__": {**vars(builtins), "__import__": selected_import},
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=branch.body, type_ignores=[])),
            "init-device-v2",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    assert calls == ["official-v2"] and runner.model_runner is not None
    # The constructor fixture returns an ordinary object. The diagnostic must
    # report its real type, rather than hard-coding the expected runner name.
    assert logged_types == [("builtins", "object")]


@pytest.mark.parametrize(
    "v2,vendor,compile_mode,expected",
    [
        (True, "metax", 3, True),
        (True, "metax", 0, False),
        (False, "metax", 3, False),
        (True, "nvidia", 3, False),
    ],
)
def test_worker_warms_cached_dispatch_after_model_load_only_when_needed(
    v2, vendor, compile_mode, expected
):
    tree = ast.parse((ROOT / "vllm_fl/worker/worker.py").read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    )
    method = next(
        node for node in cls.body if getattr(node, "name", None) == "load_model"
    )
    cls.bases, cls.decorator_list, cls.body = [], [], [method]
    events = []

    @contextmanager
    def context(*args, **kwargs):
        yield

    def selected_import(name, *args, **kwargs):
        assert name == "vllm_fl.dispatch"
        return SimpleNamespace(warmup_cached_ops=lambda: events.append("warmup"))

    scope = {
        "__builtins__": {**vars(builtins), "__import__": selected_import},
        "current_platform": SimpleNamespace(vendor_name=vendor),
        "CompilationMode": SimpleNamespace(NONE=0),
        "set_current_vllm_config": context,
        "set_torch_threads_for_runtime": lambda: events.append("threads"),
    }
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            "load-model-v2",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    worker = scope["WorkerFL"]()
    worker.use_v2_model_runner = v2
    worker.vllm_config = SimpleNamespace(
        weight_transfer_config=None,
        compilation_config=SimpleNamespace(mode=compile_mode),
    )
    worker._maybe_get_memory_pool_context = context
    worker._scoped_allocator_max_split = context
    worker.model_runner = SimpleNamespace(
        load_model=lambda **kwargs: events.append("load")
    )
    worker.load_model()
    assert events == (
        ["load", "warmup", "threads"] if expected else ["load", "threads"]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_wrapper_preserves_explicit_stream_values_and_v2_profile_registry(dtype):
    graph_source = _upstream("vllm/compilation/cuda_graph.py")
    events, active = [], []
    stream, original_pool, profiling_pool = object(), object(), object()

    class Graph:
        def replay(self):
            events.append("replay")
            self.callback()

    @contextmanager
    def capture(graph, *, pool, stream):
        assert stream is expected_stream and pool is profiling_pool
        events.append("capture")
        active.append(graph)
        try:
            yield
        finally:
            active.pop()

    @contextmanager
    def capture_pool(pool, config):
        yield pool

    expected_stream = stream
    scope = dict(
        dataclasses=dataclasses,
        weakref=weakref,
        ClassVar=ClassVar,
        Any=Any,
        Callable=Callable,
        CUDAGraphMode=Mode,
        ExitStack=ExitStack,
        patch=patch,
        capture_pool=capture_pool,
        logging=logging,
        torch=SimpleNamespace(
            Tensor=torch.Tensor, cuda=SimpleNamespace(CUDAGraph=Graph, graph=capture)
        ),
        logger=SimpleNamespace(
            isEnabledFor=lambda level: False, debug=lambda *args: None
        ),
        current_platform=SimpleNamespace(get_global_graph_pool=lambda: original_pool),
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: SimpleNamespace(
            batch_descriptor="batch", cudagraph_runtime_mode=Mode.PIECEWISE
        ),
        current_stream=lambda: stream,
        weak_ref_tensors=lambda value: value,
        get_offloader=lambda: SimpleNamespace(
            sync_prev_onload=lambda: None, join_after_forward=lambda: None
        ),
        validate_cudagraph_capturing_enabled=lambda: None,
        compilation_counter=SimpleNamespace(num_cudagraph_captured=0),
    )
    _exec_nodes(
        graph_source, {"CUDAGraphEntry", "CUDAGraphOptions", "CUDAGraphWrapper"}, scope
    )
    wrapper_class = scope["CUDAGraphWrapper"]

    def run(value):
        events.append("runnable")
        output = value * 2 + 1
        if active:
            active[-1].callback = lambda: output.copy_(value * 2 + 1)
        return output

    wrapper = wrapper_class(
        run,
        SimpleNamespace(compilation_config=None),
        Mode.PIECEWISE,
        scope["CUDAGraphOptions"](debug_log_enable=False, weak_ref_output=False),
    )
    # Execute the REAL V2 profiler's registry enumeration/pool swapping/clear
    # statements, rather than a parallel FL registry or copied profiler.
    profiler_path = _upstream("vllm/v1/worker/gpu/cudagraph_utils.py")
    profiler = next(
        node
        for node in ast.parse(profiler_path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "profile_cudagraph_memory"
    )
    assignment = next(
        node
        for node in ast.walk(profiler)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "all_wrappers"
            for target in node.targets
        )
    )
    loops = [
        node
        for node in ast.walk(profiler)
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "wrapper"
        and ast.unparse(node.iter) == "all_wrappers"
    ]
    clear = next(
        node
        for node in ast.walk(profiler)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "CUDAGraphWrapper.clear_all_graphs"
    )
    scope.update(
        BreakableCUDAGraphWrapper=SimpleNamespace(_all_instances=[]),
        original_pools={},
        throwaway_pool=profiling_pool,
    )

    def execute(nodes):
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]),
                str(profiler_path),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            scope,
        )

    execute([assignment, loops[0]])
    assert wrapper in scope["all_wrappers"] and wrapper.graph_pool is profiling_pool
    value = torch.ones(2, 4, dtype=dtype)
    output = wrapper(value)
    for fill in (3, -5, 7):
        value.fill_(fill)
        output.fill_(float("nan"))
        actual = wrapper(value)
        torch.testing.assert_close(
            actual, torch.full_like(value, fill * 2 + 1), rtol=0, atol=0
        )
    assert events.count("capture") == events.count("runnable") == 1
    assert events.count("replay") == 3
    execute([clear, loops[1]])
    assert (
        wrapper.concrete_cudagraph_entries == {} and wrapper.graph_pool is original_pool
    )

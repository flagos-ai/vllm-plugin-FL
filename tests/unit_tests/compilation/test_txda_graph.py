# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for TXDA graph staging, launch ownership and eager rejection."""

import __future__

import ast
import dataclasses
import importlib.util
import inspect
import sys
import threading
import weakref
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from contextvars import ContextVar
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_fl/compilation/graph.py"


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1


@pytest.fixture
def state(monkeypatch):
    events = []
    stream = SimpleNamespace(txda_stream=17)
    capturing = [None]

    class FakeGraph:
        def __init__(self):
            self.operations = []

        def capture_begin(self, *, pool):
            assert capturing[0] is None
            capturing[0] = self
            events.append("begin")

        def capture_end(self):
            assert capturing[0] is self
            capturing[0] = None
            events.append("end")

        def replay(self):
            events.append("replay")
            for function, value, output in self.operations:
                output.copy_(function(value))

    class Launcher:
        def __call__(self, gx, gy, gz, stream, function, value, output):
            if capturing[0] is None:
                output.copy_(function(value))
            else:
                capturing[0].operations.append((function, value, output))
            return SimpleNamespace(res=0)

    original_launch = Launcher.__call__
    # The native graph branch imports the offloader lazily. Keep that dependency
    # local to this CPU fixture: importing the real module while torch.txda is
    # mocked can cache an UnspecifiedPlatform for the rest of the pytest run.
    offloader = SimpleNamespace(sync_prev_onload=Mock(), join_after_forward=Mock())
    offloader_module = ModuleType("vllm.model_executor.offloader.base")
    offloader_module.get_offloader = lambda: offloader
    monkeypatch.setitem(sys.modules, offloader_module.__name__, offloader_module)
    marker = SimpleNamespace(name="TXDA_SINGLE_KERNEL_CAPTURE", cpu_children=[])
    profile_events = [SimpleNamespace(name="aten::mul", cpu_children=[marker])]
    profile = SimpleNamespace(events=lambda: profile_events)
    monkeypatch.setattr(
        torch.profiler, "profile", lambda **kwargs: nullcontext(profile)
    )
    monkeypatch.setattr(torch.profiler, "record_function", lambda name: nullcontext())
    monkeypatch.setitem(
        sys.modules,
        "triton.backends.tsingmicro.driver",
        SimpleNamespace(TXDALauncher=Launcher),
    )
    runtime = SimpleNamespace(
        current_stream=lambda: stream,
        TXDAGraph=FakeGraph,
        synchronize=lambda: events.append("sync"),
    )
    monkeypatch.setattr(torch, "txda", runtime, raising=False)
    pools = iter(range(100, 1000))
    platform = SimpleNamespace(
        device_type="txda",
        torch_device_fn=runtime,
        get_global_graph_pool=lambda: (7, 9),
        graph_pool_handle=lambda: (7, next(pools)),
    )
    forward = SimpleNamespace(
        batch_descriptor=object(), cudagraph_runtime_mode=Mode.PIECEWISE
    )
    module = ModuleType("txda_graph_contracts_under_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        torch=torch,
        current_platform=platform,
        dataclasses=dataclasses,
        inspect=inspect,
        threading=threading,
        weakref=weakref,
        ExitStack=ExitStack,
        contextmanager=contextmanager,
        ContextVar=ContextVar,
        suppress=suppress,
        patch=patch,
        logger=Mock(),
        envs=SimpleNamespace(VLLM_LOGGING_LEVEL="INFO"),
        CUDAGraphMode=Mode,
        compilation_counter=SimpleNamespace(num_cudagraph_captured=0),
        weak_ref_tensors=lambda output: output,
        current_stream=lambda: stream,
        get_forward_context=lambda: forward,
        is_forward_context_available=lambda: True,
        validate_cudagraph_capturing_enabled=lambda: None,
        set_graph_pool_id=lambda pool: None,
    )
    names = {
        "_graph_capture_stream",
        "_validate_txda_graph_runnable",
        "_TxdaGraphUnsupported",
        "_TxdaGraphInputs",
        "_TxdaGraphSequence",
        "Graph",
        "GraphEntry",
        "GraphOptions",
        "GraphWrapper",
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(SOURCE),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        module.__dict__,
    )
    # Device-independent ownership/copy contracts use actual CPU tensor storage.
    platform.device_type = "cpu"
    return SimpleNamespace(
        module=module,
        platform=platform,
        runtime=runtime,
        events=events,
        stream=stream,
        launcher=Launcher(),
        launcher_type=Launcher,
        original_launch=original_launch,
        offloader=offloader,
        profile_events=profile_events,
        forward=forward,
    )


def make_runnable(state):
    def runnable(value):
        intermediate = torch.empty_like(value)
        output = torch.empty_like(value)
        state.launcher(
            1, 1, 1, state.stream.txda_stream, lambda x: x * 2, value, intermediate
        )
        state.launcher(
            1, 1, 1, state.stream.txda_stream, lambda x: x + 1, intermediate, output
        )
        return output

    return runnable


def test_sequence_stages_fresh_storage_and_retains_intermediates(state):
    sequence = state.module._TxdaGraphSequence()
    initial = torch.full((3,), 1.0)
    output = sequence.capture(make_runnable(state), (initial,), {})
    torch.testing.assert_close(output, torch.full((3,), 3.0))
    assert len(sequence.graphs) == len(sequence.retained) == 2
    assert state.launcher_type.__call__ is state.original_launch
    retained_input = sequence.inputs.buffers[0]
    assert retained_input.data_ptr() != initial.data_ptr()
    old_inputs = [initial]
    for value in (3.0, 7.0, -2.0):
        fresh = torch.full((3,), value)
        old_inputs.append(fresh)  # prevent allocator reuse from masking stale reads
        assert fresh.data_ptr() != retained_input.data_ptr()
        sequence.inputs.update((fresh,), {})
        sequence.replay()
        torch.testing.assert_close(output, torch.full((3,), value * 2 + 1))
    torch.testing.assert_close(initial, torch.ones(3))
    refs = [weakref.ref(item) for item in (retained_input, sequence.retained[0][0][-1])]
    del retained_input
    sequence.clear()
    assert sequence.graphs == sequence.retained == []
    assert sequence.inputs is None
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("reason", ["native", "exception", "stream", "launch_error"])
def test_failed_recording_restores_launcher_and_drops_owned_storage(state, reason):
    sequence = state.module._TxdaGraphSequence()
    if reason == "native":
        state.profile_events.append(
            SimpleNamespace(name="aten::_to_copy", cpu_children=[])
        )
    runnable = make_runnable(state)
    if reason == "exception":

        def runnable(value):
            raise ValueError("test failure")
    elif reason == "stream":
        state.stream.txda_stream = 18

        def runnable(value):
            state.launcher(1, 1, 1, 17, lambda x: x, value, torch.empty_like(value))
    elif reason == "launch_error":
        state.launcher_type.__call__ = lambda *args: SimpleNamespace(res=3)
        state.original_launch = state.launcher_type.__call__
    with pytest.raises((RuntimeError, ValueError)):
        sequence.capture(runnable, (torch.ones(3),), {})
    assert state.launcher_type.__call__ is state.original_launch
    assert sequence.graphs == sequence.retained == []
    assert sequence.inputs is None
    assert state.events[-1] == "sync"


@pytest.mark.parametrize(
    "name", ["aten::_to_copy", "aten::mm", "aten::copy_", "custom::opaque", "aten::add"]
)
def test_audit_rejects_unrecorded_native_or_opaque_operations(state, name):
    profile = SimpleNamespace(
        events=lambda: [SimpleNamespace(name=name, cpu_children=[])]
    )
    with pytest.raises(state.module._TxdaGraphUnsupported, match="unrecorded"):
        state.module._TxdaGraphSequence._audit(profile)


def test_staging_preserves_exact_object_aliases_and_rejects_shared_storage_views(state):
    value = torch.ones(4)
    inputs = state.module._TxdaGraphInputs((value, value), {})
    assert inputs.args[0] is inputs.args[1]
    new = torch.full((4,), 7.0)
    inputs.update((new, new), {})
    torch.testing.assert_close(inputs.args[0], new)
    with pytest.raises(state.module._TxdaGraphUnsupported, match="aliases"):
        inputs.update((new, new.clone()), {})
    with pytest.raises(state.module._TxdaGraphUnsupported, match="storage aliases"):
        state.module._TxdaGraphInputs((value, value.view_as(value)), {})


@pytest.mark.parametrize(
    "different", [torch.ones(2), torch.ones(3, dtype=torch.float64), torch.ones(6)[::2]]
)
def test_changed_tensor_layout_is_rejected_before_any_staging_write(state, different):
    first = torch.ones(3)
    inputs = state.module._TxdaGraphInputs((first, torch.ones(3)), {})
    with pytest.raises(state.module._TxdaGraphUnsupported, match="layout"):
        inputs.update((torch.full((3,), 9.0), different), {})
    torch.testing.assert_close(inputs.args[0], first)


def test_changed_scalar_and_unsupported_input_layout_fail_closed(state):
    inputs = state.module._TxdaGraphInputs((torch.ones(3), 2), {})
    with pytest.raises(state.module._TxdaGraphUnsupported, match="scalar"):
        inputs.update((torch.ones(3), 3), {})
    with pytest.raises(state.module._TxdaGraphUnsupported, match="storage offsets"):
        state.module._TxdaGraphInputs((torch.ones(6)[1:4],), {})
    assert inputs.aliases_output(inputs.args[0].view(1, 3))


def test_wrapper_replay_uses_staging_and_clear_releases_sequence(state):
    state.module._validate_txda_graph_runnable = lambda _: (True, "")
    wrapper = state.module.GraphWrapper(
        make_runnable(state), SimpleNamespace(compilation_config=None), Mode.PIECEWISE
    )
    entry = state.module.GraphEntry(state.forward.batch_descriptor)
    wrapper.concrete_graph_entries[state.forward.batch_descriptor] = entry
    first = wrapper._call_txda(entry, (torch.ones(3),), {})
    second = wrapper._call_txda(entry, (torch.full((3,), 7.0),), {})
    assert first is second
    torch.testing.assert_close(second, torch.full((3,), 15.0))
    assert state.module.compilation_counter.num_cudagraph_captured == 2
    sequence = entry.graph
    wrapper.clear_graphs()
    assert sequence.inputs is None
    assert sequence.graphs == sequence.retained == []
    assert not wrapper.concrete_graph_entries


def test_opaque_runnable_stays_eager_without_capture_count(state):
    state.module._validate_txda_graph_runnable = lambda _: (False, "opaque")
    state.platform.device_type = "txda"
    runnable = Mock(side_effect=lambda x: x + 1)
    wrapper = state.module.GraphWrapper(
        runnable, SimpleNamespace(compilation_config=None), Mode.PIECEWISE
    )
    wrapper(torch.ones(1))
    wrapper(torch.ones(1))
    assert runnable.call_count == 2
    assert wrapper._txda_graph_disabled_reason == "opaque"
    assert state.module.compilation_counter.num_cudagraph_captured == 0
    assert state.events == []


def test_validate_real_eager_piecewise_adapter_and_reject_opaque_range(
    state, monkeypatch
):
    path = ROOT / "vllm_fl/compilation/txda_partition.py"
    spec = importlib.util.spec_from_file_location("txda_partition_under_test", path)
    partition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(partition)
    monkeypatch.setitem(sys.modules, "vllm_fl.compilation.txda_partition", partition)

    class PiecewiseBackend:
        def get_compiled_graph_wrapper(self, compiled_graph):
            def compiled_graph_wrapper(*args):
                output = compiled_graph(*args)
                return output if self.returns_tuple else output[0]

            return compiled_graph_wrapper

    monkeypatch.setitem(
        sys.modules,
        "vllm.compilation.piecewise_backend",
        SimpleNamespace(PiecewiseBackend=PiecewiseBackend),
    )
    graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)
    backend = PiecewiseBackend()
    backend.compilation_config = SimpleNamespace(backend="eager")
    backend.compiled_runnables = None
    backend.graph = graph
    backend.returns_tuple = True
    backend.range_entries = {1: SimpleNamespace(compiled=True, runnable=graph)}
    check = state.module._validate_txda_graph_runnable
    assert check(graph)[0]
    assert check(graph.forward)[0]
    assert check(backend)[0]
    backend.range_entries[1].runnable = backend.get_compiled_graph_wrapper(graph)
    assert check(backend)[0]
    backend.range_entries[1].runnable = lambda x: x
    assert not check(backend)[0]
    backend.range_entries[1].runnable = graph
    backend.compiled_runnables = {}
    assert not check(backend)[0]


def test_audit_failure_returns_eager_output_without_counting_capture(state):
    state.module._validate_txda_graph_runnable = lambda _: (True, "")
    state.profile_events.append(SimpleNamespace(name="aten::_to_copy", cpu_children=[]))
    wrapper = state.module.GraphWrapper(
        make_runnable(state), SimpleNamespace(compilation_config=None), Mode.PIECEWISE
    )
    entry = state.module.GraphEntry(state.forward.batch_descriptor)
    output = wrapper._call_txda(entry, (torch.ones(3),), {})
    torch.testing.assert_close(output, torch.full((3,), 3.0))
    assert entry.graph is None
    assert wrapper._txda_graph_disabled_reason.startswith("unrecorded")
    assert state.module.compilation_counter.num_cudagraph_captured == 0
    assert state.launcher_type.__call__ is state.original_launch
    state.platform.device_type = "txda"
    output = wrapper(torch.full((3,), 7.0))
    torch.testing.assert_close(output, torch.full((3,), 15.0))


def test_non_txda_capture_keeps_existing_native_graph_path(state):
    state.platform.device_type = "cuda"
    state.runtime.graph = Mock(return_value=nullcontext())
    runnable = Mock(side_effect=lambda value: value + 1)
    wrapper = state.module.GraphWrapper(
        runnable, SimpleNamespace(compilation_config=None), Mode.PIECEWISE
    )
    value = torch.ones(2)
    output = wrapper(value)
    assert wrapper(value) is output
    runnable.assert_called_once_with(value)
    state.runtime.graph.assert_called_once()
    assert state.offloader.sync_prev_onload.call_args_list == [call(), call()]
    state.offloader.join_after_forward.assert_called_once_with()
    assert state.module.compilation_counter.num_cudagraph_captured == 1
    assert state.launcher_type.__call__ is state.original_launch


def test_size_one_stride_change_is_rejected_even_when_contiguous(state):
    inputs = state.module._TxdaGraphInputs((torch.ones(2, 1),), {})
    value = torch.empty_strided((2, 1), (1, 13)).fill_(5)
    assert value.is_contiguous()
    with pytest.raises(state.module._TxdaGraphUnsupported, match="layout"):
        inputs.update((value,), {})
    torch.testing.assert_close(inputs.args[0], torch.ones(2, 1))


def test_opaque_tensor_subclass_is_not_staged(state):
    class CustomTensor(torch.Tensor):
        pass

    value = torch.ones(3).as_subclass(CustomTensor)
    with pytest.raises(state.module._TxdaGraphUnsupported):
        state.module._TxdaGraphInputs((value,), {})


def test_clear_then_recapture_gets_a_new_native_pool(state):
    state.module._validate_txda_graph_runnable = lambda _: (True, "")
    wrapper = state.module.GraphWrapper(
        make_runnable(state), SimpleNamespace(compilation_config=None), Mode.PIECEWISE
    )
    entry = state.module.GraphEntry(state.forward.batch_descriptor)
    wrapper.concrete_graph_entries[state.forward.batch_descriptor] = entry
    wrapper._call_txda(entry, (torch.ones(3),), {})
    old_pool = entry.graph.pool
    assert old_pool != wrapper.graph_pool
    wrapper.clear_graphs()
    entry = state.module.GraphEntry(state.forward.batch_descriptor)
    wrapper.concrete_graph_entries[state.forward.batch_descriptor] = entry
    output = wrapper._call_txda(entry, (torch.full((3,), 7.0),), {})
    torch.testing.assert_close(output, torch.full((3,), 15.0))
    assert entry.graph.pool != old_pool
    assert state.module.compilation_counter.num_cudagraph_captured == 4


def _capture_or_replay(state, wrapper):
    descriptor = state.forward.batch_descriptor
    entry = wrapper.concrete_graph_entries.setdefault(
        descriptor, state.module.GraphEntry(descriptor)
    )
    # The production sequence is exercised with real CPU tensor storage here.
    device_type = state.platform.device_type
    state.platform.device_type = "cpu"
    try:
        return wrapper._call_txda(entry, (torch.ones(3),), {})
    finally:
        state.platform.device_type = device_type


def test_live_graph_scope_excludes_other_models_and_tracks_clear_and_reuse(state):
    state.module._validate_txda_graph_runnable = lambda _: (True, "")
    cls = state.module.GraphWrapper
    config = SimpleNamespace(compilation_config=None)
    other_model = cls(make_runnable(state), config, Mode.PIECEWISE)
    _capture_or_replay(state, other_model)
    wrapper = cls(make_runnable(state), config, Mode.PIECEWISE)
    with cls.txda_capture_scope() as scope:
        assert cls.count_live_txda_graphs(scope) == 0
        _capture_or_replay(state, wrapper)
        assert cls.count_live_txda_graphs(scope) == 2
        captured_count = state.module.compilation_counter.num_cudagraph_captured
        _capture_or_replay(state, wrapper)
        assert cls.count_live_txda_graphs(scope) == 2
        assert state.module.compilation_counter.num_cudagraph_captured == captured_count
        wrapper.clear_graphs()
        assert cls.count_live_txda_graphs(scope) == 0
        _capture_or_replay(state, wrapper)
        assert cls.count_live_txda_graphs(scope) == 2
        wrapper._txda_graph_disabled_reason = "later region was unsupported"
        assert cls.count_live_txda_graphs(scope) == 0
    assert cls._active_txda_capture_scope.get() is None
    # The unrelated live model cannot mask this model's zero-graph result.
    assert len(other_model.concrete_graph_entries) == 1


def test_nested_graph_scopes_restore_parent_after_exception(state):
    state.module._validate_txda_graph_runnable = lambda _: (True, "")
    cls = state.module.GraphWrapper
    wrapper = cls(
        make_runnable(state), SimpleNamespace(compilation_config=None), Mode.PIECEWISE
    )
    with cls.txda_capture_scope() as outer:
        with (
            pytest.raises(ValueError, match="test failure"),
            cls.txda_capture_scope() as inner,
        ):
            _capture_or_replay(state, wrapper)
            raise ValueError("test failure")
        assert cls.count_live_txda_graphs(outer) == 0
        assert cls.count_live_txda_graphs(inner) == 2
        _capture_or_replay(state, wrapper)
        assert cls.count_live_txda_graphs(outer) == 2
    assert cls._active_txda_capture_scope.get() is None


@pytest.fixture
def capture_runner(state, monkeypatch):
    source = ROOT / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerFL"
    )
    method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "capture_model"
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        SimpleNamespace(get_world_group=lambda: SimpleNamespace(local_rank=0)),
    )
    events = []
    lock_workspace = Mock(side_effect=lambda: events.append("lock"))
    namespace = {
        "instrument": lambda **_: lambda function: function,
        "CUDAGraphMode": Mode,
        "GraphWrapper": state.module.GraphWrapper,
        "current_platform": state.platform,
        "compilation_counter": SimpleNamespace(num_gpu_runner_capture_triggers=0),
        "logger": Mock(),
        "time": SimpleNamespace(perf_counter=lambda: 1.0),
        "torch": SimpleNamespace(
            accelerator=SimpleNamespace(
                synchronize=lambda: None,
                empty_cache=lambda: None,
                get_memory_info=lambda: (1000, 2000),
            )
        ),
        "nullcontext": nullcontext,
        "graph_capture": lambda **_: nullcontext(),
        "set_cudagraph_capturing_enabled": lambda enabled: events.append(enabled),
        "lock_workspace": lock_workspace,
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=Mode.PIECEWISE),
        _maybe_init_encoder_cudagraph_manager=lambda: None,
        vllm_config=SimpleNamespace(
            profiler_config=SimpleNamespace(capture_torch_profiler=False)
        ),
        _freeze_gc=nullcontext,
        device="txda:0",
        cudagraph_dispatcher=SimpleNamespace(
            get_capture_descs=lambda: [
                (Mode.PIECEWISE, [state.forward.batch_descriptor])
            ]
        ),
        _capture_cudagraphs=lambda **_: None,
        encoder_cudagraph_manager=None,
    )
    state.platform.device_type = "txda"
    return SimpleNamespace(
        run=lambda: namespace["capture_model"](runner),
        runner=runner,
        lock=lock_workspace,
        events=events,
    )


def test_capture_model_rejects_zero_live_graphs_before_workspace_lock(
    state, capture_runner
):
    state.module._validate_txda_graph_runnable = lambda _: (True, "")
    unrelated = state.module.GraphWrapper(
        make_runnable(state), SimpleNamespace(compilation_config=None), Mode.PIECEWISE
    )
    _capture_or_replay(state, unrelated)
    assert state.module.compilation_counter.num_cudagraph_captured == 2
    with pytest.raises(RuntimeError, match="Use --enforce-eager"):
        capture_runner.run()
    capture_runner.lock.assert_not_called()
    assert capture_runner.events == [True, False]
    assert state.module.GraphWrapper._active_txda_capture_scope.get() is None


def test_capture_model_accepts_live_reuse_and_recapture_after_clear(
    state, capture_runner
):
    state.module._validate_txda_graph_runnable = lambda _: (True, "")
    wrapper = state.module.GraphWrapper(
        make_runnable(state), SimpleNamespace(compilation_config=None), Mode.PIECEWISE
    )
    capture_runner.runner._capture_cudagraphs = lambda **_: _capture_or_replay(
        state, wrapper
    )
    assert capture_runner.run() == 0
    assert state.module.compilation_counter.num_cudagraph_captured == 2
    assert capture_runner.run() == 0  # Existing graphs replay; none are newly captured.
    assert state.module.compilation_counter.num_cudagraph_captured == 2
    wrapper.clear_graphs()
    assert capture_runner.run() == 0
    assert state.module.compilation_counter.num_cudagraph_captured == 4
    assert capture_runner.lock.call_count == 3


def test_capture_model_does_not_count_graphs_cleared_in_the_same_scope(
    state, capture_runner
):
    state.module._validate_txda_graph_runnable = lambda _: (True, "")
    wrapper = state.module.GraphWrapper(
        make_runnable(state), SimpleNamespace(compilation_config=None), Mode.PIECEWISE
    )

    def capture_then_clear(**kwargs):
        _capture_or_replay(state, wrapper)
        wrapper.clear_graphs()

    capture_runner.runner._capture_cudagraphs = capture_then_clear
    with pytest.raises(RuntimeError, match="no live native graphs"):
        capture_runner.run()
    capture_runner.lock.assert_not_called()


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_other_platforms_do_not_require_txda_graphs(state, capture_runner, device_type):
    state.platform.device_type = device_type
    assert capture_runner.run() == 0
    capture_runner.lock.assert_called_once()


def test_capture_model_with_graphs_disabled_keeps_early_return(capture_runner):
    capture_runner.runner.compilation_config.cudagraph_mode = Mode.NONE
    assert capture_runner.run() == 0
    capture_runner.lock.assert_not_called()
    assert capture_runner.events == []

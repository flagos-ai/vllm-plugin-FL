# Copyright (c) 2026 BAAI. All rights reserved.
"""Exercise the production GraphWrapper with CPU stream doubles.

AST only locates the production dataclasses and full wrapper class. The tests
execute their original bodies, including capture contexts, replay and bypasses.
They do not import vendor modules or validate native GPU graphs or model output.
"""

import __future__

import ast
import builtins
import dataclasses
import weakref
from contextlib import ExitStack, contextmanager, suppress
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as cpu_torch
from torch.utils._pytree import tree_flatten


class _Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class _Tensor:
    """A backend-tag double backed by real CPU Torch storage and copy kernels."""

    def __init__(self, contract, values=(1.0, 2.0), *, payload=None):
        self.contract = contract
        self.payload = (
            cpu_torch.tensor(values, dtype=cpu_torch.float32)
            if payload is None
            else payload
        )
        self.overrides = {}

    @property
    def shape(self):
        return self.overrides.get("shape", self.payload.shape)

    def stride(self):
        return self.overrides.get("stride", self.payload.stride())

    @property
    def dtype(self):
        return self.overrides.get("dtype", self.payload.dtype)

    @property
    def device(self):
        return self.overrides.get("device", SimpleNamespace(type="musa", index=0))

    @property
    def layout(self):
        return self.overrides.get("layout", self.payload.layout)

    def storage_offset(self):
        return self.overrides.get("storage_offset", self.payload.storage_offset())

    @property
    def requires_grad(self):
        return self.overrides.get("requires_grad", self.payload.requires_grad)

    def data_ptr(self):
        return self.payload.data_ptr()

    def copy_(self, source):
        self.contract.events.append(("copy", self.data_ptr(), source.data_ptr()))
        self.payload.copy_(source.payload)
        return self


class _Contract:
    def __init__(self):
        self.events = []
        self.errors = {}
        self.initial_stream = object()
        self.active_stream = self.initial_stream
        self.cached_stream = object()
        self.pool = object()
        self.output = object()
        self.context_available = True
        self.context = SimpleNamespace(
            batch_descriptor="batch", cudagraph_runtime_mode=_Mode.PIECEWISE
        )
        self.counter = SimpleNamespace(num_cudagraph_captured=0)
        self.platform = SimpleNamespace(
            device_type="musa",
            torch_device_fn=SimpleNamespace(
                current_stream=self.native_current_stream,
                graph=self.capture,
                synchronize=self.synchronize,
            ),
            get_global_graph_pool=lambda: self.pool,
            graph_pool_handle=lambda: self.pool,
        )

    def native_current_stream(self):
        self.events.append(("native-current", self.active_stream))
        if "getter" in self.errors:
            raise self.errors["getter"]
        return self.active_stream

    def cached_current_stream(self):
        self.events.append(("cached-current", self.cached_stream))
        return self.cached_stream

    @contextmanager
    def native_context(self, stream):
        previous = self.active_stream
        self.active_stream = stream
        try:
            yield
        finally:
            self.active_stream = previous

    @contextmanager
    def capture(self, graph, *, pool, stream):
        self.events.append(("capture-enter", graph, pool, stream))
        if "enter" in self.errors:
            raise self.errors["enter"]
        with self.native_context(stream):
            try:
                yield
            finally:
                self.events.append(("capture-exit", graph, stream))

    def graph_factory(self):
        def replay():
            self.events.append(("replay", graph))
            if "replay" in self.errors:
                raise self.errors["replay"]

        graph = SimpleNamespace(replay=replay)
        self.events.append(("graph-created", graph))
        return graph

    def synchronize(self):
        self.events.append(("synchronize",))

    def runnable(self, *args, **kwargs):
        self.events.append(("runnable", self.active_stream, args, kwargs))
        if "body" in self.errors:
            raise self.errors["body"]
        return self.output

    def wrapper(self):
        return self.wrapper_type(
            self.runnable,
            SimpleNamespace(compilation_config=object()),
            _Mode.PIECEWISE,
            self.options_type(debug_log_enable=False, weak_ref_output=False),
        )

    def event_names(self):
        return [event[0] for event in self.events]


@pytest.fixture
def contract():
    source = Path(__file__).resolve().parents[3] / "vllm_fl/compilation/graph.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = {"GraphEntry", "GraphOptions", "GraphWrapper"}
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    assert {node.name for node in classes} == names
    state = _Contract()

    def deny_target_import(name, *args, **kwargs):
        if name in {"vllm", "torch", "torch_musa", "vllm_fl"} or name.startswith(
            ("vllm.", "torch.", "torch_musa.", "vllm_fl.")
        ):
            raise ImportError("CPU contract does not import target modules")
        return builtins.__import__(name, *args, **kwargs)

    namespace = {
        "__name__": __name__,
        "__builtins__": dict(vars(builtins), __import__=deny_target_import),
        "dataclasses": dataclasses,
        "weakref": weakref,
        "ExitStack": ExitStack,
        "suppress": suppress,
        "envs": SimpleNamespace(VLLM_LOGGING_LEVEL="INFO"),
        "current_platform": state.platform,
        "current_stream": state.cached_current_stream,
        "is_forward_context_available": lambda: state.context_available,
        "get_forward_context": lambda: state.context,
        "CUDAGraphMode": _Mode,
        "Graph": SimpleNamespace(graph=state.graph_factory),
        "tree_flatten": tree_flatten,
        "torch": SimpleNamespace(
            Tensor=_Tensor,
            _C=SimpleNamespace(
                _is_alias_of=lambda a, b: cpu_torch._C._is_alias_of(
                    a.payload, b.payload
                )
            ),
        ),
        "validate_cudagraph_capturing_enabled": lambda: None,
        "set_graph_pool_id": lambda pool: state.events.append(("pool", pool)),
        "weak_ref_tensors": lambda value: value,
        "compilation_counter": state.counter,
    }
    exec(
        compile(
            ast.Module(body=classes, type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    state.cpu_namespace = namespace
    state.cpu_classes = classes
    state.wrapper_type = namespace["GraphWrapper"]
    state.options_type = namespace["GraphOptions"]
    return state


def test_musa_capture_uses_native_context_instead_of_stale_tls(contract):
    wrapper = contract.wrapper()
    native = object()
    argument = object()
    keyword = object()
    assert native is not contract.cached_stream
    with contract.native_context(native):
        assert wrapper(argument, key=keyword) is contract.output
        assert contract.active_stream is native
    assert contract.active_stream is contract.initial_stream
    capture = next(e for e in contract.events if e[0] == "capture-enter")
    assert capture[2] is contract.pool
    assert capture[3] is native
    assert ("runnable", native, (argument,), {"key": keyword}) in contract.events
    assert contract.event_names().count("native-current") == 1
    assert "cached-current" not in contract.event_names()
    assert "synchronize" not in contract.event_names()
    assert contract.counter.num_cudagraph_captured == 1
    entry = wrapper.concrete_graph_entries["batch"]
    assert entry.graph is capture[1]
    assert entry.output is contract.output


def test_cuda_capture_keeps_vllm_cached_stream(contract):
    contract.platform.device_type = "cuda"
    wrapper = contract.wrapper()
    native = object()
    with contract.native_context(native):
        assert wrapper() is contract.output
        assert contract.active_stream is native
    capture = next(e for e in contract.events if e[0] == "capture-enter")
    assert capture[3] is contract.cached_stream
    assert ("runnable", contract.cached_stream, (), {}) in contract.events
    assert contract.event_names().count("cached-current") == 1
    assert "native-current" not in contract.event_names()


@pytest.mark.parametrize("device", ["musa", "cuda", "npu"])
def test_replay_reuses_output_and_keeps_only_existing_npu_fence(contract, device):
    contract.platform.device_type = device
    wrapper = contract.wrapper()
    assert wrapper() is contract.output
    graph = wrapper.concrete_graph_entries["batch"].graph
    contract.events.clear()
    assert wrapper() is contract.output
    expected = (
        [("synchronize",), ("replay", graph)]
        if device == "npu"
        else [("replay", graph)]
    )
    assert contract.events == expected
    assert contract.counter.num_cudagraph_captured == 1


@pytest.mark.parametrize("bypass", ["no-context", "none", "different-mode"])
def test_non_graph_path_delegates_once_without_stream_access(contract, bypass):
    wrapper = contract.wrapper()
    if bypass == "no-context":
        contract.context_available = False
    else:
        contract.context.cudagraph_runtime_mode = (
            _Mode.NONE if bypass == "none" else _Mode.FULL
        )
    argument = object()
    assert wrapper(argument, key=3) is contract.output
    assert contract.events == [
        ("runnable", contract.initial_stream, (argument,), {"key": 3})
    ]
    assert wrapper.concrete_graph_entries == {}
    assert contract.counter.num_cudagraph_captured == 0


@pytest.mark.parametrize("stage", ["getter", "enter", "body"])
def test_capture_error_propagates_identity_and_restores_outer_stream(contract, stage):
    wrapper = contract.wrapper()
    error = RuntimeError(stage + " sentinel")
    contract.errors[stage] = error
    native = object()
    with contract.native_context(native):
        with pytest.raises(RuntimeError) as actual:
            wrapper()
        assert actual.value is error
        assert contract.active_stream is native
    assert contract.active_stream is contract.initial_stream
    assert "cached-current" not in contract.event_names()
    assert contract.event_names().count("capture-enter") == (stage != "getter")
    assert contract.event_names().count("capture-exit") == (stage == "body")
    assert wrapper.concrete_graph_entries["batch"].graph is None
    assert contract.counter.num_cudagraph_captured == 0


def test_replay_error_propagates_without_an_added_fence(contract):
    wrapper = contract.wrapper()
    assert wrapper() is contract.output
    graph = wrapper.concrete_graph_entries["batch"].graph
    error = RuntimeError("replay sentinel")
    contract.errors["replay"] = error
    contract.events.clear()
    with pytest.raises(RuntimeError) as actual:
        wrapper()
    assert actual.value is error
    assert contract.events == [("replay", graph)]
    assert wrapper.concrete_graph_entries["batch"].output is contract.output


def test_musa_retains_capture_inputs_then_releases_them_on_clear(contract):
    import gc

    wrapper = contract.wrapper()
    argument = _Tensor(contract)
    reference = weakref.ref(argument)
    wrapper(argument)
    contract.events.clear()  # The diagnostic event log must not retain arguments.
    del argument
    gc.collect()
    assert reference() is not None
    assert wrapper.concrete_graph_entries["batch"].musa_input_leaves[0] is reference()
    wrapper.clear_graphs()
    gc.collect()
    assert reference() is None


def test_musa_fresh_interpiece_input_is_copied_before_replay(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract, (1.0, 2.0))
    wrapper(captured)
    entry = wrapper.concrete_graph_entries["batch"]
    address = captured.data_ptr()
    contract.events.clear()
    incoming = _Tensor(contract, (11.0, 29.0))
    assert incoming.data_ptr() != address
    assert wrapper(incoming) is contract.output
    assert cpu_torch.equal(captured.payload, cpu_torch.tensor([11.0, 29.0]))
    assert captured.data_ptr() == address == entry.input_addresses[0]
    assert contract.event_names() == ["copy", "replay"]
    assert entry.graph is wrapper.concrete_graph_entries["batch"].graph
    assert contract.counter.num_cudagraph_captured == 1


def test_musa_same_input_pointer_does_not_copy_or_recapture(contract):
    wrapper = contract.wrapper()
    argument = _Tensor(contract)
    wrapper(argument)
    contract.events.clear()
    wrapper(argument)
    assert contract.event_names() == ["replay"]


def test_musa_nested_keyword_inputs_stage_with_unchanged_static_leaves(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper([captured], scale=3, options={"tag": "same"})
    contract.events.clear()
    incoming = _Tensor(contract, (8.0, 17.0))
    wrapper([incoming], scale=3, options={"tag": "same"})
    assert cpu_torch.equal(captured.payload, incoming.payload)
    assert contract.event_names() == ["copy", "replay"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", (1, 2)),
        ("stride", (2,)),
        ("dtype", cpu_torch.int64),
        ("device", SimpleNamespace(type="musa", index=1)),
        ("layout", cpu_torch.sparse_coo),
        ("storage_offset", 1),
        ("requires_grad", True),
    ],
)
def test_musa_checks_all_metadata_before_any_copy(contract, field, value):
    wrapper = contract.wrapper()
    first, second = _Tensor(contract), _Tensor(contract)
    wrapper(first, second)
    incoming_first, incoming_second = _Tensor(contract, (9.0, 9.0)), _Tensor(contract)
    incoming_second.overrides[field] = value
    contract.events.clear()
    with pytest.raises(ValueError):
        wrapper(incoming_first, incoming_second)
    assert contract.events == []
    assert first.payload.tolist() == [1.0, 2.0]


@pytest.mark.parametrize("change", ["scalar", "structure"])
def test_musa_rejects_static_or_structure_changes_before_copy(contract, change):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper([captured], scale=2)
    contract.events.clear()
    incoming = _Tensor(contract, (9.0, 9.0))
    call_args = ([incoming],) if change == "scalar" else ((incoming,),)
    with pytest.raises(ValueError):
        wrapper(*call_args, scale=3 if change == "scalar" else 2)
    assert contract.events == []


def test_new_source_source_alias_is_rejected_using_real_cpu_storage(contract):
    wrapper = contract.wrapper()
    first, second = _Tensor(contract), _Tensor(contract)
    assert not cpu_torch._C._is_alias_of(first.payload, second.payload)
    wrapper(first, second)
    incoming_first = _Tensor(contract)
    incoming_second = _Tensor(contract, payload=incoming_first.payload.view(2))
    assert cpu_torch._C._is_alias_of(incoming_first.payload, incoming_second.payload)
    contract.events.clear()
    with pytest.raises(ValueError, match="alias relationship"):
        wrapper(incoming_first, incoming_second)
    assert contract.events == []


def test_cross_destination_alias_is_rejected_before_copy(contract):
    wrapper = contract.wrapper()
    first, second = _Tensor(contract), _Tensor(contract)
    wrapper(first, second)
    contract.events.clear()
    with pytest.raises(ValueError, match="staging destination"):
        wrapper(second, _Tensor(contract))
    assert contract.events == []


def test_distinct_captured_views_require_separate_policy(contract):
    wrapper = contract.wrapper()
    first = _Tensor(contract)
    second = _Tensor(contract, payload=first.payload.view(2))
    wrapper(first, second)
    new_first = _Tensor(contract)
    new_second = _Tensor(contract, payload=new_first.payload.view(2))
    contract.events.clear()
    with pytest.raises(ValueError, match="another captured input"):
        wrapper(new_first, new_second)
    assert contract.events == []


def test_repeated_tensor_copies_once_and_preserves_alias_relationship(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper(captured, captured)
    incoming = _Tensor(contract, (7.0, 13.0))
    contract.events.clear()
    wrapper(incoming, incoming)
    assert contract.event_names() == ["copy", "replay"]
    assert captured.payload.tolist() == [7.0, 13.0]
    contract.events.clear()
    with pytest.raises(ValueError, match="repeated MUSA graph input"):
        wrapper(_Tensor(contract), _Tensor(contract))
    assert contract.events == []


@pytest.mark.parametrize(
    "device,mode",
    [
        ("cuda", _Mode.PIECEWISE),
        ("npu", _Mode.PIECEWISE),
        ("musa", _Mode.FULL),
    ],
)
def test_staging_does_not_change_other_backend_or_full_replay(contract, device, mode):
    contract.platform.device_type = device
    contract.context.cudagraph_runtime_mode = mode
    wrapper = contract.wrapper_type(
        contract.runnable,
        SimpleNamespace(compilation_config=object()),
        mode,
        contract.options_type(debug_log_enable=False, weak_ref_output=False),
    )
    captured = _Tensor(contract)
    wrapper(captured)
    assert wrapper.concrete_graph_entries["batch"].musa_input_leaves is None
    contract.events.clear()
    wrapper(_Tensor(contract, (9.0, 9.0)))
    assert contract.event_names() == (
        ["synchronize", "replay"] if device == "npu" else ["replay"]
    )
    assert captured.payload.tolist() == [1.0, 2.0]


def test_staged_replay_error_preserves_exception_and_adds_no_fence(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper(captured)
    error = RuntimeError("native replay sentinel")
    contract.errors["replay"] = error
    contract.events.clear()
    with pytest.raises(RuntimeError) as raised:
        wrapper(_Tensor(contract, (5.0, 8.0)))
    assert raised.value is error
    assert captured.payload.tolist() == [5.0, 8.0]
    assert contract.event_names() == ["copy", "replay"]


def test_staging_validation_is_active_with_optimized_python_code(contract):
    namespace = dict(contract.cpu_namespace)
    exec(
        compile(
            ast.Module(body=contract.cpu_classes, type_ignores=[]),
            "optimized-production-wrapper",
            "exec",
            optimize=2,
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    contract.wrapper_type = namespace["GraphWrapper"]
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper(captured)
    incoming = _Tensor(contract, (19.0, 19.0))
    incoming.overrides["shape"] = (1, 2)
    contract.events.clear()
    with pytest.raises(ValueError, match="metadata"):
        wrapper(incoming)
    assert contract.events == []
    assert captured.payload.tolist() == [1.0, 2.0]

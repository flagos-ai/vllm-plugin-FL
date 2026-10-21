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
import gc
import importlib.util
import sys
import weakref
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

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

    def untyped_storage(self):
        return self.payload.untyped_storage()

    def copy_(self, source):
        self.contract.events.append(("copy", self.data_ptr(), source.data_ptr()))
        self.payload.copy_(source.payload)
        return self


class _Contract:
    def __init__(self):
        self.events = []
        self.errors = {}
        self.cache_calls = []
        self.cached_graph_ids = set()
        self.nonresident_graph_ids = set()
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

    def cache_graph_class(self, native_graph_class):
        self.cache_calls.append(("factory", native_graph_class))
        return native_graph_class

    @contextmanager
    def cache_capture(self, graph):
        assert self.platform.device_type == "musa"
        self.cache_calls.append(("capture-enter", graph))
        if "cache-capture" in self.errors:
            raise self.errors["cache-capture"]
        try:
            yield
            self.cached_graph_ids.add(id(graph))
        finally:
            self.cache_calls.append(("capture-exit", graph))

    def cache_replay(self, graph, before_replay=None):
        assert self.platform.device_type == "musa"
        assert id(graph) in self.cached_graph_ids
        if "cache-replay" in self.errors:
            raise self.errors["cache-replay"]
        if id(graph) in self.nonresident_graph_ids:
            return False
        if before_replay is not None and before_replay() is False:
            return False
        self.cache_calls.append(("replay", graph))
        graph.replay()
        return True

    def synchronize(self):
        self.events.append(("synchronize",))

    def runnable(self, *args, **kwargs):
        self.events.append(("runnable", self.active_stream, args, kwargs))
        if "body" in self.errors:
            raise self.errors["body"]
        return self.output

    @property
    def wrapper_type(self):
        key = (
            "MUSAGraphWrapper"
            if self.platform.device_type == "musa"
            else "GraphWrapper"
        )
        return self.cpu_namespace[key]

    @wrapper_type.setter
    def wrapper_type(self, value):
        key = (
            "MUSAGraphWrapper"
            if self.platform.device_type == "musa"
            else "GraphWrapper"
        )
        self.cpu_namespace[key] = value

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
    musa_source = source.with_name("musa_graph.py")
    musa_module = ast.parse(
        musa_source.read_text(encoding="utf-8"), filename=str(musa_source)
    )
    musa_names = {"MUSAGraphEntry", "MUSAGraphWrapper"}
    musa_classes = [
        node
        for node in musa_module.body
        if isinstance(node, ast.ClassDef) and node.name in musa_names
    ]
    assert {node.name for node in musa_classes} == musa_names
    classes += musa_classes
    helpers = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_graph_capture_stream"
    ]
    assert len(helpers) == 1
    state = _Contract()

    cache_api = SimpleNamespace(
        musa_graph_class=state.cache_graph_class,
        musa_graph_capture=state.cache_capture,
        musa_graph_replay=state.cache_replay,
    )

    def deny_target_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "vllm_fl.compilation.musa_graph_exec_cache"
            and level == 0
            and fromlist
            and set(fromlist) <= set(vars(cache_api))
        ):
            return cache_api
        if name in {"vllm", "torch", "torch_musa", "vllm_fl"} or name.startswith(
            ("vllm.", "torch.", "torch_musa.", "vllm_fl.")
        ):
            raise ImportError("CPU contract does not import target modules")
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace = {
        "__name__": __name__,
        "__builtins__": dict(vars(builtins), __import__=deny_target_import),
        "dataclasses": dataclasses,
        "weakref": weakref,
        "ExitStack": ExitStack,
        "nullcontext": nullcontext,
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
        "_weak_ref_musa_tensors": lambda value: value,
        "compilation_counter": state.counter,
    }
    exec(
        compile(
            ast.Module(body=[*helpers, *classes], type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    state.cpu_namespace = namespace
    state.cpu_classes = classes
    state.wrapper_type = namespace["MUSAGraphWrapper"]
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


def test_musa_fresh_interpiece_input_uses_original_runnable(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract, (1.0, 2.0))
    wrapper(captured)
    entry = wrapper.concrete_graph_entries["batch"]
    address = captured.data_ptr()
    contract.events.clear()
    incoming = _Tensor(contract, (11.0, 29.0))
    assert incoming.data_ptr() != address
    assert wrapper(incoming) is contract.output
    assert captured.payload.tolist() == [1.0, 2.0]
    assert incoming.payload.tolist() == [11.0, 29.0]
    assert captured.data_ptr() == address == entry.input_addresses[0]
    assert contract.event_names() == ["runnable"]
    assert contract.events[0][2] == (incoming,)
    assert entry.graph is wrapper.concrete_graph_entries["batch"].graph
    assert contract.counter.num_cudagraph_captured == 1


def test_musa_same_input_pointer_does_not_copy_or_recapture(contract):
    wrapper = contract.wrapper()
    argument = _Tensor(contract)
    wrapper(argument)
    contract.events.clear()
    wrapper(argument)
    assert contract.event_names() == ["replay"]


def test_musa_nested_keyword_inputs_fallback_with_unchanged_static_leaves(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper([captured], scale=3, options={"tag": "same"})
    contract.events.clear()
    incoming = _Tensor(contract, (8.0, 17.0))
    wrapper([incoming], scale=3, options={"tag": "same"})
    assert captured.payload.tolist() == [1.0, 2.0]
    assert incoming.payload.tolist() == [8.0, 17.0]
    assert contract.event_names() == ["runnable"]
    assert contract.events[0][2:] == (
        ([incoming],),
        {"scale": 3, "options": {"tag": "same"}},
    )


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


def test_repeated_tensor_fallback_preserves_original_alias_relationship(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper(captured, captured)
    incoming = _Tensor(contract, (7.0, 13.0))
    contract.events.clear()
    wrapper(incoming, incoming)
    assert contract.event_names() == ["runnable"]
    assert contract.events[0][2] == (incoming, incoming)
    assert captured.payload.tolist() == [1.0, 2.0]
    assert incoming.payload.tolist() == [7.0, 13.0]
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
    if device == "musa":
        assert wrapper.concrete_graph_entries["batch"].musa_input_leaves is None
    else:
        assert not hasattr(wrapper.concrete_graph_entries["batch"], "musa_input_leaves")
    contract.events.clear()
    wrapper(_Tensor(contract, (9.0, 9.0)))
    assert contract.event_names() == (
        ["synchronize", "replay"] if device == "npu" else ["replay"]
    )
    assert captured.payload.tolist() == [1.0, 2.0]


def test_stable_replay_error_preserves_exception_and_adds_no_fence(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper(captured)
    error = RuntimeError("native replay sentinel")
    contract.errors["replay"] = error
    contract.events.clear()
    with pytest.raises(RuntimeError) as raised:
        wrapper(captured)
    assert raised.value is error
    assert captured.payload.tolist() == [1.0, 2.0]
    assert contract.event_names() == ["replay"]


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
    contract.wrapper_type = namespace["MUSAGraphWrapper"]
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


@pytest.mark.parametrize("device", ["cuda", "npu", "musa", "gcu", "ptpu", "txda"])
def test_graph_factory_uses_only_the_explicit_cpu_cache_interface(contract, device):
    source = Path(__file__).resolve().parents[3] / "vllm_fl/compilation/graph.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    factory = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Graph"
    )
    native_graph = type("NativeGraph", (), {})
    contract.platform.device_type = device
    namespace = dict(contract.cpu_namespace)
    namespace["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(CUDAGraph=native_graph),
        npu=SimpleNamespace(NPUGraph=native_graph),
        musa=SimpleNamespace(MUSAGraph=native_graph),
        gcu=SimpleNamespace(GCUGraph=native_graph),
        ptpu=SimpleNamespace(PTPUGraph=native_graph),
    )
    exec(
        compile(ast.Module(body=[factory], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    assert namespace["Graph"].graph is (None if device == "txda" else native_graph)
    assert contract.cache_calls == (
        [("factory", native_graph)] if device == "musa" else []
    )


@pytest.mark.parametrize("device", ["musa", "cuda", "npu"])
def test_wrapper_cache_interface_preserves_native_capture_and_replay(contract, device):
    contract.platform.device_type = device
    wrapper = contract.wrapper()
    assert wrapper() is contract.output
    graph = wrapper.concrete_graph_entries["batch"].graph
    assert wrapper() is contract.output
    assert contract.cache_calls == (
        [("capture-enter", graph), ("capture-exit", graph), ("replay", graph)]
        if device == "musa"
        else []
    )
    assert contract.event_names().count("capture-enter") == 1
    assert contract.event_names().count("replay") == 1


@pytest.mark.parametrize("stage", ["cache-capture", "cache-replay"])
def test_cache_interface_errors_preserve_identity_and_do_not_run_native(
    contract, stage
):
    wrapper = contract.wrapper()
    if stage == "cache-replay":
        wrapper()
    contract.events.clear()
    error = RuntimeError(stage + " sentinel")
    contract.errors[stage] = error
    with pytest.raises(RuntimeError) as caught:
        wrapper()
    assert caught.value is error
    assert "capture-enter" not in contract.event_names()
    assert "replay" not in contract.event_names()
    assert contract.counter.num_cudagraph_captured == (stage == "cache-replay")


@pytest.mark.parametrize(
    "name,fromlist",
    [
        ("torch_musa", ("MUSAGraph",)),
        ("vllm_fl.compilation.musa_graph_exec", ("load_musa_graph_exec_ops",)),
        ("vllm_fl.compilation.musa_graph_exec_cache", ("_get_cache",)),
        ("vllm_fl.compilation.musa_graph_exec_cache", ()),
    ],
)
def test_cpu_contract_still_rejects_native_and_unapproved_imports(
    contract, name, fromlist
):
    importer = contract.cpu_namespace["__builtins__"]["__import__"]
    with pytest.raises(ImportError, match="does not import target"):
        importer(name, fromlist=fromlist)


def _borrow_payload(payload, *, byte_shift=0, nbytes=None):
    """Keep the caller's real owner alive whenever this descriptor is used."""
    source = payload.untyped_storage()
    storage = cpu_torch._C._construct_storage_from_data_pointer(
        source.data_ptr() + byte_shift,
        payload.device,
        source.nbytes() - byte_shift if nbytes is None else nbytes,
    )
    return cpu_torch.empty(0, dtype=payload.dtype).set_(
        storage, payload.storage_offset(), payload.size(), payload.stride()
    )


def test_borrowed_staging_self_overlap_rejected_before_copy(contract):
    owner = cpu_torch.arange(6, dtype=cpu_torch.float32)
    captured = _Tensor(contract, payload=owner[:2])
    incoming = _Tensor(
        contract, payload=_borrow_payload(captured.payload, byte_shift=4)
    )
    assert not cpu_torch._C._is_alias_of(captured.payload, incoming.payload)
    wrapper = contract.wrapper()
    wrapper(captured)
    contract.events.clear()
    with pytest.raises(ValueError, match="overlaps its destination"):
        wrapper(incoming)
    assert contract.events == []
    assert owner.tolist() == list(range(6))


def test_borrowed_source_source_overlap_rejected_before_copy(contract):
    first, second = _Tensor(contract), _Tensor(contract)
    wrapper = contract.wrapper()
    wrapper(first, second)
    owner = cpu_torch.tensor([7.0, 8.0])
    incoming_first = _Tensor(contract, payload=_borrow_payload(owner))
    incoming_second = _Tensor(contract, payload=_borrow_payload(owner))
    assert not cpu_torch._C._is_alias_of(
        incoming_first.payload, incoming_second.payload
    )
    contract.events.clear()
    with pytest.raises(ValueError, match="alias relationship"):
        wrapper(incoming_first, incoming_second)
    assert contract.events == []
    assert first.payload.tolist() == second.payload.tolist() == [1.0, 2.0]


def test_borrowed_cross_destination_overlap_rejected_before_copy(contract):
    first, second = _Tensor(contract), _Tensor(contract)
    wrapper = contract.wrapper()
    wrapper(first, second)
    incoming = _Tensor(contract, payload=_borrow_payload(second.payload))
    assert not cpu_torch._C._is_alias_of(incoming.payload, second.payload)
    contract.events.clear()
    with pytest.raises(ValueError, match="staging destination"):
        wrapper(incoming, _Tensor(contract, (8.0, 9.0)))
    assert contract.events == []


def test_borrowed_distinct_captured_aliases_rejected_before_copy(contract):
    owner = cpu_torch.tensor([1.0, 2.0])
    first = _Tensor(contract, payload=_borrow_payload(owner))
    second = _Tensor(contract, payload=_borrow_payload(owner))
    assert not cpu_torch._C._is_alias_of(first.payload, second.payload)
    wrapper = contract.wrapper()
    wrapper(first, second)
    incoming_owner = cpu_torch.tensor([8.0, 9.0])
    incoming_first = _Tensor(contract, payload=_borrow_payload(incoming_owner))
    incoming_second = _Tensor(contract, payload=_borrow_payload(incoming_owner))
    contract.events.clear()
    with pytest.raises(ValueError, match="another captured input"):
        wrapper(incoming_first, incoming_second)
    assert contract.events == []
    assert owner.tolist() == [1.0, 2.0]


def test_borrowed_same_address_replay_does_not_copy(contract):
    owner = cpu_torch.tensor([1.0, 2.0])
    captured = _Tensor(contract, payload=_borrow_payload(owner))
    wrapper = contract.wrapper()
    wrapper(captured)
    contract.events.clear()
    wrapper(_Tensor(contract, payload=owner))
    assert contract.event_names() == ["replay"]


def test_ordinary_external_input_remains_strongly_owned(contract):
    captured = _Tensor(contract)
    payload_ref = weakref.ref(captured.payload)
    storage_ref = captured.payload.untyped_storage()._weak_ref()
    wrapper = contract.wrapper()
    wrapper(captured)
    contract.events.clear()
    del captured
    import gc

    gc.collect()
    assert payload_ref() is not None
    assert not cpu_torch.UntypedStorage._expired(storage_ref)
    assert (
        wrapper.concrete_graph_entries["batch"].musa_input_leaves[0].payload
        is payload_ref()
    )
    wrapper.clear_graphs()
    gc.collect()
    assert payload_ref() is None
    assert cpu_torch.UntypedStorage._expired(storage_ref)
    cpu_torch.UntypedStorage._free_weak_ref(storage_ref)


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
def test_physical_alias_extension_is_musa_only(contract, device):
    owner = cpu_torch.ones(2)
    a = _Tensor(contract, payload=owner)
    b = _Tensor(contract, payload=_borrow_payload(owner))
    a.overrides["device"] = b.overrides["device"] = SimpleNamespace(
        type=device, index=0
    )
    assert not contract.wrapper_type._musa_tensors_alias(a, b)


def test_physical_alias_extension_respects_device_index(contract):
    owner = cpu_torch.ones(2)
    a = _Tensor(contract, payload=owner)
    b = _Tensor(contract, payload=_borrow_payload(owner))
    b.overrides["device"] = SimpleNamespace(type="musa", index=1)
    assert not contract.wrapper_type._musa_tensors_alias(a, b)


def _numeric_replay_wrapper(contract, function, replay):
    """Use the production wrapper and CPU kernels for captured native replay."""
    wrapper = contract.wrapper()
    calls = []

    def runnable(*args, **kwargs):
        calls.append((args, kwargs))
        return function(*args, **kwargs)

    wrapper.runnable = runnable

    def capture(*args, **kwargs):
        output = wrapper(*args, **kwargs)
        entry = wrapper.concrete_graph_entries["batch"]

        def native_replay():
            contract.events.append(("numeric-replay",))
            replay(entry, args, kwargs)

        entry.graph.replay = native_replay
        contract.events.clear()
        return output

    return wrapper, capture, calls


def test_fresh_mutating_input_keeps_caller_visible_state_and_output_alias(contract):
    def mutate(value):
        value.payload.add_(10)
        return value

    def replay(entry, args, kwargs):
        args[0].payload.add_(10)

    wrapper, capture, calls = _numeric_replay_wrapper(contract, mutate, replay)
    captured = _Tensor(contract, (1.0, 2.0))
    assert capture(captured) is captured
    assert captured.payload.tolist() == [11.0, 12.0]
    incoming = _Tensor(contract, (3.0, 4.0))
    output = wrapper(incoming)
    assert incoming.payload.tolist() == [13.0, 14.0]
    assert output is incoming
    assert cpu_torch._C._is_alias_of(output.payload, incoming.payload)
    assert captured.payload.tolist() == [11.0, 12.0]
    assert len(calls) == 2
    assert contract.events == []
    assert contract.counter.num_cudagraph_captured == 1


def test_fresh_mutating_view_output_aliases_incoming_not_capture(contract):
    def mutate(value):
        value.payload.mul_(2)
        return _Tensor(contract, payload=value.payload.view(2))

    def replay(entry, args, kwargs):
        args[0].payload.mul_(2)

    wrapper, capture, calls = _numeric_replay_wrapper(contract, mutate, replay)
    captured = _Tensor(contract, (1.0, 2.0))
    captured_output = capture(captured)
    incoming = _Tensor(contract, (3.0, 4.0))
    output = wrapper(incoming)
    assert incoming.payload.tolist() == output.payload.tolist() == [6.0, 8.0]
    assert cpu_torch._C._is_alias_of(output.payload, incoming.payload)
    assert not cpu_torch._C._is_alias_of(output.payload, captured_output.payload)
    assert captured.payload.tolist() == [2.0, 4.0]
    assert len(calls) == 2
    assert contract.events == []


@pytest.mark.parametrize("output_kind", ["identity", "view", "nested"])
def test_fresh_readonly_alias_output_preserves_caller_allocation(contract, output_kind):
    def readonly(value):
        if output_kind == "identity":
            return value
        view = _Tensor(contract, payload=value.payload.view(2))
        return view if output_kind == "view" else {"result": [view, value]}

    wrapper, capture, calls = _numeric_replay_wrapper(
        contract, readonly, lambda entry, args, kwargs: None
    )
    captured = _Tensor(contract)
    capture(captured)
    incoming = _Tensor(contract, (7.0, 9.0))
    output = wrapper(incoming)
    result = output["result"][0] if output_kind == "nested" else output
    assert result.payload.tolist() == incoming.payload.tolist() == [7.0, 9.0]
    assert cpu_torch._C._is_alias_of(result.payload, incoming.payload)
    assert not cpu_torch._C._is_alias_of(result.payload, captured.payload)
    if output_kind == "identity":
        assert output is incoming
    if output_kind == "nested":
        assert output["result"][1] is incoming
    assert captured.payload.tolist() == [1.0, 2.0]
    assert len(calls) == 2
    assert contract.events == []


def test_fresh_readonly_fresh_output_uses_original_callable(contract):
    def readonly(value):
        return _Tensor(contract, payload=value.payload + 10)

    def replay(entry, args, kwargs):
        entry.output.payload.copy_(args[0].payload + 10)

    wrapper, capture, calls = _numeric_replay_wrapper(contract, readonly, replay)
    captured = _Tensor(contract)
    first_output = capture(captured)
    incoming = _Tensor(contract, (3.0, 4.0))
    output = wrapper(incoming)
    assert output.payload.tolist() == [13.0, 14.0]
    assert incoming.payload.tolist() == [3.0, 4.0]
    assert first_output.payload.tolist() == [11.0, 12.0]
    assert not cpu_torch._C._is_alias_of(output.payload, incoming.payload)
    assert len(calls) == 2
    assert contract.events == []


def test_unknown_callable_cannot_infer_mutation_safety_from_capture(contract):
    class UnknownCallable:
        def __init__(self):
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            if self.calls == 1:
                return _Tensor(contract, payload=value.payload + 10)
            value.payload.add_(10)
            return value

    function = UnknownCallable()

    def replay(entry, args, kwargs):
        entry.output.payload.copy_(args[0].payload + 10)

    wrapper, capture, calls = _numeric_replay_wrapper(contract, function, replay)
    captured = _Tensor(contract)
    capture(captured)
    incoming = _Tensor(contract, (3.0, 4.0))
    output = wrapper(incoming)
    assert function.calls == len(calls) == 2
    assert output is incoming
    assert incoming.payload.tolist() == [13.0, 14.0]
    assert captured.payload.tolist() == [1.0, 2.0]
    assert contract.events == []


def test_stable_mutating_input_still_uses_native_graph_replay(contract):
    def mutate(value):
        value.payload.add_(10)
        return value

    def replay(entry, args, kwargs):
        args[0].payload.add_(10)

    wrapper, capture, calls = _numeric_replay_wrapper(contract, mutate, replay)
    value = _Tensor(contract)
    assert capture(value) is value
    assert value.payload.tolist() == [11.0, 12.0]
    assert wrapper(value) is value
    assert value.payload.tolist() == [21.0, 22.0]
    assert len(calls) == 1
    assert contract.event_names() == ["numeric-replay"]
    assert contract.counter.num_cudagraph_captured == 1


def test_fresh_mutation_fallback_does_not_invalidate_stable_graph(contract):
    def mutate(value):
        value.payload.add_(10)
        return value

    def replay(entry, args, kwargs):
        args[0].payload.add_(10)

    wrapper, capture, calls = _numeric_replay_wrapper(contract, mutate, replay)
    captured = _Tensor(contract)
    capture(captured)
    incoming = _Tensor(contract, (3.0, 4.0))
    assert wrapper(incoming) is incoming
    assert incoming.payload.tolist() == [13.0, 14.0]
    assert wrapper(captured) is captured
    assert captured.payload.tolist() == [21.0, 22.0]
    assert incoming.payload.tolist() == [13.0, 14.0]
    assert len(calls) == 2
    assert contract.event_names() == ["numeric-replay"]


def test_fresh_mutation_fallback_preserves_exception_and_partial_state(contract):
    error = RuntimeError("runnable mutation sentinel")
    failed = False

    def mutate(value):
        value.payload.add_(10)
        if failed:
            raise error
        return value

    wrapper, capture, calls = _numeric_replay_wrapper(
        contract, mutate, lambda entry, args, kwargs: args[0].payload.add_(10)
    )
    captured = _Tensor(contract)
    capture(captured)
    failed = True
    incoming = _Tensor(contract, (3.0, 4.0))
    with pytest.raises(RuntimeError) as actual:
        wrapper(incoming)
    assert actual.value is error
    assert incoming.payload.tolist() == [13.0, 14.0]
    assert captured.payload.tolist() == [11.0, 12.0]
    assert len(calls) == 2
    assert contract.events == []


def test_fresh_mutation_fallback_retains_nested_argument_identity(contract):
    def mutate(values, *, options):
        values[0].payload.add_(options["increment"])
        return values[0]

    wrapper, capture, calls = _numeric_replay_wrapper(
        contract,
        mutate,
        lambda entry, args, kwargs: args[0][0].payload.add_(
            kwargs["options"]["increment"]
        ),
    )
    captured = _Tensor(contract)
    capture([captured], options={"increment": 10})
    incoming = _Tensor(contract, (3.0, 4.0))
    values = [incoming]
    options = {"increment": 10}
    assert wrapper(values, options=options) is incoming
    assert incoming.payload.tolist() == [13.0, 14.0]
    assert calls[-1][0][0] is values
    assert calls[-1][1]["options"] is options
    assert captured.payload.tolist() == [11.0, 12.0]
    assert contract.events == []


@pytest.mark.parametrize(
    "device,mode",
    [("cuda", _Mode.PIECEWISE), ("npu", _Mode.PIECEWISE), ("musa", _Mode.FULL)],
)
def test_non_piecewise_musa_mutating_replay_policy_is_unchanged(contract, device, mode):
    contract.platform.device_type = device
    contract.context.cudagraph_runtime_mode = mode

    def mutate(value):
        value.payload.add_(10)
        return value

    wrapper = contract.wrapper_type(
        mutate,
        SimpleNamespace(compilation_config=object()),
        mode,
        contract.options_type(debug_log_enable=False, weak_ref_output=False),
    )
    captured = _Tensor(contract)
    assert wrapper(captured) is captured
    entry = wrapper.concrete_graph_entries["batch"]
    if device == "musa":
        assert entry.musa_input_leaves is None
    else:
        assert not hasattr(entry, "musa_input_leaves")
    entry.graph.replay = lambda: captured.payload.add_(10)
    incoming = _Tensor(contract, (3.0, 4.0))
    assert wrapper(incoming) is captured
    assert incoming.payload.tolist() == [3.0, 4.0]
    assert captured.payload.tolist() == [21.0, 22.0]


@pytest.mark.parametrize("count", [1, 16, 32])
def test_stable_inputs_skip_pairwise_alias_and_storage_queries(
    contract, monkeypatch, count
):
    wrapper = contract.wrapper()
    inputs = [_Tensor(contract) for _ in range(count)]
    wrapper(*inputs)
    checks, storage_queries = [], []
    original_alias = wrapper._musa_tensors_alias
    original_storage = _Tensor.untyped_storage

    def alias(left, right):
        checks.append((left, right))
        return original_alias(left, right)

    def storage(value):
        storage_queries.append(value)
        return original_storage(value)

    monkeypatch.setattr(
        contract.wrapper_type, "_musa_tensors_alias", staticmethod(alias)
    )
    monkeypatch.setattr(_Tensor, "untyped_storage", storage)
    contract.events.clear()
    for _ in range(3):
        assert wrapper(*inputs) is contract.output
    assert checks == storage_queries == []
    assert contract.event_names() == ["replay"] * 3
    assert contract.counter.num_cudagraph_captured == 1


def test_validated_stable_descriptors_skip_the_next_alias_scan(contract, monkeypatch):
    wrapper = contract.wrapper()
    captured = [_Tensor(contract), _Tensor(contract)]
    wrapper(*captured)
    incoming = [_Tensor(contract, payload=x.payload.view(2)) for x in captured]
    checks = []
    original_alias = wrapper._musa_tensors_alias

    def alias(left, right):
        checks.append((left, right))
        return original_alias(left, right)

    monkeypatch.setattr(
        contract.wrapper_type, "_musa_tensors_alias", staticmethod(alias)
    )
    contract.events.clear()
    assert wrapper(*incoming) is contract.output
    assert checks
    checks.clear()
    assert wrapper(*incoming) is contract.output
    assert checks == []
    assert contract.event_names() == ["replay", "replay"]


def test_stable_fast_path_checks_captured_addresses(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    owner = captured.payload
    wrapper(captured)
    captured.payload = owner.clone()
    contract.events.clear()
    with pytest.raises(ValueError, match="Captured MUSA graph input address changed"):
        wrapper(captured)
    assert contract.events == []
    assert owner.tolist() == captured.payload.tolist() == [1.0, 2.0]


def test_stable_fast_path_checks_metadata_and_static_arguments(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper(captured, scale=2)
    captured.overrides["shape"] = (1, 2)
    contract.events.clear()
    with pytest.raises(ValueError, match="Captured MUSA graph input metadata"):
        wrapper(captured, scale=2)
    captured.overrides.clear()
    with pytest.raises(ValueError, match="non-tensor input changed"):
        wrapper(captured, scale=3)
    assert contract.events == []


def test_validated_descriptor_cache_does_not_retain_new_owners(contract):
    wrapper = contract.wrapper()
    captured = _Tensor(contract)
    wrapper(captured)
    incoming = _Tensor(contract, payload=captured.payload.view(2))
    wrapper(incoming)
    reference = weakref.ref(incoming)
    del incoming
    gc.collect()
    assert reference() is None
    assert wrapper(captured) is contract.output


def test_nonresident_graph_runs_original_inputs_without_preparing_capture(
    contract, monkeypatch
):
    def mutate(value):
        value.payload.add_(10)
        return value

    wrapper, capture, calls = _numeric_replay_wrapper(
        contract, mutate, lambda entry, args, kwargs: args[0].payload.add_(10)
    )
    captured = _Tensor(contract)
    assert capture(captured) is captured
    entry = wrapper.concrete_graph_entries["batch"]
    contract.nonresident_graph_ids.add(id(entry.graph))
    incoming = _Tensor(contract, (3.0, 4.0))

    def unexpected_validation(*args):
        raise AssertionError("nonresident graph must not prepare captured inputs")

    with monkeypatch.context() as patcher:
        patcher.setattr(wrapper, "_stage_musa_inputs", unexpected_validation)
        assert wrapper(incoming) is incoming
    assert incoming.payload.tolist() == [13.0, 14.0]
    assert captured.payload.tolist() == [11.0, 12.0]
    assert contract.events == []
    assert len(calls) == 2
    contract.nonresident_graph_ids.clear()
    assert wrapper(captured) is captured
    assert captured.payload.tolist() == [21.0, 22.0]
    assert contract.event_names() == ["numeric-replay"]
    assert contract.counter.num_cudagraph_captured == 1


def test_nonresident_fallback_keeps_runnable_exception_and_mutation(contract):
    error = RuntimeError("cold runnable sentinel")
    failed = False

    def mutate(value):
        value.payload.add_(10)
        if failed:
            raise error
        return value

    wrapper, capture, calls = _numeric_replay_wrapper(
        contract, mutate, lambda entry, args, kwargs: args[0].payload.add_(10)
    )
    captured = _Tensor(contract)
    capture(captured)
    entry = wrapper.concrete_graph_entries["batch"]
    contract.nonresident_graph_ids.add(id(entry.graph))
    failed = True
    incoming = _Tensor(contract, (3.0, 4.0))
    with pytest.raises(RuntimeError) as caught:
        wrapper(incoming)
    assert caught.value is error
    assert incoming.payload.tolist() == [13.0, 14.0]
    assert captured.payload.tolist() == [11.0, 12.0]
    assert len(calls) == 2
    assert contract.events == []


def test_production_cache_and_wrapper_preserve_cold_and_hot_semantics(
    contract, monkeypatch
):
    support_path = Path(__file__).with_name("test_musa_graph_exec_cache.py")
    spec = importlib.util.spec_from_file_location(
        "_musa_cache_wrapper_support", support_path
    )
    support = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, support)
    spec.loader.exec_module(support)
    cache, device, _, _ = support._fixture(cap=1)
    namespace = contract.cpu_namespace
    original_import = namespace["__builtins__"]["__import__"]
    cache_api = SimpleNamespace(
        musa_graph_class=contract.cache_graph_class,
        musa_graph_capture=cache.capture,
        musa_graph_replay=cache.replay,
    )

    def import_cache(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_fl.compilation.musa_graph_exec_cache" and fromlist:
            assert set(fromlist) <= set(vars(cache_api))
            return cache_api
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(namespace["__builtins__"], "__import__", import_cache)
    monkeypatch.setattr(namespace["Graph"], "graph", lambda: support._Graph(device))
    original_capture = contract.platform.torch_device_fn.graph

    @contextmanager
    def native_capture(graph, *, pool, stream):
        with original_capture(graph, pool=pool, stream=stream):
            yield
        graph.capture()

    monkeypatch.setattr(contract.platform.torch_device_fn, "graph", native_capture)

    def mutate(value):
        value.payload.add_(10)
        return value

    wrapper = contract.wrapper()
    wrapper.runnable = mutate
    captured = _Tensor(contract)
    assert wrapper(captured) is captured
    cold = wrapper.concrete_graph_entries["batch"]
    contract.context.batch_descriptor = "hot"
    hot_input = _Tensor(contract, (5.0, 6.0))
    assert wrapper(hot_input) is hot_input
    hot = wrapper.concrete_graph_entries["hot"]
    assert not cold.graph.executable and hot.graph.executable
    assert len(cache._live) == 1
    native_replay = hot.graph.replay

    def replay_hot():
        native_replay()
        hot_input.payload.add_(10)

    hot.graph.replay = replay_hot
    contract.context.batch_descriptor = "batch"
    incoming = _Tensor(contract, (3.0, 4.0))
    before = list(device.calls)
    assert wrapper(incoming) is incoming
    assert incoming.payload.tolist() == [13.0, 14.0]
    assert captured.payload.tolist() == [11.0, 12.0]
    assert device.calls == before
    assert not cold.graph.executable and hot.graph.executable
    contract.context.batch_descriptor = "hot"
    assert wrapper(hot_input) is hot_input
    assert hot_input.payload.tolist() == [25.0, 26.0]
    assert device.calls[len(before) :] == [("native_replay", 0)]
    assert len(cache._live) == 1 and not cache._replay_reserved


@pytest.mark.parametrize("output_kind", ["identity", "view", "nested"])
def test_fresh_empty_inputs_preserve_output_identity_and_storage(contract, output_kind):
    def readonly(value):
        if output_kind == "identity":
            return value
        view = _Tensor(contract, payload=value.payload.view(0))
        return view if output_kind == "view" else {"result": [view, value]}

    wrapper, capture, calls = _numeric_replay_wrapper(
        contract, readonly, lambda entry, args, kwargs: None
    )
    captured = _Tensor(contract, ())
    capture(captured)
    incoming = _Tensor(contract, ())
    assert captured.data_ptr() == incoming.data_ptr() == 0
    assert not cpu_torch._C._is_alias_of(captured.payload, incoming.payload)
    output = wrapper(incoming)
    result = output["result"][0] if output_kind == "nested" else output
    assert cpu_torch._C._is_alias_of(result.payload, incoming.payload)
    assert not cpu_torch._C._is_alias_of(result.payload, captured.payload)
    if output_kind == "identity":
        assert output is incoming
    if output_kind == "nested":
        assert output["result"][1] is incoming
    assert len(calls) == 2
    assert contract.events == []


def test_original_empty_input_keeps_native_replay(contract):
    wrapper, capture, calls = _numeric_replay_wrapper(
        contract, lambda value: value, lambda entry, args, kwargs: None
    )
    captured = _Tensor(contract, ())
    assert capture(captured) is captured
    assert wrapper(captured) is captured
    assert len(calls) == 1
    assert contract.event_names() == ["numeric-replay"]


@pytest.mark.parametrize("device", ["cuda", "musa"])
def test_public_factory_selects_backend_and_preserves_shared_registry(
    contract, monkeypatch, device
):
    namespace = contract.cpu_namespace
    target = "vllm_fl.compilation.musa_graph"
    module = ModuleType(target)
    module.MUSAGraphWrapper = namespace["MUSAGraphWrapper"]
    monkeypatch.setitem(sys.modules, target, module)
    original_import = namespace["__builtins__"]["__import__"]

    def import_backend(name, globals=None, locals=None, fromlist=(), level=0):
        if name == target and tuple(fromlist) == ("MUSAGraphWrapper",):
            return builtins.__import__(name, globals, locals, fromlist, level)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(namespace["__builtins__"], "__import__", import_backend)
    contract.platform.device_type = device
    options = contract.options_type(debug_log_enable=False, weak_ref_output=False)
    wrapper = namespace["GraphWrapper"](
        contract.runnable,
        SimpleNamespace(compilation_config=object()),
        _Mode.PIECEWISE,
        options,
    )
    expected = namespace["MUSAGraphWrapper" if device == "musa" else "GraphWrapper"]
    assert type(wrapper) is expected
    assert wrapper.cudagraph_options is options
    assert (
        namespace["GraphWrapper"]._all_instances
        is namespace["MUSAGraphWrapper"]._all_instances
    )
    assert wrapper in namespace["GraphWrapper"]._all_instances
    wrapper.concrete_graph_entries["sentinel"] = object()
    namespace["GraphWrapper"].clear_all_graphs()
    assert wrapper.concrete_graph_entries == {}

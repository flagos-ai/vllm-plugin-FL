# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU controls for MUSA PIECEWISE input staging and replay admission.

The real ``MUSAGraphWrapper._stage_musa_inputs``/``_replay_graph`` run against
the production source; the shared graph wrapper is a small CPU double so no
native capture ever happens.
"""

import dataclasses
import importlib.util
import sys
from contextlib import nullcontext
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _repo_root() -> Path:
    import os

    candidates = [Path("/tmp/flsrc")]
    if os.environ.get("VLLM_FL_ROOT"):
        candidates.insert(0, Path(os.environ["VLLM_FL_ROOT"]))
    for base in [*candidates, Path.cwd(), *Path(__file__).resolve().parents]:
        if not base.is_dir():
            continue
        if (base / "vllm_fl/dispatch/backends/vendor/musa/graph.py").is_file():
            return base
    raise RuntimeError("cannot locate vllm_fl source tree")


ROOT = _repo_root()


def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install(monkeypatch, mapping):
    for name, obj in mapping.items():
        monkeypatch.setitem(sys.modules, name, obj)
    for name in mapping:
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            full = ".".join(parts[:index])
            if full not in sys.modules:
                monkeypatch.setitem(sys.modules, full, ModuleType(full))
            if index > 1:
                parent = sys.modules[".".join(parts[:index - 1])]
                monkeypatch.setattr(
                    parent, parts[index - 1], sys.modules[full], raising=False
                )


STRIDED = object()


class FakeStorage:
    def __init__(self, ptr, nbytes, cdata):
        self._ptr = ptr
        self._nbytes = nbytes
        self._cdata = cdata

    def data_ptr(self):
        return self._ptr

    def nbytes(self):
        return self._nbytes


class FakeTensor:
    def __init__(
        self,
        *,
        ptr=100,
        nbytes=16,
        cdata=None,
        storage_ptr=None,
        device_type="musa",
        layout=STRIDED,
        quantized=False,
        requires_grad=False,
        conj=False,
        neg=False,
        shape=(2, 2),
        strides=(2, 1),
        dtype="float32",
        offset=0,
    ):
        self._data_ptr = ptr
        self._storage = FakeStorage(
            ptr if storage_ptr is None else storage_ptr,
            nbytes,
            cdata if cdata is not None else object(),
        )
        self.device = SimpleNamespace(type=device_type, index=0)
        self.layout = layout
        self.is_quantized = quantized
        self.requires_grad = requires_grad
        self._conj = conj
        self._neg = neg
        self._shape = shape
        self._strides = strides
        self.dtype = dtype
        self._offset = offset

    @property
    def shape(self):
        return self._shape

    def data_ptr(self):
        return self._data_ptr

    def untyped_storage(self):
        return self._storage

    def is_conj(self):
        return self._conj

    def is_neg(self):
        return self._neg

    def size(self):
        return self._shape

    def stride(self):
        return self._strides

    def storage_offset(self):
        return self._offset


class FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def _tree_flatten(tree):
    leaves = []

    def walk(node):
        if isinstance(node, dict):
            return ("dict", tuple((key, walk(value)) for key, value in node.items()))
        if isinstance(node, list):
            return ("list", tuple(walk(value) for value in node))
        if isinstance(node, tuple):
            return ("tuple", tuple(walk(value) for value in node))
        leaves.append(node)
        return ("leaf",)

    spec = walk(tree)
    return leaves, spec


@dataclasses.dataclass
class _GraphEntry:
    batch_descriptor: object
    graph: object = None
    output: object = None
    input_addresses: object = None


class _GraphWrapper:
    def __init__(
        self,
        runnable,
        vllm_config,
        runtime_mode,
        cudagraph_options=None,
        graph_runtime=None,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.cudagraph_options = cudagraph_options
        self.graph_runtime = graph_runtime
        self.concrete_graph_entries = {}
        self.validated = 0

    def clear_graphs(self):
        self.concrete_graph_entries.clear()

    def _validate_replay_addresses(self, entry, args):
        self.validated += 1

    def _capture_graph(self, entry, args, kwargs):
        output = self.runnable(*args, **kwargs)
        entry.graph = FakeGraph()
        entry.output = output
        return output


def load_graph(monkeypatch):
    logger = SimpleNamespace(records=[])
    logger.info = lambda *args, **kwargs: logger.records.append(args)

    torch_stub = ModuleType("torch")
    torch_stub.Tensor = FakeTensor
    torch_stub.strided = STRIDED
    torch_stub._C = SimpleNamespace(
        _is_alias_of=MagicMock(side_effect=lambda left, right: left is right),
        _construct_storage_from_data_pointer=MagicMock(
            side_effect=lambda ptr, dev, nbytes: ("storage", ptr, nbytes)
        ),
    )
    torch_stub.musa = SimpleNamespace(
        MUSAGraph=FakeGraph,
        current_device=lambda: 0,
        device=lambda ordinal: nullcontext(),
    )
    utils = _module("torch.utils")
    pytree = _module("torch.utils._pytree", tree_flatten=_tree_flatten)
    utils._pytree = pytree
    torch_stub.utils = utils

    class IntermediateTensors:
        def __init__(self, tensors):
            self.tensors = tensors

    _install(
        monkeypatch,
        {
            "torch": torch_stub,
            "torch.utils": utils,
            "torch.utils._pytree": pytree,
            "vllm": _module("vllm"),
            "vllm.config": _module("vllm.config", CUDAGraphMode=Mode),
            "vllm.logger": _module("vllm.logger", init_logger=lambda name: logger),
            "vllm.sequence": _module(
                "vllm.sequence", IntermediateTensors=IntermediateTensors
            ),
            "vllm_fl": _module("vllm_fl"),
            "vllm_fl.compilation": _module("vllm_fl.compilation"),
            "vllm_fl.compilation.graph": _module(
                "vllm_fl.compilation.graph",
                GraphEntry=_GraphEntry,
                GraphWrapper=_GraphWrapper,
            ),
        },
    )

    source = ROOT / "vllm_fl/dispatch/backends/vendor/musa/graph.py"
    spec = importlib.util.spec_from_file_location("w26_musa_graph_staging", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, torch=torch_stub, logger=logger)


@pytest.fixture
def graph_env(monkeypatch):
    return load_graph(monkeypatch)


def _wrapper(env, mode=Mode.PIECEWISE):
    return env.module.MUSAGraphWrapper(
        MagicMock(return_value="eager"),
        SimpleNamespace(compilation_config=object()),
        mode,
    )


def _entry(env, descriptor="d"):
    return env.module.MUSAGraphEntry(batch_descriptor=descriptor)


def _capture_inputs(env, wrapper, entry, args, kwargs):
    wrapper._finish_capture_inputs(entry, wrapper._capture_inputs(entry, args, kwargs))


def test_stage_returns_true_for_identical_inputs(graph_env):
    wrapper = _wrapper(graph_env)
    tensor = FakeTensor(ptr=100)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (tensor,), {})
    assert wrapper._stage_musa_inputs(entry, (tensor,), {}) is True


def test_stage_rejects_input_count_change(graph_env):
    wrapper = _wrapper(graph_env)
    first, second = FakeTensor(ptr=100), FakeTensor(ptr=200)
    entry = _entry(graph_env)
    leaves, spec, metadata, addresses = wrapper._capture_inputs(
        entry, (first, second), {}
    )
    entry.musa_input_leaves = leaves[:1]
    entry.musa_input_spec = spec
    entry.musa_input_metadata = metadata[:1]
    entry.musa_input_addresses = addresses[:1]
    entry.musa_last_input_tensor_refs = [None]
    with pytest.raises(ValueError, match="input count changed"):
        wrapper._stage_musa_inputs(entry, (first, second), {})


def test_stage_rejects_tensor_type_change(graph_env):
    wrapper = _wrapper(graph_env)
    tensor = FakeTensor(ptr=100)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (tensor,), {})
    with pytest.raises(ValueError, match="tensor input type changed"):
        wrapper._stage_musa_inputs(entry, ("not-a-tensor",), {})


def test_stage_rejects_captured_metadata_mutation(graph_env):
    wrapper = _wrapper(graph_env)
    tensor = FakeTensor(ptr=100)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (tensor,), {})
    tensor.dtype = "float16"
    with pytest.raises(ValueError, match="Captured MUSA graph input metadata"):
        wrapper._stage_musa_inputs(entry, (tensor,), {})


def test_stage_rejects_incoming_metadata_change(graph_env):
    wrapper = _wrapper(graph_env)
    tensor = FakeTensor(ptr=100)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (tensor,), {})
    incoming = FakeTensor(ptr=100, shape=(4,))
    with pytest.raises(ValueError, match="input tensor metadata changed"):
        wrapper._stage_musa_inputs(entry, (incoming,), {})


def test_stage_rejects_captured_address_change(graph_env):
    wrapper = _wrapper(graph_env)
    tensor = FakeTensor(ptr=100)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (tensor,), {})
    tensor._data_ptr = 999
    with pytest.raises(ValueError, match="Captured MUSA graph input address"):
        wrapper._stage_musa_inputs(entry, (tensor,), {})


def test_stage_accepts_equal_non_tensor_and_rejects_changes(graph_env):
    wrapper = _wrapper(graph_env)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (), {"k": 1})
    assert wrapper._stage_musa_inputs(entry, (), {"k": 1}) is True
    with pytest.raises(ValueError, match="non-tensor input changed"):
        wrapper._stage_musa_inputs(entry, (), {"k": 2})


def test_stage_runs_eager_for_distinct_empty_tensor(graph_env):
    wrapper = _wrapper(graph_env)
    captured = FakeTensor(ptr=0, nbytes=0)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (captured,), {})
    incoming = FakeTensor(ptr=0, nbytes=0)
    assert wrapper._stage_musa_inputs(entry, (incoming,), {}) is False


def test_stage_runs_eager_when_a_copy_is_required(graph_env):
    wrapper = _wrapper(graph_env)
    captured = FakeTensor(ptr=100, nbytes=16, storage_ptr=100)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (captured,), {})
    incoming = FakeTensor(ptr=500, nbytes=16, storage_ptr=500)
    assert wrapper._stage_musa_inputs(entry, (incoming,), {}) is False


def test_stage_rejects_changed_alias_relationship(graph_env):
    wrapper = _wrapper(graph_env)
    left = FakeTensor(ptr=100, nbytes=32, storage_ptr=100)
    right = FakeTensor(ptr=116, nbytes=32, storage_ptr=116)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (left, right), {})

    incoming_left = FakeTensor(ptr=500, nbytes=32, storage_ptr=500)
    incoming_right = FakeTensor(ptr=600, nbytes=32, storage_ptr=600)
    with pytest.raises(ValueError, match="alias relationship changed"):
        wrapper._stage_musa_inputs(entry, (incoming_left, incoming_right), {})


def test_stage_rejects_copy_aliasing_another_captured_input(graph_env):
    wrapper = _wrapper(graph_env)
    left = FakeTensor(ptr=100, nbytes=32, storage_ptr=100)
    right = FakeTensor(ptr=116, nbytes=32, storage_ptr=116)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (left, right), {})

    incoming_left = FakeTensor(ptr=500, nbytes=32, storage_ptr=500)
    incoming_right = FakeTensor(ptr=516, nbytes=32, storage_ptr=516)
    with pytest.raises(ValueError, match="aliases another captured input"):
        wrapper._stage_musa_inputs(entry, (incoming_left, incoming_right), {})


def test_stage_rejects_repeated_input_alias_change(graph_env):
    wrapper = _wrapper(graph_env)
    shared = FakeTensor(ptr=100)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (shared, shared), {})
    first = FakeTensor(ptr=500)
    second = FakeTensor(ptr=600)
    with pytest.raises(ValueError, match="repeated MUSA graph input"):
        wrapper._stage_musa_inputs(entry, (first, second), {})


def test_stage_remembers_new_descriptor_for_same_buffer(graph_env):
    wrapper = _wrapper(graph_env)
    captured = FakeTensor(ptr=100)
    entry = _entry(graph_env)
    _capture_inputs(graph_env, wrapper, entry, (captured,), {})
    incoming = FakeTensor(ptr=100)
    assert incoming is not captured
    assert wrapper._stage_musa_inputs(entry, (incoming,), {}) is True
    assert entry.musa_last_input_tensor_refs[0]() is incoming


def test_replay_graph_syncs_offloader_before_native_replay(graph_env, monkeypatch):
    wrapper = _wrapper(graph_env, mode=Mode.FULL)
    graph = FakeGraph()
    entry = graph_env.module.MUSAGraphEntry(
        batch_descriptor="d",
        graph=graph,
        output="captured",
        musa_capture_device=0,
    )
    offloader = SimpleNamespace(sync_prev_onload=MagicMock())
    _install(
        monkeypatch,
        {
            "vllm.model_executor": _module("vllm.model_executor"),
            "vllm.model_executor.offloader": _module("vllm.model_executor.offloader"),
            "vllm.model_executor.offloader.base": _module(
                "vllm.model_executor.offloader.base",
                get_offloader=lambda: offloader,
            ),
        },
    )
    assert wrapper._replay_graph(entry, (), {}) == "captured"
    offloader.sync_prev_onload.assert_called_once_with()
    assert graph.replays == 1
    assert wrapper.validated == 1

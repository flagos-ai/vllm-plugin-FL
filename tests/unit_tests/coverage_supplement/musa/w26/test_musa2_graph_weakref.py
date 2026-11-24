# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU controls for the MUSA borrowed-output helper and storage-span aliasing.

The production ``graph.py`` is imported in place; ``torch`` and the vLLM
sibling modules are replaced by deterministic doubles so no MUSA device or
native graph is required.
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

    def _log(*args, **kwargs):
        logger.records.append(args)

    logger.info = _log

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
    spec = importlib.util.spec_from_file_location("w26_musa_graph_weakref", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module,
        torch=torch_stub,
        logger=logger,
        IntermediateTensors=IntermediateTensors,
    )


@pytest.fixture
def graph_env(monkeypatch):
    return load_graph(monkeypatch)


def _musa_constructor(monkeypatch, native):
    monkeypatch.setitem(
        sys.modules,
        "torch_musa",
        _module(
            "torch_musa",
            _MUSAC=SimpleNamespace(
                _construct_MUSA_Tensor_From_Storage_And_Metadata=native
            ),
        ),
    )


def test_weak_ref_passes_through_plain_and_foreign_values(graph_env):
    convert = graph_env.module._weak_ref_musa_tensors
    assert convert(5) == 5
    assert convert(None) is None
    assert convert("text") == "text"
    foreign = FakeTensor(device_type="cuda")
    assert convert(foreign) is foreign


@pytest.mark.parametrize(
    "options",
    [
        {"layout": "sparse"},
        {"quantized": True},
        {"requires_grad": True},
        {"conj": True},
        {"neg": True},
    ],
)
def test_weak_ref_rejects_non_inference_tensors(graph_env, options):
    with pytest.raises(ValueError, match="plain strided inference tensors"):
        graph_env.module._weak_ref_musa_tensors(FakeTensor(**options))


def test_weak_ref_returns_empty_storage_and_rejects_null_address(graph_env):
    convert = graph_env.module._weak_ref_musa_tensors
    empty = FakeTensor(ptr=0, nbytes=0)
    assert convert(empty) is empty
    with pytest.raises(ValueError, match="null address"):
        convert(FakeTensor(ptr=0, nbytes=16))


@pytest.mark.parametrize("musac", [None, SimpleNamespace()])
def test_weak_ref_requires_native_metadata_constructor(graph_env, monkeypatch, musac):
    torch_musa = _module("torch_musa")
    if musac is not None:
        torch_musa._MUSAC = musac
    monkeypatch.setitem(sys.modules, "torch_musa", torch_musa)
    with pytest.raises(RuntimeError, match="metadata tensor constructor"):
        graph_env.module._weak_ref_musa_tensors(FakeTensor())


def test_weak_ref_builds_storage_metadata_and_dedupes(graph_env, monkeypatch):
    native = MagicMock(return_value="native-tensor")
    _musa_constructor(monkeypatch, native)
    construct = graph_env.torch._C._construct_storage_from_data_pointer

    shared_storage_id = object()
    first = FakeTensor(
        ptr=100,
        nbytes=32,
        cdata=shared_storage_id,
        shape=(2, 2),
        strides=(2, 1),
        offset=1,
        dtype="float16",
    )
    second = FakeTensor(
        ptr=100,
        nbytes=32,
        cdata=shared_storage_id,
        shape=(2, 2),
        strides=(1, 2),
        offset=0,
    )

    # One traversal converts ``first`` once and reuses the shared storage for
    # the distinct ``second`` descriptor.
    converted = graph_env.module._weak_ref_musa_tensors(
        {"a": first, "b": first, "c": second}
    )
    assert converted == {
        "a": "native-tensor",
        "b": "native-tensor",
        "c": "native-tensor",
    }
    assert native.call_count == 2
    assert construct.call_count == 1
    assert construct.call_args.args == (100, first.device, 32)

    metadata = native.call_args_list[0].args[0]
    assert metadata["size"] == (2, 2)
    assert metadata["stride"] == (2, 1)
    assert metadata["storage_offset"] == 1
    assert metadata["device"] is first.device
    second_metadata = native.call_args_list[1].args[0]
    assert second_metadata["stride"] == (1, 2)
    assert native.call_args_list[1].args[1] == ("storage", 100, 32)

    # Each top-level invocation rebuilds its descriptor cache.
    assert graph_env.module._weak_ref_musa_tensors(first) == "native-tensor"
    assert native.call_count == 3
    assert construct.call_count == 2


def test_weak_ref_walks_containers_and_intermediate_tensors(graph_env, monkeypatch):
    native = MagicMock(side_effect=lambda metadata, storage: ("t", metadata["size"]))
    _musa_constructor(monkeypatch, native)
    first = FakeTensor(ptr=100, nbytes=16, shape=(2, 2))
    second = FakeTensor(ptr=300, nbytes=8, shape=(3,))
    intermediate = graph_env.IntermediateTensors({"h": second})
    result = graph_env.module._weak_ref_musa_tensors(
        {"list": [first, (None, first)], "inter": intermediate, "text": "keep"}
    )
    assert result["list"] == [("t", (2, 2)), (None, ("t", (2, 2)))]
    assert result["inter"].tensors == {"h": ("t", (3,))}
    assert result["text"] == "keep"
    # ``first`` appears twice but is converted once.
    assert native.call_count == 2


def test_musa_tensors_alias_uses_torch_and_device_guard(graph_env):
    alias = graph_env.module.MUSAGraphWrapper._musa_tensors_alias
    tensor = FakeTensor(ptr=100, nbytes=16, storage_ptr=100)
    assert alias(tensor, tensor) is True

    foreign = FakeTensor(ptr=100, nbytes=16, storage_ptr=100, device_type="cuda")
    assert alias(tensor, foreign) is False


def test_musa_tensors_alias_detects_storage_span_overlap(graph_env):
    alias = graph_env.module.MUSAGraphWrapper._musa_tensors_alias
    first = FakeTensor(ptr=100, nbytes=32, storage_ptr=100)
    overlapping = FakeTensor(ptr=116, nbytes=32, storage_ptr=116)
    disjoint = FakeTensor(ptr=500, nbytes=32, storage_ptr=500)
    empty = FakeTensor(ptr=100, nbytes=0, storage_ptr=100)

    assert alias(first, overlapping) is True
    assert alias(overlapping, first) is True
    assert alias(first, disjoint) is False
    assert alias(empty, first) is False

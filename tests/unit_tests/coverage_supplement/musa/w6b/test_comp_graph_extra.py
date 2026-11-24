# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU controls for the platform-agnostic compilation graph wrapper."""

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
        if (base / "vllm_fl/compilation/graph.py").is_file():
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
                setattr(parent, parts[index - 1], sys.modules[full])


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class _T:
    def __init__(self, address=1):
        self._address = address
        self.device = SimpleNamespace(type="musa", index=0)

    def data_ptr(self):
        return self._address


class FakeGraph:
    instances = []

    def __init__(self):
        self.replays = 0
        FakeGraph.instances.append(self)

    def replay(self):
        self.replays += 1


class _State:
    pass


def load_comp(monkeypatch, device="musa", logging_level="INFO", pool="pool"):
    state = _State()
    state.platform = SimpleNamespace(
        device_type=device,
        vendor_name=None,
        get_global_graph_pool=lambda: pool,
        graph_pool_handle=lambda: "handle",
        torch_device_fn=SimpleNamespace(
            graph=MagicMock(return_value=nullcontext()),
            current_stream=MagicMock(return_value="native-stream"),
            synchronize=MagicMock(),
        ),
    )
    state.context = SimpleNamespace(
        batch_descriptor="batch", cudagraph_runtime_mode=Mode.FULL
    )
    state.available = True
    state.validate = MagicMock()
    state.set_pool = MagicMock()
    state.counter = SimpleNamespace(num_cudagraph_captured=0)
    state.logger = SimpleNamespace(
        debug=MagicMock(), info=MagicMock(), warning=MagicMock()
    )
    state.upstream_weak_ref = MagicMock(side_effect=lambda value: value)
    state.iluvatar = MagicMock(side_effect=lambda value: ("iluvatar", value))

    torch_stub = ModuleType("torch")
    torch_stub.Tensor = _T
    torch_stub.strided = "strided"
    torch_stub._C = SimpleNamespace(
        _construct_storage_from_data_pointer=MagicMock(return_value="storage"),
        _set_conj=MagicMock(),
        _set_neg=MagicMock(),
    )
    torch_stub.accelerator = SimpleNamespace(empty_cache=lambda: "empty")
    torch_stub.cuda = SimpleNamespace(CUDAGraph=FakeGraph)
    torch_stub.npu = SimpleNamespace(NPUGraph=FakeGraph)
    torch_stub.musa = SimpleNamespace(MUSAGraph=FakeGraph)
    torch_stub.gcu = SimpleNamespace(GCUGraph=FakeGraph)
    torch_stub.ptpu = SimpleNamespace(PTPUGraph=FakeGraph)
    state.torch = torch_stub

    platform_fl = type("PlatformFL", (), {"empty_cache": staticmethod(lambda *a: None)})
    mapping = {
        "torch": torch_stub,
        "vllm": _module("vllm"),
        "vllm.envs": _module("vllm.envs", VLLM_LOGGING_LEVEL=logging_level),
        "vllm.compilation.counter": _module(
            "vllm.compilation.counter", compilation_counter=state.counter
        ),
        "vllm.compilation.monitor": _module(
            "vllm.compilation.monitor",
            validate_cudagraph_capturing_enabled=state.validate,
        ),
        "vllm.config": _module("vllm.config", CUDAGraphMode=Mode, VllmConfig=type("VllmConfig", (), {})),
        "vllm.distributed.device_communicators.pynccl_allocator": _module(
            "vllm.distributed.device_communicators.pynccl_allocator",
            set_graph_pool_id=state.set_pool,
        ),
        "vllm.forward_context": _module(
            "vllm.forward_context",
            BatchDescriptor=type("BatchDescriptor", (), {}),
            get_forward_context=lambda: state.context,
            is_forward_context_available=lambda: state.available,
        ),
        "vllm.logger": _module("vllm.logger", init_logger=lambda name: state.logger),
        "vllm.platforms": _module("vllm.platforms", current_platform=state.platform),
        "vllm.utils.torch_utils": _module(
            "vllm.utils.torch_utils",
            current_stream=MagicMock(return_value="vllm-stream"),
            weak_ref_tensors=state.upstream_weak_ref,
        ),
        "vllm.compilation.cuda_graph": _module(
            "vllm.compilation.cuda_graph", CUDAGraphStat=type("CUDAGraphStat", (), {})
        ),
        "vllm_fl": _module("vllm_fl"),
        "vllm_fl.platform": _module("vllm_fl.platform", PlatformFL=platform_fl),
        "vllm_fl.dispatch.backends.vendor.iluvatar.graph": _module(
            "vllm_fl.dispatch.backends.vendor.iluvatar.graph",
            _weak_ref_iluvatar_tensors=state.iluvatar,
        ),
    }
    _install(monkeypatch, mapping)

    class IntermediateTensors:
        def __init__(self, tensors):
            self.tensors = tensors

    monkeypatch.setitem(
        sys.modules,
        "vllm.sequence",
        _module("vllm.sequence", IntermediateTensors=IntermediateTensors),
    )
    monkeypatch.setattr(sys.modules["vllm"], "sequence", sys.modules["vllm.sequence"], raising=False)

    source = ROOT / "vllm_fl/compilation/graph.py"
    spec = importlib.util.spec_from_file_location("w6b_comp_graph", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["w6b_comp_graph"] = module
    spec.loader.exec_module(module)
    state.module = module
    state.IntermediateTensors = IntermediateTensors
    return state


@pytest.fixture
def comp(monkeypatch):
    FakeGraph.instances.clear()
    return load_comp(monkeypatch)


def _wrapper(state, runnable=None, mode=Mode.FULL, **options):
    if runnable is None:
        runnable = MagicMock(return_value="output")
    return state.module.GraphWrapper(
        runnable,
        SimpleNamespace(compilation_config=SimpleNamespace()),
        mode,
        state.module.GraphOptions(**options),
    )


def test_entry_and_options_defaults(comp):
    entry = comp.module.GraphEntry(batch_descriptor="d")
    assert entry.graph is None and entry.output is None
    assert entry.input_addresses is None
    options = comp.module.GraphOptions()
    assert options.debug_log_enable is True
    assert options.gc_disable is False
    assert options.weak_ref_output is True
    custom = comp.module.GraphOptions(False, True, False)
    assert (custom.debug_log_enable, custom.gc_disable, custom.weak_ref_output) == (
        False,
        True,
        False,
    )


@pytest.mark.parametrize(
    "device,expected",
    [
        ("cuda", FakeGraph),
        ("npu", FakeGraph),
        ("musa", FakeGraph),
        ("gcu", FakeGraph),
        ("ptpu", FakeGraph),
        ("txda", None),
    ],
)
def test_graph_factory_selects_native_class(monkeypatch, device, expected):
    state = load_comp(monkeypatch, device=device)
    assert state.module.Graph.graph is expected


def test_graph_factory_rejects_unknown_device(monkeypatch):
    with pytest.raises(NotImplementedError, match="not support graph"):
        load_comp(monkeypatch, device="cpu")


def test_wrapper_init_attributes_and_getattr(comp):
    runnable = MagicMock()
    runnable.custom = "value"
    wrapper = _wrapper(comp, runnable=runnable)
    assert wrapper.unwrap() is runnable
    assert wrapper.cudagraph_wrapper is wrapper
    assert wrapper.graph_pool == "pool"
    assert wrapper.custom == "value"
    wrapper.concrete_graph_entries["x"] = object()
    wrapper.clear_graphs()
    assert wrapper.concrete_graph_entries == {}


def test_wrapper_getattr_missing_attribute(comp):
    wrapper = _wrapper(comp, runnable=lambda: None)
    with pytest.raises(AttributeError):
        wrapper.absent


def test_wrapper_debug_getattr_message(monkeypatch):
    state = load_comp(monkeypatch, logging_level="DEBUG")
    wrapper = _wrapper(state, runnable=lambda: None)
    assert wrapper.is_debugging_mode is True
    with pytest.raises(AttributeError, match="cudagraph wrapper"):
        wrapper.absent


def test_wrapper_rejects_none_mode(comp):
    with pytest.raises(AssertionError):
        comp.module.GraphWrapper(
            MagicMock(),
            SimpleNamespace(compilation_config=SimpleNamespace()),
            Mode.NONE,
        )


def test_clear_all_graphs_visits_instances(comp):
    first = _wrapper(comp)
    second = _wrapper(comp)
    first.concrete_graph_entries["a"] = 1
    second.concrete_graph_entries["b"] = 2
    comp.module.GraphWrapper.clear_all_graphs()
    assert first.concrete_graph_entries == {}
    assert second.concrete_graph_entries == {}


@pytest.mark.parametrize("bypass", ["unavailable", "none", "mismatch"])
def test_call_bypasses_graph_path(comp, bypass):
    runnable = MagicMock(return_value="eager")
    wrapper = _wrapper(comp, runnable=runnable)
    if bypass == "unavailable":
        comp.available = False
    elif bypass == "none":
        comp.context.cudagraph_runtime_mode = Mode.NONE
    else:
        comp.context.cudagraph_runtime_mode = Mode.PIECEWISE
    tensor = _T(5)
    assert wrapper(tensor, key=2) == "eager"
    runnable.assert_called_once_with(tensor, key=2)
    assert wrapper.concrete_graph_entries == {}


def test_capture_then_replay(comp):
    runnable = MagicMock(return_value="captured-output")
    wrapper = _wrapper(comp, runnable=runnable, weak_ref_output=False)
    tensor = _T(11)
    output = wrapper(tensor, scale=3)
    assert output == "captured-output"
    entry = wrapper.concrete_graph_entries["batch"]
    assert entry.graph is FakeGraph.instances[-1]
    assert entry.output == "captured-output"
    assert entry.input_addresses == [11]
    assert comp.validate.call_count == 1
    comp.set_pool.assert_called_with("pool")
    graph_ctx = comp.platform.torch_device_fn.graph
    graph_ctx.assert_called_once_with(
        entry.graph, pool="pool", stream="native-stream"
    )
    assert comp.counter.num_cudagraph_captured == 1

    runnable.reset_mock()
    assert wrapper(tensor) == "captured-output"
    runnable.assert_not_called()
    assert entry.graph.replays == 1


def test_capture_without_pool_uses_handle(monkeypatch):
    state = load_comp(monkeypatch, pool=None)
    wrapper = _wrapper(state, weak_ref_output=False)
    wrapper(_T())
    state.set_pool.assert_called_once_with("handle")


def test_capture_weak_ref_output_routes(comp):
    seen = []
    comp.module.weak_ref_tensors = lambda value: seen.append(value) or ("weak", value)
    runnable = MagicMock(return_value="raw")
    wrapper = _wrapper(comp, runnable=runnable, weak_ref_output=True)
    assert wrapper(_T()) == ("weak", "raw")
    entry = wrapper.concrete_graph_entries["batch"]
    assert seen == ["raw", ("weak", "raw")]
    assert entry.output == ("weak", ("weak", "raw"))


def test_capture_gc_disable_patches_collect_and_empty_cache(comp):
    import gc

    observed = {}

    def runnable(*args, **kwargs):
        observed["gc"] = gc.collect()
        observed["empty"] = comp.torch.accelerator.empty_cache()
        return "out"

    wrapper = _wrapper(comp, runnable=runnable, gc_disable=True, weak_ref_output=False)
    assert wrapper(_T()) == "out"
    assert observed == {"gc": None, "empty": None}
    assert callable(gc.collect)


def test_capture_error_propagates_without_committing_entry(comp):
    runnable = MagicMock(side_effect=RuntimeError("capture failed"))
    wrapper = _wrapper(comp, runnable=runnable)
    with pytest.raises(RuntimeError, match="capture failed"):
        wrapper(_T())
    entry = wrapper.concrete_graph_entries["batch"]
    assert entry.graph is None
    assert comp.counter.num_cudagraph_captured == 0


def test_capture_uses_graph_runtime_scope(comp):
    runtime = MagicMock()
    runtime.capture_scope.return_value = nullcontext()
    runtime.replay_scope.return_value = nullcontext()
    wrapper = comp.module.GraphWrapper(
        MagicMock(return_value="out"),
        SimpleNamespace(compilation_config=SimpleNamespace()),
        Mode.FULL,
        comp.module.GraphOptions(weak_ref_output=False),
        graph_runtime=runtime,
    )
    wrapper(_T())
    runtime.capture_scope.assert_called_once_with(comp.context)
    wrapper(_T())
    runtime.replay_scope.assert_called_once_with(comp.context)


def test_replay_validates_debug_input_addresses(monkeypatch):
    state = load_comp(monkeypatch, logging_level="DEBUG")
    wrapper = _wrapper(state, weak_ref_output=False)
    wrapper(_T(1))
    wrapper(_T(1))
    with pytest.raises(AssertionError, match="Input addresses"):
        wrapper(_T(2))


def test_replay_synchronizes_on_npu(monkeypatch):
    state = load_comp(monkeypatch, device="npu")
    wrapper = _wrapper(state, weak_ref_output=False)
    wrapper(_T())
    wrapper(_T())
    state.platform.torch_device_fn.synchronize.assert_called_once_with()


@pytest.mark.parametrize("device,prefix", [("cuda", "upstream"), ("iluvatar", "iluvatar")])
def test_weak_ref_tensors_routes_by_platform(comp, device, prefix):
    comp.platform.device_type = "cuda"
    comp.platform.vendor_name = "iluvatar" if device == "iluvatar" else None
    comp.upstream_weak_ref.side_effect = lambda value: ("upstream", value)
    comp.iluvatar.side_effect = lambda value: ("iluvatar", value)
    assert comp.module.weak_ref_tensors("value") == (prefix, "value")


def test_weak_ref_tensors_npu_and_gcu_helpers(comp, monkeypatch):
    comp.platform.device_type = "npu"
    comp.module._weak_ref_npu_tensors = MagicMock(return_value="npu-ref")
    assert comp.module.weak_ref_tensors("x") == "npu-ref"
    comp.platform.device_type = "gcu"
    comp.module._weak_ref_gcu_tensors = MagicMock(return_value="gcu-ref")
    assert comp.module.weak_ref_tensors("x") == "gcu-ref"


def test_weak_ref_tensors_upstream_failure_returns_value(comp):
    comp.platform.device_type = "musa"
    comp.upstream_weak_ref.side_effect = RuntimeError("no upstream")
    value = object()
    assert comp.module.weak_ref_tensors(value) is value


def test_npu_single_weak_ref_requires_primitive(monkeypatch):
    state = load_comp(monkeypatch)
    cpu = _T()
    cpu.device = SimpleNamespace(type="cpu")
    assert state.module._weak_ref_npu_tensor(cpu) is cpu

    tensor = _T()
    tensor.device = SimpleNamespace(type="npu")
    monkeypatch.setitem(sys.modules, "torch_npu", _module("torch_npu", _C=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="NPU graph weak references require"):
        state.module._weak_ref_npu_tensor(tensor)

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        _module("torch_npu", _C=SimpleNamespace(_weak_ref_tensor=lambda value: calls.append(value) or "weak")),
    )
    assert state.module._weak_ref_npu_tensor(tensor) == "weak"
    assert calls == [tensor]


def test_npu_tree_recursion_preserves_structure(monkeypatch):
    state = load_comp(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        _module("torch_npu", _C=SimpleNamespace(_weak_ref_tensor=lambda value: ("w", value._address))),
    )
    first, second = _T(3), _T(4)
    first.device = second.device = SimpleNamespace(type="npu")
    intermediate = state.IntermediateTensors({"h": second})
    result = state.module._weak_ref_npu_tensors(
        {"list": [first, (None, first)], "inter": intermediate, "text": "keep"}
    )
    assert result["list"] == [("w", 3), (None, ("w", 3))]
    assert result["inter"].tensors == {"h": ("w", 4)}
    assert result["text"] == "keep"


class _GcuTensor:
    def __init__(self, conj=False, neg=False):
        self.device = SimpleNamespace(type="gcu")
        self._conj = conj
        self._neg = neg

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: 16, data_ptr=lambda: 100)

    shape = (2, 2)

    def stride(self):
        return (2, 1)

    dtype = "float32"

    def storage_offset(self):
        return 0

    def is_conj(self):
        return self._conj

    def is_neg(self):
        return self._neg


def test_gcu_single_weak_ref_paths(comp, monkeypatch):
    cpu = _GcuTensor()
    cpu.device = SimpleNamespace(type="cpu")
    assert comp.module._weak_ref_gcu_tensor(cpu) is cpu

    tensor = _GcuTensor()
    monkeypatch.setitem(sys.modules, "torch_gcu", _module("torch_gcu", _C=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="GCU graph weak references require"):
        comp.module._weak_ref_gcu_tensor(tensor)

    comp.torch._C._construct_storage_from_data_pointer = MagicMock(return_value="storage")
    set_conj, set_neg = MagicMock(), MagicMock()
    comp.torch._C._set_conj = set_conj
    comp.torch._C._set_neg = set_neg
    native = MagicMock(return_value="gcu-tensor")
    monkeypatch.setitem(
        sys.modules,
        "torch_gcu",
        _module(
            "torch_gcu",
            _C=SimpleNamespace(_construct_GCU_Tensor_From_Storage_And_Metadata=native),
        ),
    )
    value = _GcuTensor(conj=True, neg=True)
    result = comp.module._weak_ref_gcu_tensor(value)
    assert result == "gcu-tensor"
    storage_args = comp.torch._C._construct_storage_from_data_pointer.call_args.args
    assert storage_args == (100, value.device, 16)
    metadata = native.call_args.args[0]
    assert metadata["nbytes"] == 16 and metadata["storage_offset"] == 0
    set_conj.assert_called_once_with("gcu-tensor", True)
    set_neg.assert_called_once_with("gcu-tensor", True)


def test_gcu_tree_recursion_and_error_propagation(comp, monkeypatch):
    comp.torch.Tensor = _GcuTensor
    comp.torch._C._construct_storage_from_data_pointer = lambda *args: "storage"
    error = ValueError("incompatible metadata")

    def native(*args):
        raise error

    monkeypatch.setitem(
        sys.modules,
        "torch_gcu",
        _module(
            "torch_gcu",
            _C=SimpleNamespace(_construct_GCU_Tensor_From_Storage_And_Metadata=native),
        ),
    )
    with pytest.raises(ValueError, match="incompatible metadata"):
        comp.module._weak_ref_gcu_tensors([_GcuTensor()])

    def good(metadata, storage):
        return ("gcu", metadata["size"])

    monkeypatch.setitem(
        sys.modules,
        "torch_gcu",
        _module(
            "torch_gcu",
            _C=SimpleNamespace(_construct_GCU_Tensor_From_Storage_And_Metadata=good),
        ),
    )
    first, second = _GcuTensor(), _GcuTensor()
    intermediate = comp.IntermediateTensors({"h": second})
    result = comp.module._weak_ref_gcu_tensors(
        {"list": [first, (None, first)], "inter": intermediate, "text": "keep"}
    )
    assert result["list"] == [("gcu", (2, 2)), (None, ("gcu", (2, 2)))]
    assert result["inter"].tensors == {"h": ("gcu", (2, 2))}
    assert result["text"] == "keep"


@pytest.mark.parametrize(
    "device,expect_native",
    [("musa", True), ("npu", True), ("cuda", False)],
)
def test_graph_capture_stream_selection(comp, device, expect_native):
    comp.platform.device_type = device
    assert comp.module._graph_capture_stream() == (
        "native-stream" if expect_native else "vllm-stream"
    )
    if expect_native:
        comp.platform.torch_device_fn.current_stream.assert_called_once_with()









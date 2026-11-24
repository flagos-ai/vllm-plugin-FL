# Copyright (c) 2026 BAAI. All rights reserved.
"""Extra CPU controls for the MUSA native-graph wrapper.

The shared GraphEntry/GraphWrapper base is replaced by a CPU double that
performs no capture; the MUSA admission, staging and weak-reference logic runs
against the real production source.
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
import torch


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
                setattr(parent, parts[index - 1], sys.modules[full])


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


@dataclasses.dataclass
class GraphEntry:
    batch_descriptor: object
    graph: object | None = None
    output: object | None = None
    input_addresses: list[int] | None = None


class GraphWrapper:
    entry_type = GraphEntry

    def __init__(self, runnable, vllm_config, runtime_mode, cudagraph_options=None, graph_runtime=None):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.graph_runtime = graph_runtime
        self.cudagraph_options = cudagraph_options
        self.concrete_graph_entries = {}
        self.capture_calls = 0
        self.validated = 0

    def _capture_graph(self, entry, args, kwargs):
        self.capture_calls += 1
        output = self.runnable(*args, **kwargs)
        entry.graph = FakeGraph()
        entry.output = output
        return output

    def clear_graphs(self):
        self.concrete_graph_entries.clear()

    def _validate_replay_addresses(self, entry, args):
        self.validated += 1


class Log:
    def __init__(self):
        self.events = []

    def info(self, message, *args):
        self.events.append((message, args))


def load_musa_graph(monkeypatch):
    logs = Log()
    musa_native = SimpleNamespace(
        MUSAGraph=FakeGraph,
        current_device=lambda: 0,
        device=lambda ordinal: nullcontext(),
    )
    monkeypatch.setattr(torch, "musa", musa_native, raising=False)
    musa_module = _module(
        "vllm_fl.dispatch.backends.vendor.musa.graph",
        GraphEntry=GraphEntry,
        GraphWrapper=GraphWrapper,
    )
    _install(
        monkeypatch,
        {
            "vllm": _module("vllm"),
            "vllm.config": _module("vllm.config", CUDAGraphMode=Mode),
            "vllm.logger": _module("vllm.logger", init_logger=lambda name: logs),
            "vllm_fl": _module("vllm_fl"),
            "vllm_fl.compilation": _module("vllm_fl.compilation"),
            "vllm_fl.compilation.graph": musa_module,
        },
    )
    source = ROOT / "vllm_fl/dispatch/backends/vendor/musa/graph.py"
    spec = importlib.util.spec_from_file_location("w6b_musa_graph", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["w6b_musa_graph"] = module
    spec.loader.exec_module(module)
    return module, logs


@pytest.fixture
def musa(monkeypatch):
    module, logs = load_musa_graph(monkeypatch)
    # Fresh global budget so admission state never leaks across tests.
    module._musa_admission_budget = module._MusaGraphAdmissionBudget()
    wrapper = module.MUSAGraphWrapper(
        MagicMock(return_value="out"),
        SimpleNamespace(compilation_config=object()),
        Mode.FULL,
        module.GraphOptions(debug_log_enable=False, weak_ref_output=False)
        if hasattr(module, "GraphOptions")
        else None,
    )
    return module, wrapper, logs


def _entry(module, descriptor="d"):
    return module.MUSAGraphEntry(batch_descriptor=descriptor)


def test_budget_defaults_and_snapshot(musa):
    module, _, _ = musa
    budget = module._MusaGraphAdmissionBudget()
    assert budget.snapshot() == {
        "cap": 1900,
        "live": 0,
        "pending": 0,
        "total_admitted": 0,
        "epoch": 0,
        "poisoned": False,
    }
    for cap in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            module._MusaGraphAdmissionBudget(cap)


def test_budget_register_requires_reservation(musa):
    module, _, _ = musa
    budget = module._MusaGraphAdmissionBudget(cap=2)
    with pytest.raises(RuntimeError, match="no capture reservation"):
        budget.register(FakeGraph())
    with budget.capture() as admitted:
        assert admitted is True
        events = budget.register(FakeGraph())
    assert events == ["first_native_admission"]
    assert budget.snapshot()["total_admitted"] == 1


def test_budget_cap_event_is_announced_once(musa):
    module, _, _ = musa
    budget = module._MusaGraphAdmissionBudget(cap=1)
    keep = FakeGraph()
    with budget.capture() as admitted:
        assert admitted
        assert budget.register(keep) == ["first_native_admission", "cap_reached"]
    with budget.capture() as denied:
        assert denied is False
    assert budget.rejected() is True
    assert budget.rejected() is False


def test_budget_nested_capture_and_replay_rejected(musa):
    module, _, _ = musa
    budget = module._MusaGraphAdmissionBudget(cap=2)
    with budget.capture() as admitted:
        assert admitted
        with pytest.raises(RuntimeError, match="during capture or replay"), budget.capture():
            pass
        with pytest.raises(RuntimeError, match="during capture"), budget.replay():
            pass
        keep = FakeGraph()
        budget.register(keep)
    with budget.replay():
        with pytest.raises(RuntimeError, match="during capture or replay"), budget.capture():
            pass
        with pytest.raises(RuntimeError, match="during capture or replay"), budget.clearing():
            pass


def test_budget_uncommitted_capture_poisons(musa):
    module, _, _ = musa
    budget = module._MusaGraphAdmissionBudget(cap=1)
    with pytest.raises(RuntimeError, match="not committed"):
        with budget.capture() as admitted:
            assert admitted
    assert budget.snapshot()["poisoned"] is True
    with pytest.raises(RuntimeError, match="poisoned"), budget.eager():
        pass


def test_budget_eager_and_replay_errors_poison(musa):
    module, _, _ = musa
    budget = module._MusaGraphAdmissionBudget()
    with pytest.raises(ValueError, match="eager"):
        with budget.eager():
            raise ValueError("eager")
    assert budget.snapshot()["poisoned"] is True


def test_budget_cleared_to_zero_and_reopen(musa):
    module, _, _ = musa
    budget = module._MusaGraphAdmissionBudget(cap=1)
    assert budget.cleared_to_zero() is False
    keep = FakeGraph()
    with budget.capture():
        budget.register(keep)
    deleted = keep
    del keep
    # An externally-held graph keeps the budget live.
    assert budget.cleared_to_zero() is False
    del deleted
    import gc

    gc.collect()
    assert budget.cleared_to_zero() is True
    assert budget.snapshot()["epoch"] == 1
    assert budget.snapshot()["live"] == 0
    assert budget.rejected() is True


def test_tensor_metadata_and_alias_checks(musa):
    module, wrapper, _ = musa
    tensor = torch.arange(4)
    metadata = module.MUSAGraphWrapper._musa_tensor_metadata(tensor)
    assert metadata == (
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tensor.storage_offset(),
        tensor.requires_grad,
    )
    assert module.MUSAGraphWrapper._musa_tensors_alias(tensor, tensor) is True
    other = torch.arange(4)
    assert module.MUSAGraphWrapper._musa_tensors_alias(tensor, other) is False
    view = tensor[1:]
    assert module.MUSAGraphWrapper._musa_tensors_alias(tensor, view) is True


def test_capture_inputs_only_for_piecewise(musa):
    module, wrapper, _ = musa
    entry = _entry(module)
    wrapper.runtime_mode = Mode.FULL
    assert wrapper._capture_inputs(entry, (torch.ones(2),), {"k": 1}) is None
    wrapper.runtime_mode = Mode.PIECEWISE
    leaves, spec, metadata, addresses = wrapper._capture_inputs(
        entry, (torch.ones(2),), {"k": 1}
    )
    assert len(leaves) == 2
    assert metadata[0][0] == torch.Size([2])
    assert addresses[0] == leaves[0].data_ptr()
    assert metadata[1] is None and addresses[1] is None
    wrapper._finish_capture_inputs(entry, (leaves, spec, metadata, addresses))
    assert entry.musa_input_leaves == leaves
    assert entry.musa_input_spec == spec
    assert entry.musa_last_input_tensor_refs[0]() is leaves[0]
    wrapper._finish_capture_inputs(entry, None)


def test_capture_admitted_runs_native_and_logs_first(musa):
    module, wrapper, logs = musa
    entry = _entry(module)
    out = wrapper._capture_graph(entry, (), {})
    assert out == "out"
    assert isinstance(entry.graph, FakeGraph)
    assert entry.musa_capture_device == 0
    assert entry.musa_eager_only is False
    events = [args[0] for message, args in logs.events if "admission" in message]
    assert events == ["first_native_admission"]
    assert module._musa_admission_budget.snapshot()["live"] == 1


def test_capture_at_cap_falls_back_to_eager_and_rejects_once(musa):
    module, wrapper, logs = musa
    wrapper.runnable = MagicMock(return_value="eager")
    module._musa_admission_budget = module._MusaGraphAdmissionBudget(cap=1)
    first = _entry(module, "a")
    wrapper._capture_graph(first, (), {})
    denied = _entry(module, "b")
    assert wrapper._capture_graph(denied, (), {}) == "eager"
    assert denied.graph is None
    assert denied.musa_eager_only is True
    assert denied.musa_capture_device is None
    events = [args[0] for message, args in logs.events if "admission" in message]
    assert "first_admission_rejected" in events
    again = _entry(module, "c")
    wrapper._capture_graph(again, (), {})
    events = [args[0] for message, args in logs.events if "admission" in message]
    assert events.count("first_admission_rejected") == 1
    assert wrapper.runnable.call_count == 3


def test_capture_eager_only_entry_skips_admission(musa):
    module, wrapper, _ = musa
    entry = _entry(module)
    entry.musa_eager_only = True
    assert wrapper._capture_graph(entry, (), {}) == "out"
    assert entry.graph is None
    assert module._musa_admission_budget.snapshot()["live"] == 0


def test_capture_error_propagates_and_poisons(musa):
    module, wrapper, _ = musa
    wrapper.runnable = MagicMock(side_effect=RuntimeError("native boom"))
    with pytest.raises(RuntimeError, match="native boom"):
        wrapper._capture_graph(_entry(module), (), {})
    assert module._musa_admission_budget.snapshot()["poisoned"] is True
    with pytest.raises(RuntimeError, match="poisoned"):
        wrapper._capture_graph(_entry(module), (), {})


def test_replay_without_captured_inputs_validates_and_replays(musa):
    module, wrapper, _ = musa
    entry = _entry(module)
    entry.musa_capture_device = 0
    entry.graph = FakeGraph()
    entry.output = "captured"
    assert wrapper._replay_graph(entry, (), {}) == "captured"
    assert wrapper.validated == 1
    assert entry.graph.replays == 1


def test_replay_with_stable_piecewise_inputs_replays(musa):
    module, wrapper, _ = musa
    wrapper.runtime_mode = Mode.PIECEWISE
    entry = _entry(module)
    args = (torch.ones(3),)
    inputs = wrapper._capture_inputs(entry, args, {})
    wrapper._finish_capture_inputs(entry, inputs)
    entry.musa_capture_device = 0
    entry.graph = FakeGraph()
    entry.output = "captured"
    assert wrapper._replay_graph(entry, args, {}) == "captured"
    assert entry.graph.replays == 1
    assert wrapper.capture_calls == 0


def test_replay_with_changed_structure_falls_back_to_eager(musa):
    module, wrapper, _ = musa
    wrapper.runtime_mode = Mode.PIECEWISE
    entry = _entry(module)
    inputs = wrapper._capture_inputs(entry, (torch.ones(3),), {})
    wrapper._finish_capture_inputs(entry, inputs)
    entry.musa_capture_device = 0
    entry.graph = FakeGraph()
    entry.output = "captured"
    with pytest.raises(ValueError):
        wrapper._replay_graph(entry, (torch.ones(3), torch.ones(3)), {})
    assert entry.graph.replays == 0


def test_replay_stable_flag_false_runs_eager(musa):
    module, wrapper, _ = musa
    wrapper.runnable = MagicMock(return_value="eager")
    entry = _entry(module)
    entry.musa_capture_device = 0
    entry.musa_input_leaves = [torch.ones(2)]
    entry.musa_input_spec = "spec"
    entry.musa_input_metadata = [None]
    entry.musa_input_addresses = [None]
    entry.musa_last_input_tensor_refs = [None]
    wrapper._stage_musa_inputs = MagicMock(return_value=False)
    assert wrapper._replay_graph(entry, (), {}) == "eager"
    assert wrapper.runnable.call_count == 1


def test_clear_graphs_logs_cleared_to_zero(musa):
    module, wrapper, logs = musa
    entry = _entry(module)
    wrapper._capture_graph(entry, (), {})
    wrapper.concrete_graph_entries["d"] = entry
    del entry
    import gc

    gc.collect()
    wrapper.clear_graphs()
    wrapper.concrete_graph_entries.clear()
    events = [args[0] for message, args in logs.events if "admission" in message]
    assert "cleared_to_zero" in events


def test_capture_context_is_nullcontext(musa):
    module, wrapper, _ = musa
    ctx = wrapper._capture_context(FakeGraph())
    assert ctx.__class__ is nullcontext


def test_weak_ref_output_routes_to_musa_helper(musa, monkeypatch):
    module, wrapper, _ = musa
    seen = []
    monkeypatch.setattr(module, "_weak_ref_musa_tensors", lambda value: seen.append(value) or "weak")
    assert wrapper._weak_ref_output("value") == "weak"
    assert seen == ["value"]






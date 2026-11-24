"""Coverage tests for model_runner.py lines 1-1200: module helpers + async output.

The heavy ``model_runner`` module is never imported. Instead the exact
``FunctionDef``/``ClassDef`` nodes are lifted out of the source with ``ast`` and
executed with a hand-built namespace of stubs, compiling under the *real* file
path so coverage is attributed to ``vllm_fl/worker/model_runner.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch


def _source_path() -> Path:
    spec = importlib.util.find_spec("vllm_fl")
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parent / "worker" / "model_runner.py"
    return Path.cwd() / "vllm_fl" / "worker" / "model_runner.py"


_SRC = _source_path()


def _load(names, namespace):
    namespace.setdefault("NamedTuple", NamedTuple)
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    wanted = set(names)
    body = [n for n in tree.body if getattr(n, "name", None) in wanted]
    assert {getattr(n, "name", None) for n in body} == wanted
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *body], type_ignores=[])
    )
    exec(compile(module, str(_SRC), "exec"), namespace)
    return namespace


# --------------------------------------------------------------------------- #
# fakes for the accelerator / torch-device API
# --------------------------------------------------------------------------- #
class FakeStream:
    def __init__(self, *args, **kwargs):
        self.device = kwargs.get("device")
        self.waited = None

    def wait_stream(self, other):
        self.waited = other


class FakeEvent:
    def __init__(self, *args, **kwargs):
        self.recorded = False
        self.sync_count = 0

    def record(self):
        self.recorded = True

    def synchronize(self):
        self.sync_count += 1


class FakeCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class FakeDeviceFn:
    def __init__(self):
        self.current = object()
        self.streams = []
        self.events = []

    def Stream(self, *args, **kwargs):
        s = FakeStream(*args, **kwargs)
        self.streams.append(s)
        return s

    def Event(self, *args, **kwargs):
        e = FakeEvent(*args, **kwargs)
        self.events.append(e)
        return e

    def current_stream(self, *args, **kwargs):
        return self.current

    def stream(self, stream):
        return FakeCtx()


class AsyncModelRunnerOutput:
    pass


class BaseLayerWithLoRA:
    pass


class GraphCaptureContext:
    def __init__(self, stream):
        self.stream = stream


def _namespace():
    device_fn = FakeDeviceFn()
    platform = SimpleNamespace(device_type="cpu", torch_device_fn=device_fn)
    envs = SimpleNamespace(VLLM_RAISE_ON_LOGIT_NANS=False)
    manager = SimpleNamespace(
        query_fault=lambda: torch.tensor(False),
        query_active_mask=lambda: torch.tensor([7]),
    )
    ns = {
        "torch": torch,
        "np": None,
        "contextmanager": contextmanager,
        "nullcontext": nullcontext,
        "GraphCaptureContext": GraphCaptureContext,
        "current_platform": platform,
        "AsyncModelRunnerOutput": AsyncModelRunnerOutput,
        "BaseLayerWithLoRA": BaseLayerWithLoRA,
        "PoolerOutput": object,
        "envs": envs,
        "get_ep_all2all_manager": lambda: manager,
        "raise_if_nan_logits": lambda counts: None,
        "RejectionSampler": SimpleNamespace(),
        "_device_fn": device_fn,
        "_manager": manager,
    }
    _load(
        {
            "_accelerator_synchronize",
            "_device_graph_capture",
            "_get_parameter_for_reload",
            "count_nans_per_row",
            "nans_to_dict",
            "_copy_pooler_output_to_cpu",
            "AsyncGPUModelRunnerOutput",
            "AsyncGPUPoolingModelRunnerOutput",
            "ExecuteModelState",
        },
        ns,
    )
    return ns


@pytest.fixture()
def ns():
    return _namespace()


# --------------------------------------------------------------------------- #
# module level helpers
# --------------------------------------------------------------------------- #
def test_count_nans_per_row_and_nans_to_dict(ns):
    logits = torch.tensor([[1.0, float("nan"), 2.0], [float("nan"), float("nan"), 3.0]])
    counts = ns["count_nans_per_row"](logits)
    assert counts.dtype == torch.int32
    assert counts.tolist() == [1, 2]

    mapped = ns["nans_to_dict"]([1, 2], {"a": 0, "b": 1, "missing": 5})
    # row index beyond len(counts) -> 0
    assert mapped == {"a": 1, "b": 2, "missing": 0}


def test_accelerator_synchronize_uses_torch_musa(monkeypatch, ns):
    calls = []
    fake = SimpleNamespace(synchronize=lambda: calls.append("musa"))
    monkeypatch.setitem(sys.modules, "torch_musa", fake)
    monkeypatch.setattr(ns["current_platform"], "device_type", "musa")
    ns["_accelerator_synchronize"]()
    assert calls == ["musa"]


def test_accelerator_synchronize_uses_torch_accelerator(monkeypatch, ns):
    calls = []
    monkeypatch.setattr(ns["current_platform"], "device_type", "cuda")
    monkeypatch.setattr(
        ns["torch"], "accelerator", SimpleNamespace(synchronize=lambda: calls.append("acc")), raising=False
    )
    ns["_accelerator_synchronize"]()
    assert calls == ["acc"]


def test_device_graph_capture_waits_when_stream_differs(ns):
    provided = GraphCaptureContext(FakeStream())
    with ns["_device_graph_capture"](torch.device("cpu"), provided) as ctx:
        assert ctx is provided
    # current_stream() returns a different fake object -> wait_stream invoked
    assert provided.stream.waited is ns["_device_fn"].current


def test_device_graph_capture_creates_stream_when_none(ns):
    with ns["_device_graph_capture"](torch.device("cpu"), None) as ctx:
        assert ctx.stream is not None
    assert len(ns["_device_fn"].streams) == 1


def test_device_graph_capture_no_wait_when_same_stream(ns):
    same = FakeStream()
    ns["_device_fn"].current = same
    provided = GraphCaptureContext(same)
    with ns["_device_graph_capture"](torch.device("cpu"), provided):
        pass
    assert same.waited is None


def test_get_parameter_for_reload_plain_and_lora(ns):
    param = torch.nn.Parameter(torch.zeros(2))

    class Mod:
        def __init__(self, use_lora):
            if use_lora:
                self.base_layer = SimpleNamespace(get_parameter=lambda n: param)
            else:
                self.get_parameter = lambda n: param

    class Model:
        def __init__(self, mod):
            self._mod = mod

        def get_submodule(self, name):
            assert name == "layer.0"
            return self._mod

    assert ns["_get_parameter_for_reload"](Model(Mod(False)), "layer.0.w") is param

    # isinstance(module, BaseLayerWithLoRA) True -> unwrap .base_layer
    obj = BaseLayerWithLoRA.__new__(BaseLayerWithLoRA)
    obj.base_layer = Mod(True).base_layer
    assert ns["_get_parameter_for_reload"](Model(obj), "layer.0.w") is param


def test_execute_model_state_is_namedtuple(ns):
    state = ns["ExecuteModelState"](
        "sched", torch.zeros(1), None, None, torch.zeros(2),
        torch.zeros(2), None, None, None, None,
    )
    assert state.scheduler_output == "sched"
    assert state.slot_mappings is None


# --------------------------------------------------------------------------- #
# _copy_pooler_output_to_cpu
# --------------------------------------------------------------------------- #
def test_copy_pooler_tensor_none_finished(ns):
    raw = torch.arange(6).reshape(3, 2)
    out = ns["_copy_pooler_output_to_cpu"](raw, [False, False, False])
    assert out == [None, None, None]


def test_copy_pooler_tensor_all_finished(ns):
    raw = torch.arange(6).reshape(3, 2)
    out = ns["_copy_pooler_output_to_cpu"](raw, [True, True, True])
    assert len(out) == 3
    assert torch.equal(out[1], raw[1])


def test_copy_pooler_tensor_partial_finished(ns):
    raw = torch.arange(6).reshape(3, 2)
    out = ns["_copy_pooler_output_to_cpu"](raw, [True, False, True])
    assert torch.equal(out[0], raw[0])
    assert out[1] is None
    assert torch.equal(out[2], raw[2])


def test_copy_pooler_tensor_batch_mismatch(ns):
    with pytest.raises(ValueError, match="does not match"):
        ns["_copy_pooler_output_to_cpu"](torch.zeros(2, 2), [True, True, True])


def test_copy_pooler_list_branch(ns):
    vals = [torch.tensor([1.0]), None, torch.tensor([3.0])]
    out = ns["_copy_pooler_output_to_cpu"](vals, [True, True, True])
    assert torch.equal(out[0], vals[0])
    assert out[1] is None
    assert torch.equal(out[2], vals[2])


def test_copy_pooler_list_mismatch(ns):
    with pytest.raises(ValueError, match="does not match"):
        ns["_copy_pooler_output_to_cpu"]([torch.zeros(1)], [True, True])


# --------------------------------------------------------------------------- #
# AsyncGPUModelRunnerOutput
# --------------------------------------------------------------------------- #
def _mr_output(req_map=None):
    return SimpleNamespace(
        sampled_token_ids=None,
        logprobs=None,
        req_id_to_index={} if req_map is None else req_map,
    )


def test_async_output_single_token_and_invalid_req(ns):
    DeviceStream = ns["_device_fn"].Stream
    sampled = torch.tensor([[10], [11], [12]])
    out = ns["AsyncGPUModelRunnerOutput"](
        model_runner_output=_mr_output(),
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        invalid_req_indices=[0, 2],
        async_output_copy_stream=DeviceStream(),
        vocab_size=100,
    )
    assert out.async_copy_ready_event.recorded is True
    result = out.get_output()
    assert out.async_copy_ready_event.sync_count == 1
    assert result.sampled_token_ids == [[], [11], []]
    assert result.logprobs is None


def test_async_output_multi_token_uses_rejection_parser(ns, monkeypatch):
    seen = {}

    def parse(sampled, vocab, invalid, logprobs_tensors=None):
        seen["vocab"] = vocab
        return [[1, 2], [3, 4]], ["lp"]

    monkeypatch.setattr(
        ns["RejectionSampler"], "parse_output", staticmethod(parse), raising=False
    )
    sampled = torch.tensor([[1, 2], [3, 4]])
    out = ns["AsyncGPUModelRunnerOutput"](
        model_runner_output=_mr_output(),
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=ns["_device_fn"].Stream(),
        vocab_size=321,
    )
    result = out.get_output()
    assert result.sampled_token_ids == [[1, 2], [3, 4]]
    assert result.logprobs == ["lp"]
    assert seen["vocab"] == 321


def test_async_output_nans_mapped(ns):
    sampled = torch.tensor([[5], [6]])
    out = ns["AsyncGPUModelRunnerOutput"](
        model_runner_output=_mr_output({"a": 0, "b": 4}),
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=ns["_device_fn"].Stream(),
        vocab_size=10,
        num_nans=torch.tensor([3, 1], dtype=torch.int32),
    )
    result = out.get_output()
    assert result.num_nans_in_logits == {"a": 3, "b": 0}


def test_async_output_raises_on_nan_when_env_set(ns, monkeypatch):
    calls = []
    monkeypatch.setattr(ns["envs"], "VLLM_RAISE_ON_LOGIT_NANS", True)
    ns["raise_if_nan_logits"] = lambda counts: calls.append(counts)
    out = ns["AsyncGPUModelRunnerOutput"](
        model_runner_output=_mr_output({"a": 0}),
        sampled_token_ids=torch.tensor([[1]]),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=ns["_device_fn"].Stream(),
        vocab_size=10,
        num_nans=torch.tensor([2], dtype=torch.int32),
    )
    out.get_output()
    assert calls == [{"a": 2}]


def test_async_output_ep_fault_query_at_init(ns):
    out = ns["AsyncGPUModelRunnerOutput"](
        model_runner_output=_mr_output(),
        sampled_token_ids=torch.tensor([[1]]),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=ns["_device_fn"].Stream(),
        vocab_size=10,
        check_ep_fault=True,
    )
    assert out._has_fault is not None
    assert out._has_fault.item() is False


def test_async_output_ep_fault_raises(ns):
    out = ns["AsyncGPUModelRunnerOutput"](
        model_runner_output=_mr_output(),
        sampled_token_ids=torch.tensor([[1]]),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=ns["_device_fn"].Stream(),
        vocab_size=10,
    )
    out._has_fault = torch.tensor(True)
    with pytest.raises(RuntimeError, match="Fault detected"):
        out.get_output()


# --------------------------------------------------------------------------- #
# AsyncGPUPoolingModelRunnerOutput
# --------------------------------------------------------------------------- #
def test_pooling_output_tensor(ns):
    raw = torch.arange(6).reshape(3, 2)
    mr = _mr_output()
    out = ns["AsyncGPUPoolingModelRunnerOutput"](
        model_runner_output=mr,
        raw_pooler_output=raw,
        finished_mask=[True, False, True],
        async_output_copy_stream=ns["_device_fn"].Stream(),
    )
    assert mr.pooler_output[0] is not None
    assert mr.pooler_output[1] is None
    result = out.get_output()
    assert result is mr
    assert out.async_copy_ready_event.sync_count == 1
    assert not hasattr(out, "_raw_pooler_output")

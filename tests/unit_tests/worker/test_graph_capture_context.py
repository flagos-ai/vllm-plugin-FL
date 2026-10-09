# Copyright (c) 2026 BAAI. All rights reserved.
"""Execute the production graph_capture helper with CPU stream doubles.

AST locates the unchanged production function; the assertions exercise its
context-manager behavior. This does not import the full model runner, select a
backend, or validate native graph capture, GPU ABI, or model inference.
"""

import __future__

import ast
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


@dataclass
class _CaptureContext:
    stream: object
    marker: object = None


class _Stream:
    def __init__(self, backend, name):
        self.backend = backend
        self.name = name

    def wait_stream(self, current):
        self.backend.events.append(("wait", self, current))
        if "wait" in self.backend.errors:
            raise self.backend.errors["wait"]


class _EqualStream(_Stream):
    def __eq__(self, other):
        return isinstance(other, _EqualStream)

    def __ne__(self, other):
        return not self == other


class _Backend:
    def __init__(self):
        self.events = []
        self.errors = {}
        self.stream_devices = []
        self.contexts_created = []
        self.current = _Stream(self, "previous")

    def Stream(self, *, device):
        self.stream_devices.append(device)
        self.events.append(("create-stream", device))
        if "stream" in self.errors:
            raise self.errors["stream"]
        return _Stream(self, "created")

    def context_factory(self, stream):
        context = _CaptureContext(stream)
        self.contexts_created.append(context)
        self.events.append(("create-context", context))
        return context

    def current_stream(self):
        self.events.append(("current", self.current))
        if "current" in self.errors:
            raise self.errors["current"]
        return self.current

    @contextmanager
    def stream(self, target):
        self.events.append(("enter", target))
        if "enter" in self.errors:
            raise self.errors["enter"]
        previous = self.current
        self.current = target
        try:
            yield
        finally:
            self.current = previous
            self.events.append(("exit", target))


@pytest.fixture
def helper_and_backend():
    source = Path(__file__).resolve().parents[3] / "vllm_fl/worker/model_runner.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [
        child
        for node in module.body
        if isinstance(node, ast.If)
        for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name == "graph_capture"
    ]
    assert len(functions) == 1
    backend = _Backend()
    namespace = {
        "__name__": "isolated_production_graph_capture_contract",
        "contextmanager": contextmanager,
        "nullcontext": nullcontext,
        "GraphCaptureContext": backend.context_factory,
        "current_platform": SimpleNamespace(torch_device_fn=backend),
    }
    function_module = ast.Module(body=functions, type_ignores=[])
    exec(
        compile(
            function_module,
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace["graph_capture"], backend


@pytest.mark.parametrize(
    "explicit_none", [False, True], ids=["default", "explicit-none"]
)
def test_creates_context_when_absent(helper_and_backend, explicit_none):
    helper, backend = helper_and_backend
    previous = backend.current
    device = object()
    kwargs = {"graph_capture_context": None} if explicit_none else {}
    with helper(device=device, **kwargs) as context:
        assert backend.current is context.stream
        assert backend.stream_devices == [device]
        assert backend.contexts_created == [context]
        assert [event[0] for event in backend.events] == [
            "create-stream",
            "create-context",
            "current",
            "wait",
            "enter",
        ]
        assert backend.events[3] == ("wait", context.stream, previous)
    assert backend.current is previous
    assert backend.events[-1] == ("exit", context.stream)


def test_reuses_supplied_context_and_waits_before_enter(helper_and_backend):
    helper, backend = helper_and_backend
    previous = backend.current
    context = _CaptureContext(_Stream(backend, "supplied"), marker=object())
    with helper(device=object(), graph_capture_context=context) as yielded:
        assert yielded is context
        assert backend.current is context.stream
        assert backend.stream_devices == []
        assert backend.contexts_created == []
        assert backend.events == [
            ("current", previous),
            ("wait", context.stream, previous),
            ("enter", context.stream),
        ]
    assert backend.current is previous
    assert backend.events[-1] == ("exit", context.stream)


@pytest.mark.parametrize(
    "equal_wrapper", [False, True], ids=["same-object", "equal-wrapper"]
)
def test_skips_wait_for_current_stream(helper_and_backend, equal_wrapper):
    helper, backend = helper_and_backend
    if equal_wrapper:
        backend.current = _EqualStream(backend, "previous")
        capture_stream = _EqualStream(backend, "equal-wrapper")
        assert capture_stream is not backend.current
    else:
        capture_stream = backend.current
    previous = backend.current
    context = _CaptureContext(capture_stream)
    with helper(device=object(), graph_capture_context=context) as yielded:
        assert yielded is context
        assert backend.current is capture_stream
        assert [event[0] for event in backend.events] == ["current", "enter"]
    assert backend.current is previous
    assert backend.stream_devices == backend.contexts_created == []
    assert [event[0] for event in backend.events] == ["current", "enter", "exit"]


def test_reuses_falsy_context(helper_and_backend):
    helper, backend = helper_and_backend

    class FalsyContext(_CaptureContext):
        def __bool__(self):
            return False

    context = FalsyContext(_Stream(backend, "supplied"), marker=object())
    with helper(device=object(), graph_capture_context=context) as yielded:
        assert yielded is context
        assert yielded.marker is context.marker
    assert backend.stream_devices == backend.contexts_created == []


def test_nested_capture_restores_the_entering_stream(helper_and_backend):
    helper, backend = helper_and_backend
    previous = backend.current
    outer = _CaptureContext(_Stream(backend, "outer"))
    inner = _CaptureContext(_Stream(backend, "inner"))
    with helper(device=object(), graph_capture_context=outer) as outer_result:
        assert outer_result is outer
        with helper(device=object(), graph_capture_context=inner) as inner_result:
            assert inner_result is inner
            assert backend.current is inner.stream
        assert backend.current is outer.stream
    assert backend.current is previous
    assert backend.events == [
        ("current", previous),
        ("wait", outer.stream, previous),
        ("enter", outer.stream),
        ("current", outer.stream),
        ("wait", inner.stream, outer.stream),
        ("enter", inner.stream),
        ("exit", inner.stream),
        ("exit", outer.stream),
    ]
    assert backend.stream_devices == backend.contexts_created == []


@pytest.mark.parametrize(
    "supplied", [False, True], ids=["new-context", "supplied-context"]
)
def test_body_error_propagates_and_restores_stream(helper_and_backend, supplied):
    helper, backend = helper_and_backend
    previous = backend.current
    context = _CaptureContext(_Stream(backend, "supplied"))
    kwargs = {"graph_capture_context": context} if supplied else {}
    error = RuntimeError("body sentinel")
    with (
        pytest.raises(RuntimeError) as caught,
        helper(device=object(), **kwargs) as yielded,
    ):
        assert backend.current is yielded.stream
        raise error
    assert caught.value is error
    assert backend.current is previous
    assert [event[0] for event in backend.events].count("exit") == 1


@pytest.mark.parametrize("stage", ["stream", "current", "wait", "enter"])
def test_setup_error_propagates_without_entering_body(helper_and_backend, stage):
    helper, backend = helper_and_backend
    previous = backend.current
    error = RuntimeError(stage + " sentinel")
    backend.errors[stage] = error
    entered_body = False
    with pytest.raises(RuntimeError) as caught, helper(device=object()):
        entered_body = True
    assert caught.value is error
    assert entered_body is False
    assert backend.current is previous
    assert not any(event[0] == "exit" for event in backend.events)
    if stage != "enter":
        assert not any(event[0] == "enter" for event in backend.events)

# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagCX must follow the selected TXDA stream between graph regions."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def stream_selector():
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/distributed/device_communicators/flagcx.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PyFlagcxCommunicator"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_current_stream"
    )
    runtime = SimpleNamespace(
        npu=SimpleNamespace(current_stream=Mock()),
        txda=SimpleNamespace(current_stream=Mock()),
    )
    cached = Mock(return_value=object())
    namespace = {"torch": runtime, "current_stream": cached}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["_current_stream"], runtime, cached


def test_txda_collectives_follow_each_selected_stream(stream_selector):
    select, runtime, cached = stream_selector
    comm = SimpleNamespace(device=SimpleNamespace(type="txda", index=3))
    default, capture, after_capture = object(), object(), object()
    runtime.txda.current_stream.side_effect = [default, capture, after_capture]
    assert [select(comm) for _ in range(3)] == [default, capture, after_capture]
    assert runtime.txda.current_stream.call_count == 3
    runtime.txda.current_stream.assert_called_with(comm.device)
    cached.assert_not_called()


def test_npu_collectives_keep_the_native_stream(stream_selector):
    select, runtime, cached = stream_selector
    comm = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert select(comm) is runtime.npu.current_stream.return_value
    runtime.npu.current_stream.assert_called_once_with(comm.device)
    runtime.txda.current_stream.assert_not_called()
    cached.assert_not_called()


def test_cuda_collectives_keep_the_upstream_stream_helper(stream_selector):
    select, runtime, cached = stream_selector
    comm = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    assert select(comm) is cached.return_value
    cached.assert_called_once_with()
    runtime.txda.current_stream.assert_not_called()
    runtime.npu.current_stream.assert_not_called()

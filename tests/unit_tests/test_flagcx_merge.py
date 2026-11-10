"""Keep upstream trace protection and the Enflame stream adapter together."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode


def test_trace_guard_and_gcu_stream_adapter():
    path = Path(__file__).resolve().parents[2] / (
        "vllm_fl/distributed/device_communicators/flagcx.py"
    )
    tree = ast.parse(path.read_text())
    guard = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "is_trace_tensor"
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PyFlagcxCommunicator"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_copy_flagcx_stream", "all_reduce"}
    ]
    ns = {
        "torch": torch,
        "_FAKE_TENSOR_CLS": (FakeTensor,),
        "SimpleNamespace": SimpleNamespace,
        "ReduceOp": SimpleNamespace(SUM=0),
    }
    exec(
        compile(ast.Module(body=[guard, *methods], type_ignores=[]), str(path), "exec"),
        ns,
    )
    assert not ns["is_trace_tensor"](None)
    assert not ns["is_trace_tensor"](torch.zeros(2))
    assert ns["is_trace_tensor"](torch.empty(2, device="meta"))
    native = Mock()
    communicator = SimpleNamespace(
        disabled=False, flagcx=native, device=SimpleNamespace(type="gcu")
    )
    ns["_copy_flagcx_stream"](communicator, SimpleNamespace(gcu_stream=17))
    assert native.adaptor_stream_copy.call_args.args[0].cuda_stream == 17
    native.reset_mock()
    with FakeTensorMode():
        tensor = torch.empty(3)
        assert ns["is_trace_tensor"](tensor)
        result = ns["all_reduce"](communicator, tensor)
        assert result.shape == tensor.shape
    assert not native.mock_calls

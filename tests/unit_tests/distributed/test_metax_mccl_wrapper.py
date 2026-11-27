# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU ABI and alias checks for vLLM 0.32's MACA communicator bindings."""

import ast
import ctypes
import importlib
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/metax/patches"
)


def _definitions(filename, names, **namespace):
    tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(SOURCE / filename), "exec"), namespace)
    return SimpleNamespace(**namespace)


@dataclass
class _Function:
    name: str
    restype: object
    argtypes: list


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


class _Library:
    def __init__(self, missing=()):
        self.missing = missing
        self.calls = []
        self.functions = {}

    def __getattr__(self, name):
        if name in self.missing:
            raise AttributeError(name)
        if name not in self.functions:
            self.functions[name] = Mock(
                side_effect=lambda *args: self._call(name, args)
            )
        return self.functions[name]

    def _call(self, name, args):
        self.calls.append((name, args))
        return 0


def _wrapper(monkeypatch, library):
    loader = Mock(return_value=library)
    monkeypatch.setattr(ctypes, "CDLL", loader)
    namespace = dict(
        ctypes=ctypes,
        platform=platform,
        Function=_Function,
        buffer_type=ctypes.c_void_p,
        cudaStream_t=ctypes.c_void_p,
        ncclComm_t=ctypes.c_void_p,
        ncclDataType_t=ctypes.c_int,
        ncclRedOp_t=ctypes.c_int,
        ncclResult_t=ctypes.c_int,
        ncclWindow_t=ctypes.c_void_p,
        ncclUniqueId=_UniqueId,
        logger=SimpleNamespace(error=Mock()),
        find_mccl_library=lambda: "test-mccl.so",
    )
    module = _definitions("pynccl_wrapper.py", ["MCCLLibrary"], **namespace)
    return module.MCCLLibrary, loader


def test_optional_suspend_symbols_do_not_disable_ordinary_collectives(monkeypatch):
    library = _Library(missing=("mcclCommSuspend", "mcclCommResume"))
    wrapper, loader = _wrapper(monkeypatch, library)
    first = wrapper()
    assert not first.has_symbol("ncclCommSuspend")
    assert not first.has_symbol("ncclCommResume")
    assert first.has_symbol("ncclAllReduce")
    first.ncclAllReduce(1, 2, 3, 4, 5, 6, 7)
    assert library.calls == [("mcclAllReduce", (1, 2, 3, 4, 5, 6, 7))]
    second = wrapper()
    assert first.lib is second.lib
    loader.assert_called_once_with("test-mccl.so")


def test_abort_suspend_resume_use_maca_abi(monkeypatch):
    library = _Library()
    wrapper, _ = _wrapper(monkeypatch, library)
    comm = wrapper()
    assert comm.has_symbol("ncclCommAbort")
    assert comm.has_symbol("ncclCommSuspend")
    comm.ncclCommAbort(123)
    comm.ncclCommSuspend(456, 1)
    comm.ncclCommResume(456)
    assert library.calls == [
        ("mcclCommAbort", (123,)),
        ("mcclCommSuspend", (456, 1)),
        ("mcclCommResume", (456,)),
    ]
    assert library.functions["mcclCommSuspend"].argtypes == [
        ctypes.c_void_p,
        ctypes.c_int,
    ]


def test_missing_abort_never_falls_back_to_collective_destroy(monkeypatch):
    library = _Library(missing=("mcclCommAbort",))
    wrapper, _ = _wrapper(monkeypatch, library)
    with pytest.raises(AttributeError, match="mcclCommAbort"):
        wrapper()
    assert not library.calls


def test_symmetric_windows_fail_instead_of_returning_null_handles(monkeypatch):
    wrapper, _ = _wrapper(monkeypatch, _Library())
    comm = wrapper()
    assert not comm.has_symbol("ncclCommWindowRegister")
    with pytest.raises(NotImplementedError, match="symmetric-memory"):
        comm.ncclCommWindowRegister(1, 2, 3, 4)
    with pytest.raises(NotImplementedError, match="symmetric-memory"):
        comm.ncclCommWindowDeregister(1, 2)


def test_mccl_library_environment_override_precedence(monkeypatch):
    find = _definitions(
        "utils_patch.py",
        ["find_mccl_library"],
        os=os,
        torch=SimpleNamespace(version=SimpleNamespace(cuda="12.0")),
    ).find_mccl_library
    monkeypatch.delenv("VLLM_MCCL_SO_PATH", raising=False)
    monkeypatch.delenv("VLLM_NCCL_SO_PATH", raising=False)
    assert find() == "libmccl.so"
    monkeypatch.setenv("VLLM_NCCL_SO_PATH", "/native/override.so")
    assert find() == "/native/override.so"
    monkeypatch.setenv("VLLM_MCCL_SO_PATH", "/vendor/override.so")
    assert find() == "/vendor/override.so"


def test_discovery_rebinds_loaded_aliases_by_identity_only(monkeypatch):
    replace = _definitions(
        "utils_patch.py",
        ["replace_loaded_vllm_aliases"],
        importlib=importlib,
        sys=sys,
    ).replace_loaded_vllm_aliases
    defining = ModuleType("vllm.mccl_test_defining")
    alias = ModuleType("vllm.mccl_test_alias")
    unrelated = ModuleType("vllm.mccl_test_other")
    external = ModuleType("non_vllm_mccl_test")
    original, replacement, different = object(), object(), object()
    defining.find_nccl_library = original
    alias.find_nccl_library = original
    unrelated.find_nccl_library = different
    external.find_nccl_library = original
    for module in (defining, alias, unrelated, external):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    replace(defining.__name__, "find_nccl_library", replacement)
    assert defining.find_nccl_library is replacement
    assert alias.find_nccl_library is replacement
    assert unrelated.find_nccl_library is different
    assert external.find_nccl_library is original


@pytest.mark.parametrize("strided", [False, True])
def test_compile_friendly_all_reduce_preserves_input_and_returns_independent_tensor(
    strided,
):
    calls = []
    sum_op, group = object(), object()

    def all_reduce(tensor, *, op, group):
        calls.append((tensor, op, group))
        tensor.mul_(4)  # Simulate SUM over four identical CPU rank inputs.

    reduce = _definitions(
        "pynccl_wrapper.py",
        ["_compile_friendly_all_reduce"],
        dist=SimpleNamespace(
            all_reduce=all_reduce, ReduceOp=SimpleNamespace(SUM=sum_op)
        ),
    )._compile_friendly_all_reduce
    owner = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    input_ = owner[1::2, 1::2] if strided else owner
    original_owner, original_input = owner.clone(), input_.clone()
    result = reduce(SimpleNamespace(device_group=group), input_)
    torch.testing.assert_close(owner, original_owner)
    torch.testing.assert_close(input_, original_input)
    torch.testing.assert_close(result, original_input * 4)
    assert result is not input_
    assert result.untyped_storage().data_ptr() != input_.untyped_storage().data_ptr()
    assert len(calls) == 1
    assert calls[0][0] is result and calls[0][1] is sum_op and calls[0][2] is group

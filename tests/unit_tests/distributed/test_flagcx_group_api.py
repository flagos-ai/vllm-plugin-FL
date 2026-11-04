# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU coverage of the legacy and packaged FlagCX group signatures."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def group_methods():
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
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"_native_group_start", "_native_group_end", "group_start", "group_end"}
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return type(
        "GroupMethods", (), {method.name: namespace[method.name] for method in methods}
    )


class CommunicatorGroupLibrary:
    """The modern wrapper, plus the legacy Ascend signature."""

    def __init__(self):
        self.calls = []

    def flagcxGroupStart(self, comm):
        self.calls.append(("start", comm))

    def flagcxGroupEnd(self, comm):
        self.calls.append(("end", comm))


class LegacyGroupLibrary:
    def __init__(self):
        self.calls = []

    def flagcxGroupStart(self):
        self.calls.append(("start",))

    def flagcxGroupEnd(self):
        self.calls.append(("end",))


@pytest.mark.parametrize("device_type", ["txda", "npu", "cuda"])
def test_packaged_wrapper_groups_receive_the_communicator(group_methods, device_type):
    library = CommunicatorGroupLibrary()
    native_comm = object()
    comm = group_methods()
    comm.__dict__.update(
        device=SimpleNamespace(type=device_type),
        flagcx=library,
        comm=native_comm,
        disabled=False,
        _legacy_unique_id_api=False,
    )

    comm.group_start()
    comm.group_end()

    assert library.calls == [("start", native_comm), ("end", native_comm)]


@pytest.mark.parametrize("device_type", ["txda", "npu", "cuda"])
def test_legacy_wrapper_keeps_its_backend_signature(group_methods, device_type):
    library = (
        CommunicatorGroupLibrary() if device_type == "npu" else LegacyGroupLibrary()
    )
    native_comm = object()
    comm = group_methods()
    comm.__dict__.update(
        device=SimpleNamespace(type=device_type),
        flagcx=library,
        comm=native_comm,
        disabled=False,
        _legacy_unique_id_api=True,
    )

    comm.group_start()
    comm.group_end()

    expected = (
        [("start", native_comm), ("end", native_comm)]
        if device_type == "npu"
        else [("start",), ("end",)]
    )
    assert library.calls == expected


def test_disabled_communicator_groups_are_noops_without_initialized_handles(
    group_methods,
):
    # A single-rank communicator returns before setting device, flagcx, or
    # the wrapper ABI flag. It must still safely accept group operations.
    comm = group_methods()
    comm.disabled = True

    assert comm.group_start() is None
    assert comm.group_end() is None

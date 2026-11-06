# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for optional MetaX FlagCX device communication."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _compile(nodes, path, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "vendor,device,backend,expected",
    [
        ("metax", "cuda", "flagcx", "flagcx"),
        ("metax", "cuda", "nccl", "nccl"),
        ("metax", "cuda", "gloo", "gloo"),
        ("nvidia", "cuda", "flagcx", "flagcx"),
        ("iluvatar", "cuda", "flagcx", "flagcx"),
        ("ascend", "npu", "flagcx", "hccl"),
        ("sunrise", "ptpu", "flagcx", "pccl"),
        ("sunrise", "ptpu", "nccl", "pccl"),
        ("mthreads", "musa", "flagcx", "flagcx"),
    ],
)
def test_optional_flagcx_preserves_process_group_policy(
    vendor, device, backend, expected
):
    path = ROOT / "vllm_fl/worker/worker.py"
    function = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_normalize_process_group_backend"
    )
    namespace = _compile(
        [function],
        path,
        {"current_platform": SimpleNamespace(vendor_name=vendor, device_type=device)},
    )
    assert namespace[function.name](backend) == expected


@pytest.mark.parametrize(
    "vendor,device,disabled,must_raise",
    [
        ("metax", "cuda", True, True),
        ("metax", "cuda", False, False),
        ("nvidia", "cuda", True, False),
        ("iluvatar", "cuda", True, False),
        ("ascend", "npu", True, True),
        ("ascend", "npu", False, False),
    ],
)
def test_requested_metax_flagcx_cannot_silently_fall_back(
    monkeypatch, vendor, device, disabled, must_raise
):
    path = ROOT / "vllm_fl/distributed/communicator.py"
    cls = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "CommunicatorFL"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]

    class Base:
        def __init__(self, cpu_group, device, *args, use_all2all=False):
            self.cpu_group, self.device = cpu_group, device
            self.world_size, self.use_all2all = 2, use_all2all

    init_error = OSError("FlagCX library unavailable")
    native = SimpleNamespace(disabled=disabled, init_error=init_error)
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(vendor_name=vendor)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    namespace = _compile(
        [cls],
        path,
        {
            "DeviceCommunicatorBase": Base,
            "PyFlagcxCommunicator": lambda **kwargs: native,
        },
    )
    constructor = namespace[cls.name]
    if must_raise:
        with pytest.raises(RuntimeError, match="FlagCX requested") as error:
            constructor(object(), SimpleNamespace(type=device))
        assert error.value.__cause__ is init_error
    else:
        communicator = constructor(object(), SimpleNamespace(type=device))
        assert communicator.pyflagcx_comm is native


@pytest.mark.parametrize(
    "method,native_name",
    [("group_start", "flagcxGroupStart"), ("group_end", "flagcxGroupEnd")],
)
@pytest.mark.parametrize(
    "device,legacy,with_comm",
    [
        ("cuda", True, False),
        ("cuda", False, True),
        ("cuda", None, False),
        ("npu", True, True),
        ("npu", False, True),
    ],
)
def test_flagcx_group_adapts_known_wrapper_abi(
    method, native_name, device, legacy, with_comm
):
    path = ROOT / "vllm_fl/distributed/device_communicators/flagcx.py"
    cls = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "PyFlagcxCommunicator"
    )
    function = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    namespace = _compile([function], path, {})
    native = Mock()
    handle = object()
    communicator = SimpleNamespace(
        device=SimpleNamespace(type=device),
        comm=handle,
        flagcx=SimpleNamespace(**{native_name: native}),
    )
    if legacy is not None:
        communicator._legacy_unique_id_api = legacy
    namespace[method](communicator)
    native.assert_called_once_with(*((handle,) if with_comm else ()))


@pytest.mark.parametrize(
    "vendor,device,backend,opaque,native_custom_ar",
    [
        ("metax", "cuda", "flagcx", True, False),
        ("metax", "cuda", "nccl", False, True),
        ("ascend", "npu", "flagcx", False, False),
        ("nvidia", "cuda", "flagcx", True, False),
        ("iluvatar", "cuda", "nccl", True, True),
        ("thead", "cuda", "nccl", True, True),
        ("tsingmicro", "txda", "flagcx", True, False),
        ("mthreads", "musa", "flagcx", False, False),
        ("hygon", "cuda", "nccl", False, False),
    ],
)
def test_only_metax_flagcx_adds_opaque_collective_boundary(
    vendor, device, backend, opaque, native_custom_ar
):
    path = ROOT / "vllm_fl/platform.py"
    cls = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("use_custom_op_collectives", "use_custom_allreduce")
    ]
    for method in methods:
        method.decorator_list = []
    namespace = _compile(methods, path, {})
    platform = SimpleNamespace(
        vendor_name=vendor, device_type=device, dist_backend=backend
    )
    assert namespace["use_custom_op_collectives"](platform) is opaque
    assert namespace["use_custom_allreduce"](platform) is native_custom_ar


@pytest.mark.parametrize(
    "vendor,backend,v2,expected",
    [
        (
            "metax",
            "flagcx",
            "1",
            "vllm_fl.distributed.metax_communicator.MetaxCommunicatorFL",
        ),
        ("metax", "flagcx", "0", "vllm_fl.distributed.communicator.CommunicatorFL"),
        ("metax", "flagcx", None, "vllm_fl.distributed.communicator.CommunicatorFL"),
        (
            "metax",
            "nccl",
            "1",
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator",
        ),
        ("nvidia", "flagcx", "1", "vllm_fl.distributed.communicator.CommunicatorFL"),
        ("mthreads", "flagcx", "1", "vllm_fl.distributed.communicator.CommunicatorFL"),
        ("ascend", "flagcx", "1", "vllm_fl.distributed.communicator.CommunicatorFL"),
        ("iluvatar", "flagcx", "1", "vllm_fl.distributed.communicator.CommunicatorFL"),
    ],
)
def test_only_explicit_metax_v2_flagcx_selects_cuda_graph_bridge(
    vendor, backend, v2, expected
):
    path = ROOT / "vllm_fl/platform.py"
    cls = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_device_communicator_cls"
    )
    method.decorator_list = []
    env = {} if v2 is None else {"VLLM_USE_V2_MODEL_RUNNER": v2}
    namespace = _compile(
        [method], path, {"os": SimpleNamespace(environ=env), "logger": Mock()}
    )
    platform = SimpleNamespace(vendor_name=vendor, dist_backend=backend)
    assert namespace[method.name](platform) == expected

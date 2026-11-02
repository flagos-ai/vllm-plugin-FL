# Copyright (c) 2025 BAAI. All rights reserved.

"""Behavioral contract tests for the vLLM 0.28 FlagCX communicator adapter."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm")

from vllm.distributed.device_communicators.base_device_communicator import (  # noqa: E402
    DeviceCommunicatorBase,
)
from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    CudaCommunicator,
)

from vllm_fl.distributed.communicator import CommunicatorFL  # noqa: E402
from vllm_fl.distributed.device_communicators import flagcx  # noqa: E402
from vllm_fl.distributed.nvidia_communicator import (  # noqa: E402
    NvidiaCommunicatorFL,
)


@pytest.mark.parametrize("device_type", ["npu", "gcu"])
def test_flagcx_initialization_uses_native_device_context(monkeypatch, device_type):
    class Device:
        type = device_type

    class Group:
        rank = 0
        world_size = 2

        def broadcast_obj(self, value, src):
            assert src == 0
            return value

    device = Device()
    stream = MagicMock()
    backend = SimpleNamespace(
        device=MagicMock(), current_stream=MagicMock(return_value=stream)
    )
    cuda_device = MagicMock(side_effect=AssertionError("native CUDA context used"))
    fake_torch = SimpleNamespace(
        device=Device,
        cuda=SimpleNamespace(device=cuda_device),
        zeros=MagicMock(),
        **{device_type: backend},
    )
    library = MagicMock()
    monkeypatch.setattr(flagcx, "torch", fake_torch)
    monkeypatch.setattr(flagcx, "StatelessProcessGroup", Group)
    monkeypatch.setattr(flagcx, "FLAGCXLibrary", MagicMock(return_value=library))
    warmup = MagicMock()
    monkeypatch.setattr(flagcx.PyFlagcxCommunicator, "all_reduce", warmup)

    communicator = flagcx.PyFlagcxCommunicator(Group(), device, "/test/libflagcx.so")

    assert communicator.available and not communicator.disabled
    backend.device.assert_called_once_with(device)
    backend.device.return_value.__enter__.assert_called_once()
    backend.current_stream.assert_called_once_with(device)
    stream.synchronize.assert_called_once()
    warmup.assert_called_once()
    cuda_device.assert_not_called()


@pytest.mark.parametrize("device_type", ["npu", "gcu"])
def test_flagcx_uses_current_native_stream_handle(monkeypatch, device_type):
    streams = [
        SimpleNamespace(**{f"{device_type}_stream": handle}) for handle in (11, 29)
    ]
    backend = SimpleNamespace(current_stream=MagicMock(side_effect=streams))
    monkeypatch.setattr(flagcx, "torch", SimpleNamespace(**{device_type: backend}))
    monkeypatch.setattr(
        flagcx,
        "current_stream",
        MagicMock(side_effect=AssertionError("CUDA stream used")),
    )
    communicator = flagcx.PyFlagcxCommunicator.__new__(flagcx.PyFlagcxCommunicator)
    communicator.device = SimpleNamespace(type=device_type)
    handles = []
    communicator.flagcx = SimpleNamespace(
        adaptor_stream_copy=lambda stream: handles.append(stream.cuda_stream)
    )

    for _ in streams:
        communicator._copy_flagcx_stream(communicator._current_stream())

    assert handles == [11, 29]
    assert backend.current_stream.call_count == 2


@pytest.mark.parametrize("vendor", ["enflame", "ascend"])
def test_flagcx_collectives_opt_into_native_opaque_ops(monkeypatch, vendor):
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(PlatformFL, "dist_backend", "flagcx")

    assert PlatformFL.use_custom_op_collectives() is True


@pytest.mark.parametrize(
    ("vendor", "backend", "expected"),
    [
        ("nvidia", "nccl", True),
        ("thead", "tccl", True),
        ("iluvatar", "nccl", True),
        ("enflame", "eccl", False),
        ("ascend", "hccl", False),
    ],
)
def test_non_flagcx_collectives_preserve_platform_policy(
    monkeypatch, vendor, backend, expected
):
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(PlatformFL, "dist_backend", backend)

    assert PlatformFL.use_custom_op_collectives() is expected


@pytest.fixture
def mocked_base_init(monkeypatch):
    received = {}

    def fake_base_init(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name="",
        global_ranks=None,
        global_world_size=None,
        use_all2all=False,
    ):
        received.update(
            cpu_group=cpu_group,
            device=device,
            device_group=device_group,
            unique_name=unique_name,
            global_ranks=global_ranks,
            global_world_size=global_world_size,
            use_all2all=use_all2all,
        )
        self.cpu_group = cpu_group
        self.device = device
        self.device_group = device_group
        self.world_size = 1
        self.use_all2all = use_all2all
        self.all2all_backend = "allgather_reducescatter"
        self.all2all_manager = None

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base_init)
    return received


def test_constructor_forwards_vllm_028_arguments(
    monkeypatch,
    mocked_base_init,
):
    manager_init = []
    monkeypatch.setattr(
        CommunicatorFL,
        "_init_all2all_manager",
        lambda self, group: manager_init.append(group),
    )

    cpu_group = object()
    device = object()
    device_group = object()
    tcp_store_group = object()
    communicator = CommunicatorFL(
        cpu_group=cpu_group,
        device=device,
        device_group=device_group,
        unique_name="ep:0",
        global_ranks=[3, 7],
        global_world_size=8,
        tcp_store_group=tcp_store_group,
        use_all2all=True,
    )

    assert communicator.pyflagcx_comm is None
    assert mocked_base_init == {
        "cpu_group": cpu_group,
        "device": device,
        "device_group": device_group,
        "unique_name": "ep:0",
        "global_ranks": [3, 7],
        "global_world_size": 8,
        "use_all2all": True,
    }
    assert manager_init == [tcp_store_group]


@pytest.mark.parametrize("backend", ["naive", "allgather_reducescatter"])
def test_default_all2all_backend_uses_agrs_manager(monkeypatch, backend):
    from vllm.distributed.device_communicators import all2all

    manager = MagicMock()
    manager_factory = MagicMock(return_value=manager)
    monkeypatch.setattr(all2all, "AgRsAll2AllManager", manager_factory)

    communicator = CommunicatorFL.__new__(CommunicatorFL)
    communicator.all2all_backend = backend
    communicator.cpu_group = object()
    tcp_store_group = object()

    communicator._init_all2all_manager(tcp_store_group)

    manager_factory.assert_called_once_with(
        communicator.cpu_group,
        tcp_store_group,
    )
    assert communicator.all2all_manager is manager


@pytest.mark.parametrize(
    ("method_name", "arguments"),
    [
        (
            "dispatch_router_logits",
            (object(), object(), True, [object()]),
        ),
        (
            "dispatch",
            (object(), object(), object(), True, [object()]),
        ),
        (
            "combine",
            (object(), True),
        ),
    ],
)
def test_all2all_calls_use_vllm_028_signatures(method_name, arguments):
    communicator = CommunicatorFL.__new__(CommunicatorFL)
    communicator.all2all_manager = MagicMock()
    expected = object()
    manager_method = getattr(communicator.all2all_manager, method_name)
    manager_method.return_value = expected

    result = getattr(communicator, method_name)(*arguments)

    manager_method.assert_called_once_with(*arguments)
    assert result is expected


def test_nvidia_bridge_satisfies_cuda_graph_contract(mocked_base_init):
    communicator = NvidiaCommunicatorFL(cpu_group=object())

    assert isinstance(communicator, CudaCommunicator)
    assert communicator.ca_comm is None
    assert communicator.pynccl_comm is None
    assert communicator.pyflagcx_comm is None
    assert NvidiaCommunicatorFL.all_reduce is CommunicatorFL.all_reduce
    assert NvidiaCommunicatorFL.all_gather is CommunicatorFL.all_gather
    assert NvidiaCommunicatorFL.broadcast is CommunicatorFL.broadcast


def test_nvidia_bridge_enters_native_graph_capture(monkeypatch, mocked_base_init):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    from vllm.distributed.parallel_state import GroupCoordinator

    communicator = NvidiaCommunicatorFL(cpu_group=object())
    coordinator = SimpleNamespace(device_communicator=communicator)
    stream = object()
    capture_context = SimpleNamespace(stream=stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())

    # Exercise upstream's attribute reads instead of copying its expected
    # field list. vLLM 0.32 also consults fi_pcie_ipc_ar_comm during capture.
    with GroupCoordinator.graph_capture(coordinator, capture_context) as context:
        assert context is capture_context
    assert communicator.pyflagcx_comm is None

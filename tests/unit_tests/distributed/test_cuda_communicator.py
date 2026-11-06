from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_fl.distributed.communicator import CommunicatorFL
from vllm_fl.distributed.cuda_communicator import CudaCommunicatorFL
from vllm_fl.platform import PlatformFL


@pytest.mark.parametrize("device_type", ["cuda", "npu", "musa"])
def test_flagcx_platform_selection(monkeypatch, device_type):
    monkeypatch.setattr(PlatformFL, "device_type", device_type)
    monkeypatch.setattr(PlatformFL, "dist_backend", "flagcx")
    selected = PlatformFL.get_device_communicator_cls()
    if device_type == "cuda":
        assert selected == "vllm_fl.distributed.cuda_communicator.CudaCommunicatorFL"
    else:
        assert selected == "vllm_fl.distributed.communicator.CommunicatorFL"


def test_cuda_bridge_graph_contract(monkeypatch):
    def initialize(self, **kwargs):
        self.pyflagcx_comm = object()

    monkeypatch.setattr(CommunicatorFL, "__init__", initialize)
    native_init = MagicMock(side_effect=AssertionError("must not initialize PyNCCL"))
    monkeypatch.setattr(CudaCommunicator, "__init__", native_init)
    communicator = CudaCommunicatorFL(cpu_group=object())
    assert isinstance(communicator, CudaCommunicator)
    assert communicator.pyflagcx_comm is not None
    assert communicator.use_aiter_ag_rs is False
    stream = object()
    context = SimpleNamespace(stream=stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    with GroupCoordinator.graph_capture(
        SimpleNamespace(device_communicator=communicator), context
    ) as result:
        assert result is context
    native_init.assert_not_called()
    for method in (
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_gatherv",
        "send",
        "recv",
        "destroy",
    ):
        assert getattr(CudaCommunicatorFL, method) is getattr(CommunicatorFL, method)

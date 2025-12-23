import ctypes
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
from torch.distributed import ReduceOp
from vllm.distributed.utils import StatelessProcessGroup

fake_wrapper = types.ModuleType("plugin.interservice.flagcx_wrapper")


class FakeUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 128)]


class FakeFLAGCXLibrary:
    def __init__(self, *args, **kwargs):
        self.flagcxGetUniqueId = MagicMock(return_value=ctypes.pointer(FakeUniqueId()))
        self.flagcxCommInitRank = MagicMock(return_value="fake_comm")

        # collectives
        self.flagcxAllReduce = MagicMock()
        self.flagcxAllGather = MagicMock()
        self.flagcxReduceScatter = MagicMock()
        self.flagcxReduce = MagicMock()
        self.flagcxBroadcast = MagicMock()

        # p2p
        self.flagcxSend = MagicMock()
        self.flagcxRecv = MagicMock()

        # group
        self.flagcxGroupStart = MagicMock()
        self.flagcxGroupEnd = MagicMock()

        # stream adaptor
        self.adaptor_stream_copy = MagicMock(return_value="fake_stream")
        self.adaptor_stream_free = MagicMock()


fake_wrapper.FLAGCXLibrary = FakeFLAGCXLibrary
fake_wrapper.buffer_type = lambda *args, **kwargs: None
fake_wrapper.cudaStream_t = ctypes.c_void_p
fake_wrapper.flagcxComm_t = ctypes.c_void_p
fake_wrapper.flagcxUniqueId = FakeUniqueId

fake_wrapper.flagcxDataTypeEnum = MagicMock()
fake_wrapper.flagcxDataTypeEnum.from_torch = MagicMock(return_value=0)

fake_wrapper.flagcxRedOpTypeEnum = MagicMock()
fake_wrapper.flagcxRedOpTypeEnum.from_torch = MagicMock(return_value=0)

sys.modules["plugin"] = types.ModuleType("plugin")
sys.modules["plugin.interservice"] = types.ModuleType("plugin.interservice")
sys.modules["plugin.interservice.flagcx_wrapper"] = fake_wrapper


class DummyStatelessProcessGroup(StatelessProcessGroup):
    def __init__(self, rank=0, world_size=2):
        self.rank = rank
        self.world_size = world_size

    def broadcast_obj(self, obj, src=0):
        return obj


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def group():
    return DummyStatelessProcessGroup(rank=0, world_size=2)


@pytest.fixture
def communicator(group, device):
    from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

    return PyFlagcxCommunicator(
        group=group,
        device=device,
        library_path="dummy.so",
    )


def test_world_size_one_disables(device):
    from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

    group = DummyStatelessProcessGroup(rank=0, world_size=1)
    comm = PyFlagcxCommunicator(group=group, device=device)

    assert comm.disabled
    assert not comm.available


def test_init(communicator):
    assert communicator.available
    assert communicator.comm == "fake_comm"
    communicator.flagcx.flagcxCommInitRank.assert_called_once()

    y = communicator.all_reduce(x, op=ReduceOp.SUM)

    assert y.shape == x.shape
    communicator.flagcx.flagcxAllReduce.assert_called()


def test_all_gather(communicator):
    inp = torch.ones(2, device=communicator.device)
    out = torch.empty(4, device=communicator.device)

    communicator.all_gather(out, inp)
    communicator.flagcx.flagcxAllGather.assert_called_once()


def test_all_gatherv(communicator):
    inp = torch.ones(2, device=communicator.device)
    out = torch.empty(4, device=communicator.device)

    communicator.all_gatherv(out, inp, sizes=[2, 2])

    communicator.flagcx.flagcxGroupStart.assert_called_once()
    communicator.flagcx.flagcxBroadcast.assert_called()
    communicator.flagcx.flagcxGroupEnd.assert_called_once()


def test_reduce_scatter(communicator):
    inp = torch.ones(4, device=communicator.device)
    out = torch.empty(2, device=communicator.device)

    communicator.reduce_scatter(out, inp)
    communicator.flagcx.flagcxReduceScatter.assert_called_once()


def test_reduce_scatterv(communicator):
    inp = torch.ones(4, device=communicator.device)
    out = torch.empty(2, device=communicator.device)

    communicator.reduce_scatterv(out, inp, sizes=[2, 2])

    communicator.flagcx.flagcxGroupStart.assert_called_once()
    communicator.flagcx.flagcxReduce.assert_called()
    communicator.flagcx.flagcxGroupEnd.assert_called_once()


def test_broadcast_root(communicator):
    t = torch.ones(4, device=communicator.device)
    communicator.broadcast(t, src=communicator.rank)

    communicator.flagcx.flagcxBroadcast.assert_called_once()


def test_broadcast_non_root(communicator):
    t = torch.ones(4, device=communicator.device)
    communicator.broadcast(t, src=1)

    communicator.flagcx.flagcxBroadcast.assert_called_once()


def test_send_recv(communicator):
    t = torch.ones(4, device=communicator.device)

    communicator.send(t, dst=1)
    communicator.recv(t, src=1)

    communicator.flagcx.flagcxSend.assert_called_once()
    communicator.flagcx.flagcxRecv.assert_called_once()


def test_group_start_end(communicator):
    communicator.group_start()
    communicator.group_end()

    communicator.flagcx.flagcxGroupStart.assert_called()
    communicator.flagcx.flagcxGroupEnd.assert_called()


def test_device_mismatch_raises(communicator):
    cpu_tensor = torch.ones(2)

    with pytest.raises(AssertionError):
        communicator.all_reduce(cpu_tensor)

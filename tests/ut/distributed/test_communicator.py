import sys
from unittest.mock import MagicMock

import pytest
import torch

sys.modules["plugin"] = MagicMock()
sys.modules["plugin.interservice"] = MagicMock()
sys.modules["plugin.interservice.flagcx_wrapper"] = MagicMock()

from vllm_fl.distributed.communicator import CommunicatorFL


class TestCommunicatorFL(CommunicatorFL):
    def __init__(self):
        self.world_size = 2
        self.rank_in_group = 0
        self.device_group = None
        self.all2all_manager = MagicMock()
        self.pyflagcx_comm = MagicMock()
        self.ranks = [0, 1]

        self.pyflagcx_comm.disabled = False

        self.all_reduce = MagicMock(side_effect=lambda x: x.clone())
        self.reduce_scatter = MagicMock(side_effect=lambda x, dim=-1: x[:1])
        self.reduce_scatterv = MagicMock(
            side_effect=lambda x, dim=-1, sizes=None: x[:1]
        )
        self.send = MagicMock()
        self.recv = MagicMock(
            side_effect=lambda size, dtype, src=None: torch.zeros(
                size, dtype=dtype, device="cuda"
            )
        )
        self.destroy = MagicMock()
        self.dispatch = MagicMock(side_effect=lambda h, l, is_seq=False: (h, l))
        self.combine = MagicMock(side_effect=lambda h, is_seq=False: h)


@pytest.fixture
def communicator():
    comm = TestCommunicatorFL()
    yield comm


def test_all_reduce(communicator):
    x = torch.tensor([1.0, 2.0], device="cuda")
    out = communicator.all_reduce(x)
    assert torch.equal(out, x)


def test_reduce_scatter(communicator):
    x = torch.arange(4.0, device="cuda")
    result = communicator.reduce_scatter(x, dim=0)
    assert result.numel() == 1


def test_reduce_scatterv(communicator):
    x = torch.arange(4.0, device="cuda")
    result = communicator.reduce_scatterv(x, sizes=[2, 2])
    assert result.numel() == 1


def test_send_recv(communicator):
    communicator.send(torch.tensor([1, 2], device="cuda"))
    out = communicator.recv(size=(2,), dtype=torch.int)
    assert torch.equal(out, torch.zeros(2, dtype=torch.int, device="cuda"))


def test_destroy(communicator):
    communicator.destroy()
    communicator.destroy.assert_called()


def test_dispatch_combine(communicator):
    h = torch.tensor([1.0, 2.0], device="cuda")
    l = torch.tensor([0.5, 0.5], device="cuda")
    h_out, l_out = communicator.dispatch(h, l)
    assert torch.equal(h_out, h)
    assert torch.equal(l_out, l)

    h2 = communicator.combine(h)
    assert torch.equal(h2, h)

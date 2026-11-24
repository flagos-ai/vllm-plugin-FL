# Copyright (c) 2025 BAAI. All rights reserved.
"""Residual CPU branches for ``vllm_fl.distributed.communicator``.

The FlagCX peer and the base class are replaced; tensor math is real.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

import vllm_fl.distributed.communicator as cm


def make(**attrs):
    obj = cm.CommunicatorFL.__new__(cm.CommunicatorFL)
    obj.pyflagcx_comm = None
    obj.all2all_manager = None
    obj.world_size = 2
    obj.rank_in_group = 0
    obj.device = torch.device("cpu")
    obj.device_group = "DG"
    obj.ranks = ["r0", "r1"]
    obj.cpu_group = "CPU"
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def flagcx(**methods):
    obj = SimpleNamespace(disabled=False)
    obj.__dict__.update(methods)
    return obj


def test_destroy_is_idempotent_without_children():
    comm = make()
    comm.destroy()
    assert comm.pyflagcx_comm is None
    assert comm.all2all_manager is None


def test_destroy_drops_non_txda_flagcx_but_keeps_no_manager():
    destroy = Mock()
    comm = make(pyflagcx_comm=flagcx(destroy=destroy))
    comm.destroy()
    destroy.assert_not_called()
    assert comm.pyflagcx_comm is None


def test_all_gather_dim_zero_keeps_leading_axis():
    seen = {}

    def fake(output, inp):
        seen["out"] = output
        output.copy_(torch.arange(output.numel(), dtype=output.dtype).reshape(output.shape))

    comm = make()
    comm.pyflagcx_comm = flagcx(all_gather=Mock(side_effect=fake))
    inp = torch.zeros(2, 3)
    out = comm.all_gather(inp, dim=0)
    assert seen["out"].shape == (4, 3)
    assert out.shape == (4, 3)


def test_all_gather_positive_dim_no_conversion():
    seen = {}

    def fake(output, inp):
        seen["out"] = output
        output.copy_(torch.zeros(output.shape))

    comm = make()
    comm.pyflagcx_comm = flagcx(all_gather=Mock(side_effect=fake))
    inp = torch.zeros(2, 2)
    out = comm.all_gather(inp, dim=1)
    assert seen["out"].shape == (4, 2)
    assert out.shape == (2, 4)


def test_reduce_scatter_explicit_dim_zero():
    comm = make()
    comm.pyflagcx_comm = flagcx(reduce_scatter=Mock())
    inp = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    out = comm.reduce_scatter(inp, dim=0)
    assert out.shape == (1, 5)
    assert comm.pyflagcx_comm.reduce_scatter.call_args.args[1].shape == (2, 5)


def test_reduce_scatter_three_dim_negative_axis():
    comm = make()
    comm.pyflagcx_comm = flagcx(reduce_scatter=Mock())
    inp = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    out = comm.reduce_scatter(inp, dim=-1)
    assert out.shape == (3, 2, 2)
    assert comm.pyflagcx_comm.reduce_scatter.call_args.args[1].shape == (4, 3, 2)


def test_all_gatherv_tensor_without_sizes():
    comm = make()
    comm.pyflagcx_comm = flagcx(all_gather=Mock(), all_gatherv=Mock())
    out = comm.all_gatherv(torch.ones(3))
    assert out.shape == (6,)
    comm.pyflagcx_comm.all_gather.assert_called_once()
    comm.pyflagcx_comm.all_gatherv.assert_not_called()


def test_all_gatherv_uniform_list_sizes_falls_back_to_all_gather():
    comm = make()
    comm.pyflagcx_comm = flagcx(
        all_gather=Mock(), all_gatherv=Mock(), group_start=Mock(), group_end=Mock()
    )
    inputs = [torch.ones(2), torch.ones(2)]
    out = comm.all_gatherv(inputs, sizes=[2, 2])
    assert [tensor.shape for tensor in out] == [(4,), (4,)]
    assert comm.pyflagcx_comm.all_gather.call_count == 2
    comm.pyflagcx_comm.all_gatherv.assert_not_called()


def test_all_gatherv_nonuniform_list_validates_each_rank_size():
    comm = make(rank_in_group=0)
    comm.pyflagcx_comm = flagcx(
        all_gather=Mock(), all_gatherv=Mock(), group_start=Mock(), group_end=Mock()
    )
    out = comm.all_gatherv([torch.ones(2), torch.ones(2)], sizes=[2, 4])
    assert [tensor.shape for tensor in out] == [(6,), (6,)]
    assert comm.pyflagcx_comm.all_gatherv.call_count == 2

    with pytest.raises(AssertionError):
        comm.all_gatherv([torch.ones(4), torch.ones(4)], sizes=[2, 4])
    comm.pyflagcx_comm.group_end.assert_called_once()


def test_all_gatherv_rejects_other_dims_before_comm_check():
    comm = make()
    with pytest.raises(NotImplementedError):
        comm.all_gatherv(torch.ones(2), dim=1)


def test_send_default_dst_wraps_at_end_of_group():
    comm = make(rank_in_group=1)
    comm.pyflagcx_comm = flagcx(send=Mock())
    tensor = torch.ones(2)
    comm.send(tensor)
    comm.pyflagcx_comm.send.assert_called_once_with(tensor, 0)


def test_recv_default_src_wraps_at_group_start():
    comm = make(rank_in_group=0)
    comm.pyflagcx_comm = flagcx(recv=Mock())
    out = comm.recv(torch.Size([2]), torch.float32)
    comm.pyflagcx_comm.recv.assert_called_once_with(out, 1)


def test_broadcast_without_flagcx_delegates_to_base(monkeypatch):
    marker = object()
    seen = []
    monkeypatch.setattr(
        DeviceCommunicatorBase,
        "broadcast",
        lambda self, tensor, src: seen.append((tensor, src)) or marker,
    )
    comm = make(pyflagcx_comm=None)
    tensor = torch.ones(2)
    assert comm.broadcast(tensor, 0) is marker
    assert seen == [(tensor, 0)]


def test_checkpoint_hooks_noop_without_manager():
    comm = make()
    assert comm.checkpoint_prepare() is None
    assert comm.checkpoint_restore() is None


def test_dispatch_forwards_extra_tensors():
    comm = make(all2all_manager=MagicMock())
    args = (object(), object(), object(), False, None)
    comm.dispatch(*args)
    comm.all2all_manager.dispatch.assert_called_once_with(*args)


@pytest.fixture
def base_init(monkeypatch):
    def fake(self, cpu_group, device=None, device_group=None, unique_name="",
             global_ranks=None, global_world_size=None, use_all2all=False):
        self.cpu_group = cpu_group
        self.device = device
        self.device_group = device_group
        self.world_size = 1
        self.use_all2all = use_all2all
        self.all2all_backend = "allgather_reducescatter"
        self.all2all_manager = None
        self.rank_in_group = 0
        self.ranks = [0]

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake)


def test_world_one_with_all2all_still_builds_manager(base_init, monkeypatch):
    seen = []
    monkeypatch.setattr(
        cm.CommunicatorFL,
        "_init_all2all_manager",
        lambda self, group: seen.append(group),
    )
    tcp = object()
    comm = cm.CommunicatorFL(
        cpu_group=object(), device=torch.device("cpu"),
        tcp_store_group=tcp, use_all2all=True,
    )
    assert comm.pyflagcx_comm is None
    assert seen == [tcp]


@pytest.fixture
def multi_init(monkeypatch):
    def fake(self, cpu_group, device=None, device_group=None, unique_name="",
             global_ranks=None, global_world_size=None, use_all2all=False):
        self.cpu_group = cpu_group
        self.device = device
        self.device_group = device_group
        self.world_size = 2
        self.use_all2all = use_all2all
        self.all2all_backend = "allgather_reducescatter"
        self.all2all_manager = None
        self.rank_in_group = 0
        self.ranks = [0, 1]

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake)


def test_npu_without_vendor_name_reports_npu(multi_init, monkeypatch):
    monkeypatch.setattr(
        cm, "PyFlagcxCommunicator",
        Mock(return_value=SimpleNamespace(disabled=True, init_error=OSError("x"))),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace()),
    )
    with pytest.raises(RuntimeError, match="NPU"):
        cm.CommunicatorFL(
            cpu_group=object(), device=SimpleNamespace(type="npu")
        )


def test_amd_vendor_with_non_npu_device_reports_amd(multi_init, monkeypatch):
    monkeypatch.setattr(
        cm, "PyFlagcxCommunicator",
        Mock(return_value=SimpleNamespace(disabled=True, init_error=OSError("x"))),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace(vendor_name="amd")),
    )
    with pytest.raises(RuntimeError, match="AMD"):
        cm.CommunicatorFL(cpu_group=object(), device=torch.device("cpu"))

# Copyright (c) 2025 BAAI. All rights reserved.
"""CPU branch tests for vllm_fl.distributed.communicator.CommunicatorFL."""

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


# ---------------------------------------------------------------------------
# all_reduce
# ---------------------------------------------------------------------------
def test_all_reduce_requires_flagcx_comm():
    with pytest.raises(AssertionError):
        make().all_reduce(torch.ones(2))


def test_all_reduce_returns_flagcx_result():
    comm = make()
    token = object()
    comm.pyflagcx_comm = flagcx(all_reduce=Mock(return_value=token))
    assert comm.all_reduce(torch.ones(2)) is token


def test_all_reduce_falls_back_when_flagcx_returns_none(monkeypatch):
    seen = []
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, group=None: seen.append((tensor, group)),
    )
    comm = make()
    comm.pyflagcx_comm = flagcx(all_reduce=Mock(return_value=None))
    src = torch.arange(3, dtype=torch.float32)
    out = comm.all_reduce(src)
    assert out is not src
    assert torch.equal(out, src)
    assert seen and seen[0][1] == "DG"


# ---------------------------------------------------------------------------
# all_gather
# ---------------------------------------------------------------------------
def test_all_gather_world_size_one_returns_input():
    comm = make(world_size=1)
    tensor = torch.ones(2)
    assert comm.all_gather(tensor) is tensor


def test_all_gather_falls_back_when_unavailable(monkeypatch):
    marker = object()
    calls = []

    def fake_base(self, input_, dim):
        calls.append((input_, dim))
        return marker

    monkeypatch.setattr(DeviceCommunicatorBase, "all_gather", fake_base)
    comm = make()
    tensor = torch.ones(2, 3)
    assert comm.all_gather(tensor, dim=-1) is marker
    assert len(calls) == 1
    assert calls[0][0] is tensor and calls[0][1] == 1

    comm.pyflagcx_comm = flagcx(disabled=True)
    assert comm.all_gather(tensor) is marker


def test_all_gather_flagcx_path_shapes_1d():
    def fake_all_gather(output, inp):
        output.copy_(torch.arange(output.numel(), dtype=output.dtype))

    comm = make()
    comm.pyflagcx_comm = flagcx(all_gather=Mock(side_effect=fake_all_gather))
    out = comm.all_gather(torch.zeros(2))
    assert out.shape == (4,)
    assert torch.equal(out, torch.arange(4, dtype=out.dtype))


def test_all_gather_flagcx_path_preserves_dim_layout():
    seen = {}

    def fake_all_gather(output, inp):
        seen["output"] = output
        seen["input"] = inp
        output.copy_(torch.arange(output.numel(), dtype=output.dtype).reshape(output.shape))

    comm = make()
    comm.pyflagcx_comm = flagcx(all_gather=Mock(side_effect=fake_all_gather))
    inp = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = comm.all_gather(inp, dim=1)
    assert out.shape == (2, 6)
    assert seen["output"].shape == (4, 3)
    assert torch.equal(seen["input"], inp)


# ---------------------------------------------------------------------------
# reduce_scatter / reduce_scatterv
# ---------------------------------------------------------------------------
def test_reduce_scatter_requires_comm():
    with pytest.raises(AssertionError):
        make().reduce_scatter(torch.ones(4))


def test_reduce_scatter_1d_roundtrip():
    comm = make()
    comm.pyflagcx_comm = flagcx(reduce_scatter=Mock())
    inp = torch.arange(4, dtype=torch.float32)
    out = comm.reduce_scatter(inp)
    assert out.shape == (2,)
    call_arg = comm.pyflagcx_comm.reduce_scatter.call_args.args
    assert call_arg[0].shape == (2,)
    assert call_arg[1].shape == (4,)


def test_reduce_scatter_rejects_indivisible_size():
    comm = make()
    comm.pyflagcx_comm = flagcx(reduce_scatter=Mock())
    with pytest.raises(AssertionError):
        comm.reduce_scatter(torch.ones(5))


def test_reduce_scatter_negative_dim_moves_axis():
    comm = make()
    comm.pyflagcx_comm = flagcx(reduce_scatter=Mock())
    inp = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = comm.reduce_scatter(inp, dim=-1)
    # world 2 splits the last axis (4 -> 2), output is (3, 2)
    assert out.shape == (3, 2)


def test_reduce_scatterv_sizes_and_default_paths():
    comm = make()
    comm.rank_in_group = 1
    comm.pyflagcx_comm = flagcx(reduce_scatter=Mock(), reduce_scatterv=Mock())
    inp = torch.arange(6, dtype=torch.float32)

    out = comm.reduce_scatterv(inp, sizes=[2, 4])
    assert out.shape == (4,)
    comm.pyflagcx_comm.reduce_scatterv.assert_called_once()
    assert comm.pyflagcx_comm.reduce_scatterv.call_args.kwargs["sizes"] == [2, 4]

    out = comm.reduce_scatterv(inp)
    assert out.shape == (3,)
    comm.pyflagcx_comm.reduce_scatter.assert_called_once()


def test_reduce_scatterv_validates_sizes():
    comm = make()
    comm.pyflagcx_comm = flagcx(reduce_scatter=Mock(), reduce_scatterv=Mock())
    inp = torch.ones(6)
    with pytest.raises(AssertionError):
        comm.reduce_scatterv(inp, sizes=[1, 2])
    with pytest.raises(AssertionError):
        comm.reduce_scatterv(inp, sizes=[1, 2, 3])


# ---------------------------------------------------------------------------
# send / recv
# ---------------------------------------------------------------------------
def test_send_uses_flagcx_and_default_dst():
    comm = make()
    comm.rank_in_group = 0
    comm.pyflagcx_comm = flagcx(send=Mock())
    tensor = torch.ones(2)
    comm.send(tensor)
    comm.pyflagcx_comm.send.assert_called_once_with(tensor, 1)


def test_send_falls_back_to_torch(monkeypatch):
    seen = []
    monkeypatch.setattr(
        torch.distributed, "send", lambda tensor, dst, group=None: seen.append((tensor, dst, group))
    )
    comm = make()
    comm.pyflagcx_comm = flagcx(disabled=True)
    tensor = torch.ones(2)
    comm.send(tensor, dst=1)
    assert len(seen) == 1
    assert seen[0][0] is tensor and seen[0][1:] == ("r1", "DG")

    comm.pyflagcx_comm = None
    comm.send(tensor, dst=0)
    assert seen[-1][0] is tensor and seen[-1][1:] == ("r0", "DG")


def test_recv_uses_flagcx_and_default_src():
    comm = make()
    comm.rank_in_group = 0
    comm.pyflagcx_comm = flagcx(recv=Mock())
    out = comm.recv(torch.Size([2, 3]), torch.float32)
    assert out.shape == (2, 3) and out.device.type == "cpu"
    comm.pyflagcx_comm.recv.assert_called_once_with(out, 1)


def test_recv_falls_back_to_torch(monkeypatch):
    seen = []
    monkeypatch.setattr(
        torch.distributed, "recv", lambda tensor, src, group=None: seen.append((tensor, src, group))
    )
    comm = make()
    comm.rank_in_group = 1
    comm.pyflagcx_comm = flagcx(disabled=True)
    out = comm.recv(torch.Size([2]), torch.float32)
    assert len(seen) == 1
    assert seen[0][0] is out and seen[0][1:] == ("r0", "DG")


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------
def test_broadcast_world_size_one_returns_tensor():
    comm = make(world_size=1)
    tensor = torch.ones(2)
    assert comm.broadcast(tensor, 0) is tensor


def test_broadcast_uses_flagcx_or_falls_back(monkeypatch):
    comm = make()
    comm.pyflagcx_comm = flagcx(broadcast=Mock())
    tensor = torch.ones(2)
    assert comm.broadcast(tensor, 0) is tensor
    comm.pyflagcx_comm.broadcast.assert_called_once_with(tensor, 0)

    marker = object()
    calls = []
    monkeypatch.setattr(
        DeviceCommunicatorBase,
        "broadcast",
        lambda self, t, src: calls.append((t, src)) or marker,
    )
    comm.pyflagcx_comm = flagcx(disabled=True)
    assert comm.broadcast(tensor, 1) is marker
    assert len(calls) == 1
    assert calls[0][0] is tensor and calls[0][1] == 1


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
def test_destroy_txda_calls_flagcx_destroy():
    destroy = Mock()
    comm = make()
    comm.device = SimpleNamespace(type="txda")
    comm.pyflagcx_comm = flagcx(destroy=destroy)
    comm.all2all_manager = MagicMock()
    comm.destroy()
    destroy.assert_called_once_with()
    assert comm.pyflagcx_comm is None
    assert comm.all2all_manager is None


def test_destroy_non_txda_skips_native_destroy():
    destroy = Mock()
    comm = make()
    comm.pyflagcx_comm = flagcx(destroy=destroy)
    comm.destroy()
    destroy.assert_not_called()
    assert comm.pyflagcx_comm is None
    assert comm.all2all_manager is None


def test_checkpoint_hooks_forward_only_when_manager_present():
    comm = make()
    comm.checkpoint_prepare()
    comm.checkpoint_restore()
    comm.all2all_manager = MagicMock()
    comm.checkpoint_prepare()
    comm.checkpoint_restore()
    comm.all2all_manager.checkpoint_prepare.assert_called_once()
    comm.all2all_manager.checkpoint_restore.assert_called_once()


# ---------------------------------------------------------------------------
# all_gatherv
# ---------------------------------------------------------------------------
def test_all_gatherv_rejects_nonzero_dim():
    with pytest.raises(NotImplementedError):
        make().all_gatherv(torch.ones(2), dim=1)


def test_all_gatherv_requires_comm():
    with pytest.raises(AssertionError):
        make().all_gatherv(torch.ones(2))
    comm = make()
    comm.pyflagcx_comm = flagcx(disabled=True)
    with pytest.raises(AssertionError):
        comm.all_gatherv(torch.ones(2))


def test_all_gatherv_uniform_sizes_use_all_gather():
    comm = make()
    comm.pyflagcx_comm = flagcx(all_gather=Mock(), all_gatherv=Mock())
    inp = torch.ones(3)
    out = comm.all_gatherv(inp, sizes=[7, 7])
    assert out.shape == (6,)
    comm.pyflagcx_comm.all_gather.assert_called_once()
    comm.pyflagcx_comm.all_gatherv.assert_not_called()


def test_all_gatherv_nonuniform_sizes_use_all_gatherv():
    comm = make()
    comm.rank_in_group = 0
    comm.pyflagcx_comm = flagcx(all_gather=Mock(), all_gatherv=Mock())
    inp = torch.ones(2)
    out = comm.all_gatherv(inp, sizes=[2, 4])
    assert out.shape == (6,)
    comm.pyflagcx_comm.all_gatherv.assert_called_once()
    assert comm.pyflagcx_comm.all_gatherv.call_args.kwargs["sizes"] == [2, 4]


def test_all_gatherv_validates_rank_size():
    comm = make()
    comm.rank_in_group = 1
    comm.pyflagcx_comm = flagcx(all_gather=Mock(), all_gatherv=Mock())
    with pytest.raises(AssertionError):
        comm.all_gatherv(torch.ones(2), sizes=[2, 4])


def test_all_gatherv_list_input_groups_calls():
    comm = make()
    comm.pyflagcx_comm = flagcx(
        all_gather=Mock(), all_gatherv=Mock(), group_start=Mock(), group_end=Mock()
    )
    out = comm.all_gatherv([torch.ones(2), torch.ones(2)])
    assert isinstance(out, list) and len(out) == 2
    comm.pyflagcx_comm.group_start.assert_called_once()
    comm.pyflagcx_comm.group_end.assert_called_once()


# ---------------------------------------------------------------------------
# all2all forwarding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method,args",
    [
        ("dispatch_router_logits", (object(), object(), True, [object()])),
        ("dispatch", (object(), object(), object(), True, [object()])),
        ("combine", (object(), True)),
    ],
)
def test_all2all_forwarding(method, args):
    comm = make()
    with pytest.raises(AssertionError):
        getattr(comm, method)(*args)
    comm.all2all_manager = MagicMock()
    getattr(comm, method)(*args)
    getattr(comm.all2all_manager, method).assert_called_once_with(*args)


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------
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


def test_init_world_size_one_skips_flagcx(base_init):
    comm = cm.CommunicatorFL(cpu_group=object(), device=torch.device("cpu"))
    assert comm.pyflagcx_comm is None


def test_init_builds_flagcx_comm_for_multi_rank(base_init, monkeypatch):
    factory = Mock(return_value=SimpleNamespace(disabled=False))
    monkeypatch.setattr(cm, "PyFlagcxCommunicator", factory)

    def fake_base(self, *args, **kwargs):
        self.world_size = 2
        self.all2all_manager = None
        self.all2all_backend = "allgather_reducescatter"
        self.cpu_group = args[0]
        self.device = args[1]
        self.device_group = args[2]
        self.use_all2all = kwargs.get("use_all2all", False)

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base)
    device = torch.device("cpu")
    cpu_group = object()
    comm = cm.CommunicatorFL(cpu_group=cpu_group, device=device)
    factory.assert_called_once_with(group=cpu_group, device=device)
    assert comm.pyflagcx_comm is not None


def test_init_raises_for_npu_when_flagcx_disabled(base_init, monkeypatch):
    factory = Mock(return_value=SimpleNamespace(disabled=True, init_error=OSError("no lib")))
    monkeypatch.setattr(cm, "PyFlagcxCommunicator", factory)

    def fake_base(self, *args, **kwargs):
        self.world_size = 2
        self.all2all_manager = None
        self.all2all_backend = "allgather_reducescatter"
        self.rank_in_group = 0
        self.ranks = [0, 1]
        self.cpu_group = args[0] if args else kwargs.get("cpu_group")
        self.device = args[1] if len(args) > 1 else kwargs.get("device")
        self.device_group = args[2] if len(args) > 2 else kwargs.get("device_group")
        self.use_all2all = kwargs.get("use_all2all", False)

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base)
    import vllm.platforms as vllm_platforms

    monkeypatch.setattr(
        vllm_platforms,
        "current_platform",
        SimpleNamespace(vendor_name="nvidia"),
    )
    with pytest.raises(RuntimeError, match="NPU"):
        cm.CommunicatorFL(cpu_group=object(), device=SimpleNamespace(type="npu"))


def test_init_raises_for_amd_when_flagcx_disabled(base_init, monkeypatch):
    factory = Mock(return_value=SimpleNamespace(disabled=True, init_error=OSError("no lib")))
    monkeypatch.setattr(cm, "PyFlagcxCommunicator", factory)

    def fake_base(self, *args, **kwargs):
        self.world_size = 2
        self.all2all_manager = None
        self.all2all_backend = "allgather_reducescatter"
        self.rank_in_group = 0
        self.ranks = [0, 1]
        self.cpu_group = args[0] if args else kwargs.get("cpu_group")
        self.device = args[1] if len(args) > 1 else kwargs.get("device")
        self.device_group = args[2] if len(args) > 2 else kwargs.get("device_group")
        self.use_all2all = kwargs.get("use_all2all", False)

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base)
    import vllm.platforms as vllm_platforms

    monkeypatch.setattr(
        vllm_platforms,
        "current_platform",
        SimpleNamespace(vendor_name="amd"),
    )
    with pytest.raises(RuntimeError, match="AMD"):
        cm.CommunicatorFL(cpu_group=object(), device=torch.device("cpu"))


def test_init_tolerates_disabled_on_other_vendors(base_init, monkeypatch):
    factory = Mock(return_value=SimpleNamespace(disabled=True, init_error=OSError("no lib")))
    monkeypatch.setattr(cm, "PyFlagcxCommunicator", factory)

    def fake_base(self, *args, **kwargs):
        self.world_size = 2
        self.all2all_manager = None
        self.all2all_backend = "allgather_reducescatter"
        self.rank_in_group = 0
        self.ranks = [0, 1]
        self.cpu_group = args[0] if args else kwargs.get("cpu_group")
        self.device = args[1] if len(args) > 1 else kwargs.get("device")
        self.device_group = args[2] if len(args) > 2 else kwargs.get("device_group")
        self.use_all2all = kwargs.get("use_all2all", False)

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base)
    import vllm.platforms as vllm_platforms

    monkeypatch.setattr(
        vllm_platforms,
        "current_platform",
        SimpleNamespace(vendor_name="mthreads"),
    )
    comm = cm.CommunicatorFL(cpu_group=object(), device=torch.device("cpu"))
    assert comm.pyflagcx_comm.disabled


def test_init_use_all2all_builds_manager(base_init, monkeypatch):
    factory = Mock(return_value=SimpleNamespace(disabled=False))
    monkeypatch.setattr(cm, "PyFlagcxCommunicator", factory)
    seen = []
    monkeypatch.setattr(
        cm.CommunicatorFL, "_init_all2all_manager", lambda self, group: seen.append(group)
    )
    tcp = object()
    cm.CommunicatorFL(cpu_group=object(), device=torch.device("cpu"),
                      tcp_store_group=tcp, use_all2all=True)
    assert seen == [tcp]


# ---------------------------------------------------------------------------
# _init_all2all_manager
# ---------------------------------------------------------------------------
@pytest.fixture
def all2all_mod():
    from vllm.distributed.device_communicators import all2all

    return all2all


ALL2ALL_CASES = [
    ("naive", "AgRsAll2AllManager", "tcp"),
    ("allgather_reducescatter", "AgRsAll2AllManager", "tcp"),
    ("deepep_high_throughput", "DeepEPHTAll2AllManager", "tcp"),
    ("deepep_low_latency", "DeepEPLLAll2AllManager", "tcp"),
    ("mori_high_throughput", "MoriAll2AllManager", "backend"),
    ("mori_low_latency", "MoriAll2AllManager", "backend"),
    ("deepep_v2", "DeepEPV2All2AllManager", "v2"),
    ("nixl_ep", "NixlEPAll2AllManager", "tcp"),
    ("flashinfer_all2allv", "FlashInferNVLinkTwoSidedManager", "tcp"),
    ("flashinfer_nvlink_two_sided", "FlashInferNVLinkTwoSidedManager", "tcp"),
    ("flashinfer_nvlink_one_sided", "FlashInferNVLinkOneSidedManager", "one_sided"),
]


@pytest.mark.parametrize("backend,name,kind", ALL2ALL_CASES)
def test_all2all_manager_selection(all2all_mod, monkeypatch, backend, name, kind):
    factory = MagicMock()
    for attr in {
        "AgRsAll2AllManager",
        "DeepEPHTAll2AllManager",
        "DeepEPLLAll2AllManager",
        "MoriAll2AllManager",
        "DeepEPV2All2AllManager",
        "NixlEPAll2AllManager",
        "FlashInferNVLinkTwoSidedManager",
        "FlashInferNVLinkOneSidedManager",
    }:
        monkeypatch.setattr(all2all_mod, attr, MagicMock(), raising=False)
    monkeypatch.setattr(all2all_mod, name, factory)
    monkeypatch.setattr(cm.logger, "info_once", Mock())
    monkeypatch.setattr(cm.logger, "warning_once", Mock())

    comm = make(all2all_backend=backend, device_group="DEVG")
    tcp = "TCP"
    comm._init_all2all_manager(tcp)

    factory.assert_called_once()
    assert comm.all2all_manager is factory.return_value
    cm.logger.info_once.assert_called_once()


def test_all2all_manager_selection_argument_shapes(all2all_mod, monkeypatch):
    names = {
        "AgRsAll2AllManager",
        "DeepEPHTAll2AllManager",
        "DeepEPLLAll2AllManager",
        "MoriAll2AllManager",
        "DeepEPV2All2AllManager",
        "NixlEPAll2AllManager",
        "FlashInferNVLinkTwoSidedManager",
        "FlashInferNVLinkOneSidedManager",
    }
    factories = {name: MagicMock() for name in names}
    for name, factory in factories.items():
        monkeypatch.setattr(all2all_mod, name, factory, raising=False)
    monkeypatch.setattr(cm.logger, "info_once", Mock())
    monkeypatch.setattr(cm.logger, "warning_once", Mock())

    comm = make(all2all_backend="deepep_v2", cpu_group="CPU", device_group="DEV")
    comm._init_all2all_manager("TCP")
    factories["DeepEPV2All2AllManager"].assert_called_once_with(
        "CPU", "TCP", device_group="DEV"
    )

    comm = make(all2all_backend="mori_low_latency", cpu_group="CPU")
    comm._init_all2all_manager("TCP")
    factories["MoriAll2AllManager"].assert_called_once_with("CPU", "mori_low_latency")

    comm = make(all2all_backend="flashinfer_nvlink_one_sided", cpu_group="CPU")
    comm._init_all2all_manager("TCP")
    factories["FlashInferNVLinkOneSidedManager"].assert_called_once_with("CPU")


def test_all2all_deprecated_alias_warns(all2all_mod, monkeypatch):
    monkeypatch.setattr(all2all_mod, "FlashInferNVLinkTwoSidedManager", MagicMock(), raising=False)
    warn = Mock()
    monkeypatch.setattr(cm.logger, "warning_once", warn)
    monkeypatch.setattr(cm.logger, "info_once", Mock())
    comm = make(all2all_backend="flashinfer_all2allv", cpu_group="CPU")
    comm._init_all2all_manager("TCP")
    warn.assert_called_once()


def test_all2all_unknown_backend_raises(monkeypatch):
    comm = make(all2all_backend="does-not-exist", cpu_group="CPU")
    with pytest.raises(ValueError, match="Unknown all2all backend"):
        comm._init_all2all_manager(None)

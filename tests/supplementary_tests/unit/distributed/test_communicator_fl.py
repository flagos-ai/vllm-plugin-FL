# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for vllm_fl.distributed.communicator.CommunicatorFL.

CPU-only tests complementing tests/unit_tests/distributed/test_communicator.py:

* ``__init__`` with world_size > 1 installs a (faked) PyFlagcxCommunicator
  built on the cpu_group, and the NPU/AMD disabled-hard-fail guard.
* ``_init_all2all_manager`` dispatch for every backend name (each with the
  manager class imported from the real vllm all2all module and mocked).
* ``all_reduce`` passthrough and the disabled fall back to torch.distributed.
* ``all_gather`` world_size 1 identity, disabled/base fallback, and the
  flagcx-active reshape/movedim path.
* ``reduce_scatter``/``reduce_scatterv`` dim handling and sizes dispatch.
* ``send``/``recv``/``broadcast`` flagcx vs torch.distributed dispatch.
* ``destroy`` (txda child destroyed, all2all manager destroyed exactly once).
* ``checkpoint_prepare``/``checkpoint_restore`` delegate when set.
* ``all_gatherv`` dim!=0 rejection, uniform-sizes collapse, single-tensor
  and list-of-tensor paths (group_start/group_end pairing).

The DeviceCommunicatorBase.__init__ mock follows the core pattern at
tests/unit_tests/distributed/test_communicator.py:32-56.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

pytestmark = [pytest.mark.supplementary]

from vllm.distributed.device_communicators.base_device_communicator import (  # noqa: E402
    DeviceCommunicatorBase,
)
from vllm_fl.distributed.communicator import CommunicatorFL  # noqa: E402

ALL2ALL_BACKENDS = [
    "naive",
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "mori_high_throughput",
    "mori_low_latency",
    "deepep_v2",
    "nixl_ep",
    "flashinfer_all2allv",
    "flashinfer_nvlink_two_sided",
    "flashinfer_nvlink_one_sided",
]


@pytest.fixture
def mocked_base_init(monkeypatch):
    """Mock DeviceCommunicatorBase.__init__ (core pattern, world_size=2).

    Follows tests/unit_tests/distributed/test_communicator.py:32-56, but
    leaves world_size configurable so the pyflagcx_comm branch runs.
    """
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
        self.world_size = 2
        self.rank_in_group = 0
        self.ranks = [0, 1]
        self.use_all2all = use_all2all
        self.all2all_backend = "allgather_reducescatter"
        self.all2all_manager = None

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base_init)
    return received


class _FakePyFlagcx:
    """Recording stand-in for PyFlagcxCommunicator."""

    def __init__(self, *, disabled=False):
        self.disabled = disabled
        self.init_error = RuntimeError("flagcx unavailable") if disabled else None
        self.calls = []

    def all_reduce(self, tensor, out_tensor=None, op=None, stream=None):
        self.calls.append(("all_reduce", tensor, out_tensor))
        if out_tensor is not None:
            out_tensor.copy_(tensor)
            return out_tensor
        return tensor.clone()

    def all_gather(self, output_tensor, input_tensor, stream=None):
        self.calls.append(("all_gather", output_tensor, input_tensor))
        output_tensor[: input_tensor.shape[0]] = input_tensor
        output_tensor[input_tensor.shape[0] :] = input_tensor
        return output_tensor

    def all_gatherv(self, output_tensor, input_tensor, sizes, stream=None):
        self.calls.append(("all_gatherv", output_tensor, input_tensor, tuple(sizes)))
        offset = 0
        n = input_tensor.shape[0]
        for size in sizes:
            # Each rank broadcasts its own slice; fill only what we have.
            take = min(size, n)
            output_tensor[offset : offset + take] = input_tensor[:take]
            offset += size
        return output_tensor

    def reduce_scatter(self, output_tensor, input_tensor, op=None, stream=None):
        self.calls.append(("reduce_scatter", output_tensor, input_tensor))
        output_tensor.copy_(input_tensor[: output_tensor.shape[0]])
        return output_tensor

    def reduce_scatterv(self, output_tensor, input_tensor, sizes, op=None, stream=None):
        self.calls.append(("reduce_scatterv", output_tensor, input_tensor, tuple(sizes)))
        size = sizes[0]
        output_tensor.copy_(input_tensor[:size])
        return output_tensor

    def send(self, tensor, dst, stream=None):
        self.calls.append(("send", tensor, dst))

    def recv(self, tensor, src, stream=None):
        self.calls.append(("recv", tensor, src))

    def broadcast(self, tensor, src, stream=None):
        self.calls.append(("broadcast", tensor, src))
        return tensor

    def group_start(self):
        self.calls.append(("group_start",))

    def group_end(self):
        self.calls.append(("group_end",))

    def destroy(self):
        self.calls.append(("destroy",))


def _make_communicator(world_size=2, pyflagcx=None, rank_in_group=0, ranks=None):
    """Build a CommunicatorFL via __new__ with a preset base state."""
    comm = CommunicatorFL.__new__(CommunicatorFL)
    comm.world_size = world_size
    comm.rank_in_group = rank_in_group
    comm.ranks = ranks or list(range(world_size))
    comm.device = torch.device("cpu")
    comm.device_group = "device-group"
    comm.cpu_group = "cpu-group"
    comm.all2all_backend = "allgather_reducescatter"
    comm.all2all_manager = None
    comm.pyflagcx_comm = pyflagcx
    return comm


class TestInit:
    def test_world_size_two_installs_pyflagcx_on_cpu_group(
        self, mocked_base_init, monkeypatch
    ):
        installed = []
        monkeypatch.setattr(
            "vllm_fl.distributed.communicator.PyFlagcxCommunicator",
            lambda group, device: installed.append((group, device))
            or _FakePyFlagcx(disabled=False),
        )

        cpu_group = object()
        device = torch.device("cpu")
        comm = CommunicatorFL(cpu_group=cpu_group, device=device)

        assert installed == [(cpu_group, device)]
        assert comm.pyflagcx_comm is not None
        assert not comm.pyflagcx_comm.disabled
        assert mocked_base_init["cpu_group"] is cpu_group

    def test_disabled_pyflagcx_raises_on_npu(
        self, mocked_base_init, monkeypatch
    ):
        from vllm.platforms import current_platform

        monkeypatch.setattr(
            "vllm_fl.distributed.communicator.PyFlagcxCommunicator",
            lambda group, device: _FakePyFlagcx(disabled=True),
        )
        # Force the npu device-type branch.
        comm_device = SimpleNamespace(type="npu")
        with pytest.raises(RuntimeError, match="FlagCX requested on NPU"):
            CommunicatorFL(cpu_group=object(), device=comm_device)

    def test_disabled_pyflagcx_raises_on_amd_vendor(
        self, mocked_base_init, monkeypatch
    ):
        from vllm.platforms import current_platform

        monkeypatch.setattr(
            "vllm_fl.distributed.communicator.PyFlagcxCommunicator",
            lambda group, device: _FakePyFlagcx(disabled=True),
        )
        # current_platform may not expose vendor_name on this build; setattr
        # with raising=False on the instance-equivalent module attribute.
        monkeypatch.setattr(
            current_platform, "vendor_name", "amd", raising=False
        )
        comm_device = SimpleNamespace(type="cuda")
        try:
            with pytest.raises(RuntimeError, match="FlagCX requested on AMD"):
                CommunicatorFL(cpu_group=object(), device=comm_device)
        finally:
            if not hasattr(type(current_platform), "vendor_name"):
                del current_platform.vendor_name

    def test_use_all2all_true_initializes_manager(
        self, mocked_base_init, monkeypatch
    ):
        """__init__ calls _init_all2all_manager when use_all2all is set."""
        manager_init = []
        monkeypatch.setattr(
            CommunicatorFL,
            "_init_all2all_manager",
            lambda self, group: manager_init.append(group),
        )
        monkeypatch.setattr(
            "vllm_fl.distributed.communicator.PyFlagcxCommunicator",
            lambda group, device: _FakePyFlagcx(),
        )
        tcp_store_group = object()
        comm = CommunicatorFL(
            cpu_group=object(),
            device=torch.device("cpu"),
            tcp_store_group=tcp_store_group,
            use_all2all=True,
        )
        assert comm.pyflagcx_comm is not None
        assert manager_init == [tcp_store_group]

    def test_disabled_pyflagcx_tolerated_on_other_devices(
        self, mocked_base_init, monkeypatch
    ):
        monkeypatch.setattr(
            "vllm_fl.distributed.communicator.PyFlagcxCommunicator",
            lambda group, device: _FakePyFlagcx(disabled=True),
        )
        comm = CommunicatorFL(cpu_group=object(), device=torch.device("cpu"))
        assert comm.pyflagcx_comm.disabled
        assert comm.all2all_manager is None


class TestAll2AllManagerDispatch:
    def _make(self, backend):
        comm = CommunicatorFL.__new__(CommunicatorFL)
        comm.all2all_backend = backend
        comm.cpu_group = object()
        comm.device_group = object()
        comm.all2all_manager = None
        return comm

    @pytest.mark.parametrize("backend", ALL2ALL_BACKENDS)
    def test_each_backend_selects_its_manager(self, backend):
        from vllm.distributed.device_communicators import all2all

        expectations = {
            "naive": ("AgRsAll2AllManager", "group"),
            "allgather_reducescatter": ("AgRsAll2AllManager", "group"),
            "deepep_high_throughput": ("DeepEPHTAll2AllManager", "group"),
            "deepep_low_latency": ("DeepEPLLAll2AllManager", "group"),
            "mori_high_throughput": ("MoriAll2AllManager", "backend"),
            "mori_low_latency": ("MoriAll2AllManager", "backend"),
            "deepep_v2": ("DeepEPV2All2AllManager", "device_group"),
            "nixl_ep": ("NixlEPAll2AllManager", "group"),
            "flashinfer_all2allv": (
                "FlashInferNVLinkTwoSidedManager",
                "group",
            ),
            "flashinfer_nvlink_two_sided": (
                "FlashInferNVLinkTwoSidedManager",
                "group",
            ),
            "flashinfer_nvlink_one_sided": (
                "FlashInferNVLinkOneSidedManager",
                "plain",
            ),
        }
        comm = self._make(backend)
        manager = Mock()
        factory = Mock(return_value=manager)
        original = getattr(all2all, expectations[backend][0])
        try:
            setattr(all2all, expectations[backend][0], factory)
            tcp_store_group = object()

            comm._init_all2all_manager(tcp_store_group)

            assert comm.all2all_manager is manager
            if expectations[backend][1] == "backend":
                factory.assert_called_once_with(comm.cpu_group, backend)
            elif expectations[backend][1] == "plain":
                factory.assert_called_once_with(comm.cpu_group)
            elif expectations[backend][1] == "device_group":
                factory.assert_called_once_with(
                    comm.cpu_group, tcp_store_group, device_group=comm.device_group
                )
            else:
                factory.assert_called_once_with(comm.cpu_group, tcp_store_group)
        finally:
            setattr(all2all, expectations[backend][0], original)

    def test_unknown_backend_raises_value_error(self):
        comm = self._make("not_a_backend")
        with pytest.raises(ValueError, match="Unknown all2all backend"):
            comm._init_all2all_manager(None)


class TestAllReduce:
    def test_flagcx_active_path(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(pyflagcx=fake)
        x = torch.tensor([1.0, 2.0])
        out = comm.all_reduce(x)
        assert fake.calls == [("all_reduce", x, None)]
        assert out is not x
        assert torch.equal(out, x)

    def test_flagcx_returns_none_falls_back_to_torch(self, monkeypatch):
        """A disabled flagcx returning None falls back to torch.distributed."""
        fake = _FakePyFlagcx(disabled=True)
        fake.all_reduce = lambda tensor, out_tensor=None, op=None, stream=None: None
        comm = _make_communicator(pyflagcx=fake)
        torch_calls = []
        monkeypatch.setattr(
            torch.distributed,
            "all_reduce",
            lambda tensor, group=None: torch_calls.append((tensor, group))
            or tensor.mul_(2),
        )
        x = torch.tensor([1.0, 2.0])
        out = comm.all_reduce(x)
        # out is a clone, not the input
        assert out is not x
        assert torch.equal(out, torch.tensor([2.0, 4.0]))
        assert torch.equal(x, torch.tensor([1.0, 2.0]))
        assert torch_calls == [(out, "device-group")]

    def test_missing_pyflagcx_asserts(self):
        comm = _make_communicator(pyflagcx=None)
        with pytest.raises(AssertionError):
            comm.all_reduce(torch.zeros(2))


class TestAllGather:
    def test_world_size_one_returns_input(self):
        comm = _make_communicator(world_size=1, pyflagcx=None)
        x = torch.zeros(2, 3)
        assert comm.all_gather(x) is x

    def test_missing_or_disabled_flagcx_falls_back_to_base(self, monkeypatch):
        from vllm.distributed.device_communicators.base_device_communicator import (
            DeviceCommunicatorBase,
        )

        expected = torch.zeros(4)
        base_calls = []
        monkeypatch.setattr(
            DeviceCommunicatorBase,
            "all_gather",
            lambda self, inp, dim: base_calls.append((inp, dim)) or expected,
        )
        x = torch.zeros(2, 3)

        no_flagcx = _make_communicator(pyflagcx=None)
        assert no_flagcx.all_gather(x, dim=1) is expected

        disabled = _make_communicator(pyflagcx=_FakePyFlagcx(disabled=True))
        assert disabled.all_gather(x, dim=1) is expected
        assert base_calls == [(x, 1), (x, 1)]

    def test_flagcx_active_path_reshapes_last_dim(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake)
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        out = comm.all_gather(x, dim=-1)

        assert out.shape == (2, 6)
        assert fake.calls[0][0] == "all_gather"
        gathered = fake.calls[0][1]
        assert gathered.shape == (4, 3)
        # dim=-1 resolved to the last dimension
        assert out.shape == x.shape[:-1] + (x.shape[-1] * 2,)

    def test_flagcx_active_path_explicit_dim(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake)
        x = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        out = comm.all_gather(x, dim=0)

        assert out.shape == (4, 2)
        assert fake.calls[0][0] == "all_gather"


class TestReduceScatter:
    def test_last_dim_moves_and_restores(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake)
        x = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

        out = comm.reduce_scatter(x, dim=-1)

        # dim=-1: movedim(0, 2) permutes the axes; chunk 2/2=1 row out.
        assert out.shape == (2, 2, 1)
        assert fake.calls[0][0] == "reduce_scatter"
        lib_in = fake.calls[0][2]
        # The library received the moved contiguous tensor (dim permuted to 0).
        assert torch.equal(lib_in, x.movedim(0, 2).contiguous())

    def test_explicit_dim_zero(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake)
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2)

        out = comm.reduce_scatter(x, dim=0)

        assert out.shape == (2, 2)
        assert fake.calls[0][2] is not None
        assert fake.calls[0][2].shape == (4, 2)

    def test_missing_pyflagcx_asserts(self):
        comm = _make_communicator(pyflagcx=None)
        with pytest.raises(AssertionError):
            comm.reduce_scatter(torch.zeros(4, 4))


class TestReduceScatterV:
    def test_uniform_sizes_dispatch_to_reduce_scatterv(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake, rank_in_group=0)
        x = torch.arange(8, dtype=torch.float32)

        out = comm.reduce_scatterv(x, dim=0, sizes=[4, 4])

        # reduce_scatterv always forwards sizes (no uniform collapse here).
        assert out.shape == (4,)
        assert fake.calls[0][0] == "reduce_scatterv"
        assert fake.calls[0][3] == (4, 4)

    def test_none_sizes_dispatch_to_reduce_scatter(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake)
        x = torch.arange(8, dtype=torch.float32)

        out = comm.reduce_scatterv(x, dim=0, sizes=None)

        # sizes=None: chunk = shape[0] // world_size, plain reduce_scatter.
        assert out.shape == (4,)
        assert fake.calls[0][0] == "reduce_scatter"

    def test_uneven_sizes_dispatch_to_reduce_scatterv(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake, rank_in_group=0)
        x = torch.arange(7, dtype=torch.float32)

        out = comm.reduce_scatterv(x, dim=0, sizes=[3, 4])

        assert out.shape == (3,)  # sizes[rank_in_group]
        assert fake.calls[0][0] == "reduce_scatterv"
        assert fake.calls[0][3] == (3, 4)

    def test_negative_dim_converted(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake, rank_in_group=0)
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)

        out = comm.reduce_scatterv(x, dim=-1, sizes=[2, 2])

        # dim=-1 resolves to 1: movedim(0,1) permutes (2,4)->(4,2); the
        # sizes[rank] slice comes back and moves to dim 1 again.
        assert out.shape == (2, 2)
        assert fake.calls[0][0] == "reduce_scatterv"

    def test_missing_pyflagcx_asserts(self):
        comm = _make_communicator(pyflagcx=None)
        with pytest.raises(AssertionError):
            comm.reduce_scatterv(torch.zeros(8))


class TestPointToPoint:
    def test_send_uses_flagcx_when_enabled(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(pyflagcx=fake)
        x = torch.zeros(3)
        comm.send(x, dst=1)
        assert fake.calls == [("send", x, 1)]

    def test_send_default_dst_wraps_rank(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=4, pyflagcx=fake, rank_in_group=2, ranks=[10, 11, 12, 13])
        x = torch.zeros(3)
        comm.send(x)
        assert fake.calls == [("send", x, 3)]  # (rank+1) % world_size

    def test_send_falls_back_to_torch_when_no_flagcx(self, monkeypatch):
        torch_calls = []
        monkeypatch.setattr(
            torch.distributed,
            "send",
            lambda tensor, dst, group=None: torch_calls.append((tensor, dst, group)),
        )
        comm = _make_communicator(pyflagcx=None, ranks=[7, 9])
        x = torch.zeros(3)
        comm.send(x, dst=1)
        assert torch_calls == [(x, 9, "device-group")]

    def test_recv_uses_flagcx_when_enabled(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(pyflagcx=fake)
        result = comm.recv(torch.Size([3]), torch.float32, src=1)
        assert fake.calls[0][0] == "recv"
        assert result.shape == (3,)
        assert result.dtype == torch.float32

    def test_recv_default_src_wraps_rank(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=4, pyflagcx=fake, rank_in_group=2, ranks=[10, 11, 12, 13])
        comm.recv(torch.Size([2]), torch.float32)
        assert fake.calls[0][2] == 1  # (rank-1) % world_size

    def test_recv_falls_back_to_torch_when_no_flagcx(self, monkeypatch):
        torch_calls = []
        monkeypatch.setattr(
            torch.distributed,
            "recv",
            lambda tensor, src, group=None: torch_calls.append((tensor, src, group)),
        )
        comm = _make_communicator(pyflagcx=None, ranks=[7, 9])
        result = comm.recv((3,), torch.float32, src=1)
        assert result.shape == (3,)
        assert torch_calls[0][1:] == (9, "device-group")

    def test_broadcast_world_size_one_returns_tensor(self):
        comm = _make_communicator(world_size=1, pyflagcx=None)
        x = torch.zeros(3)
        assert comm.broadcast(x, src=0) is x

    def test_broadcast_uses_flagcx_when_enabled(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(pyflagcx=fake)
        x = torch.zeros(3)
        result = comm.broadcast(x, src=1)
        assert result is x
        assert fake.calls == [("broadcast", x, 1)]

    def test_broadcast_falls_back_to_base_when_no_flagcx(self, monkeypatch):
        from vllm.distributed.device_communicators.base_device_communicator import (
            DeviceCommunicatorBase,
        )

        expected = object()
        monkeypatch.setattr(
            DeviceCommunicatorBase,
            "broadcast",
            lambda self, tensor, src: expected,
        )
        comm = _make_communicator(pyflagcx=None)
        assert comm.broadcast(torch.zeros(3), src=0) is expected


class TestAllGatherV:
    def test_nonzero_dim_raises(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(pyflagcx=fake)
        with pytest.raises(NotImplementedError, match="only dim 0"):
            comm.all_gatherv(torch.zeros(4), dim=1)

    def test_missing_pyflagcx_asserts(self):
        comm = _make_communicator(pyflagcx=None)
        with pytest.raises(AssertionError):
            comm.all_gatherv(torch.zeros(4))

    def test_uniform_sizes_collapse_to_all_gather(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake)
        x = torch.arange(4, dtype=torch.float32)

        out = comm.all_gatherv(x, dim=0, sizes=[2, 2])

        # Uniform sizes collapse: output is input_size[0] * world_size.
        assert out.shape == (8,)
        assert fake.calls[0][0] == "all_gather"

    def test_uneven_sizes_use_all_gatherv(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake, rank_in_group=0)
        x = torch.arange(3, dtype=torch.float32)

        out = comm.all_gatherv(x, dim=0, sizes=[3, 4])

        assert out.shape == (7,)
        assert fake.calls[0][0] == "all_gatherv"
        assert fake.calls[0][3] == (3, 4)

    def test_list_input_gathers_each_with_group_marks(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(world_size=2, pyflagcx=fake)
        tensors = [torch.arange(2, dtype=torch.float32), torch.arange(2, dtype=torch.float32)]

        outs = comm.all_gatherv(tensors, dim=0)

        assert isinstance(outs, list) and len(outs) == 2
        names = [c[0] for c in fake.calls]
        assert names == [
            "group_start",
            "all_gather",
            "all_gather",
            "group_end",
        ]
        assert all(o.shape == (4,) for o in outs)


class TestLifecycle:
    def test_destroy_txda_child_is_destroyed(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(pyflagcx=fake)
        comm.device = SimpleNamespace(type="txda")
        manager = Mock()
        comm.all2all_manager = manager

        comm.destroy()

        assert ("destroy",) in fake.calls
        assert comm.pyflagcx_comm is None
        manager.destroy.assert_called_once_with()

    def test_destroy_non_txda_child_not_destroyed(self):
        fake = _FakePyFlagcx()
        comm = _make_communicator(pyflagcx=fake)
        comm.device = torch.device("cpu")

        comm.destroy()

        assert ("destroy",) not in fake.calls
        assert comm.pyflagcx_comm is None

    def test_destroy_without_pyflagcx_clears_manager(self):
        comm = _make_communicator(pyflagcx=None)
        manager = Mock()
        comm.all2all_manager = manager

        comm.destroy()

        manager.destroy.assert_called_once_with()
        assert comm.all2all_manager is None

    def test_checkpoint_prepare_and_restore_delegate(self):
        comm = _make_communicator(pyflagcx=None)
        manager = Mock()
        comm.all2all_manager = manager

        comm.checkpoint_prepare()
        comm.checkpoint_restore()

        manager.checkpoint_prepare.assert_called_once_with()
        manager.checkpoint_restore.assert_called_once_with()

    def test_checkpoint_prepare_without_manager_is_noop(self):
        comm = _make_communicator(pyflagcx=None)
        comm.checkpoint_prepare()
        comm.checkpoint_restore()


class TestAll2AllDelegation:
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("dispatch_router_logits", (object(), object(), True, [object()])),
            ("dispatch", (object(), object(), object(), True, [object()])),
            ("combine", (object(), True)),
        ],
    )
    def test_delegation(self, method, args):
        comm = _make_communicator(pyflagcx=None)
        manager = Mock()
        expected = object()
        getattr(manager, method).return_value = expected
        comm.all2all_manager = manager

        result = getattr(comm, method)(*args)

        getattr(manager, method).assert_called_once_with(*args)
        assert result is expected

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("dispatch_router_logits", (object(), object(), True, [object()])),
            ("dispatch", (object(), object(), object(), True, [object()])),
            ("combine", (object(), True)),
        ],
    )
    def test_missing_manager_asserts(self, method, args):
        comm = _make_communicator(pyflagcx=None)
        with pytest.raises(AssertionError):
            getattr(comm, method)(*args)

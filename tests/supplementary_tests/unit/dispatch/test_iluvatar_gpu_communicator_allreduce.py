# Copyright 2025 BAAI. All rights reserved.
"""Real-BI-V150 coverage for the Iluvatar communicator construction path.

``vllm_fl/dispatch/backends/vendor/iluvatar/communicator.py`` is 0/34 on the
CPU baseline: constructing the class needs a CUDA device and an initialized
``torch.distributed`` group. The CPU supplementary test
(``tests/supplementary_tests/unit/distributed/test_iluvatar_communicator.py``)
covers ``__init__`` with the base mocked out; these tests run the *real*
``DeviceCommunicatorBase.__init__`` through a single-process gloo group with
the device bound to the BI-V150 — the exact single-card startup shape:

* ``world_size == 1`` keeps ``pyflagcx_comm`` None (no FlagCX stack), matching
  production single-card execution;
* the CudaCommunicator isinstance contract vLLM 0.32 ``graph_capture`` relies
  on holds, while all eleven CUDA fields stay disabled/None so no second
  (NCCL) communication stack is ever initialized;
* the device attribute is the real ``torch.device("cuda")`` the group will
  bind collectives to.

Executing the collective itself requires a second rank (FlagCX or the
torch.distributed fallback), which is covered by the multi-process
functional test
``tests/supplementary_tests/functional/test_iluvatar_gpu_flagcx_allreduce.py``;
``CommunicatorFL.all_reduce`` deliberately asserts ``pyflagcx_comm is not
None`` because in production the GroupCoordinator short-circuits
``world_size == 1`` before the device communicator is ever consulted.
"""

from __future__ import annotations

import uuid

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.platforms("iluvatar"),
    pytest.mark.gpu,
    pytest.mark.serial,
]

# The disabled/None CUDA fields IluvatarCommunicatorFL.__init__ installs so
# GroupCoordinator.graph_capture sees a CudaCommunicator without
# initializing a second (NCCL) comm stack (communicator.py L60-71).
DISABLED_BOOL_FIELDS = (
    "use_custom_allreduce",
    "use_torch_symm_mem",
    "use_flashinfer_allreduce",
    "use_flashinfer_pcie_ipc_allreduce",
    "use_aiter_allreduce",
    "use_aiter_ag_rs",
)
NONE_FIELDS = (
    "pynccl_comm",
    "ca_comm",
    "qr_comm",
    "symm_mem_comm",
    "fi_ar_comm",
    "fi_pcie_ipc_ar_comm",
    "aiter_ar_comm",
)


def _device():
    from tests.utils.device_utils import get_device

    return get_device()


@pytest.fixture(scope="module")
def single_card_comm(tmp_path_factory):
    """A real IluvatarCommunicatorFL on a single-process gloo group."""
    from vllm_fl.dispatch.backends.vendor.iluvatar.communicator import (
        IluvatarCommunicatorFL,
    )

    store_path = tmp_path_factory.mktemp("iluvatar_comm") / f"gloo-{uuid.uuid4().hex}"
    store = dist.FileStore(str(store_path), world_size=1)
    dist.init_process_group(
        backend="gloo", store=store, rank=0, world_size=1
    )
    group = dist.group.WORLD
    device = _device()
    comm = IluvatarCommunicatorFL(
        cpu_group=group,
        device=device,
        device_group=group,
        unique_name="tp:0",
    )
    try:
        yield comm
    finally:
        dist.destroy_process_group()


class TestSingleCardConstruction:
    """``__init__`` through the real DeviceCommunicatorBase on the device."""

    def test_world_size_one_skips_flagcx(self, single_card_comm):
        comm = single_card_comm
        assert comm.world_size == 1
        assert comm.rank == 0
        assert comm.pyflagcx_comm is None, "single card must not init FlagCX"

    def test_group_attributes_from_real_group(self, single_card_comm):
        comm = single_card_comm
        assert comm.cpu_group is dist.group.WORLD
        assert comm.device_group is dist.group.WORLD
        assert comm.unique_name == "tp:0"
        assert comm.device == _device()
        assert comm.device.type == "cuda"
        assert comm.ranks == [0]

    def test_cuda_fields_disabled_or_none(self, single_card_comm):
        """The eleven CUDA fields must not open a second comm stack."""
        comm = single_card_comm
        for field in DISABLED_BOOL_FIELDS:
            value = getattr(comm, field)
            assert value is False, f"{field} must be False, got {value!r}"
        for field in NONE_FIELDS:
            value = getattr(comm, field)
            assert value is None, f"{field} must be None, got {value!r}"

    def test_isinstance_cuda_communicator(self, single_card_comm):
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )

        assert isinstance(single_card_comm, CudaCommunicator)

    def test_all2all_manager_none_for_tp_group(self, single_card_comm):
        """A tp: group never initializes the all2all manager."""
        comm = single_card_comm
        assert comm.is_ep_communicator is False
        assert comm.use_all2all is False
        assert comm.all2all_manager is None


class TestWorldSizeOneSemantics:
    """What production single-card execution actually relies on."""

    def test_group_coordinator_short_circuit_precondition(self, single_card_comm):
        """With pyflagcx None the collective entry must refuse, not silently
        no-op: upstream GroupCoordinator.all_reduce returns the input for
        world_size 1 *before* reaching the device communicator, so this
        assert is the invariant that keeps that ordering honest."""
        comm = single_card_comm
        x = torch.ones(4, dtype=torch.float32, device=_device())
        with pytest.raises(AssertionError):
            comm.all_reduce(x)

    def test_device_tensor_round_trip_unaffected_by_wrapper(self, single_card_comm):
        """The dtype bookkeeping must not mutate the caller's tensor even on
        the refusal path (the wrapper reads ``input_.dtype`` only)."""
        comm = single_card_comm
        x = torch.randn(8, dtype=torch.bfloat16, device=_device())
        before = x.clone()
        with pytest.raises(AssertionError):
            comm.all_reduce(x)
        assert torch.equal(x, before)
        assert x.dtype == torch.bfloat16

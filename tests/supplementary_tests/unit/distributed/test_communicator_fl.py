# Copyright (c) 2026 BAAI. All rights reserved.
"""Supplementary CPU tests for the vllm_fl distributed communicator wrappers.

Deepens ``tests/unit_tests/distributed/test_communicator.py`` (which skips
without a vllm install) by stubbing the vLLM boundary: all2all manager
selection for every backend branch, send/recv/broadcast fallbacks to
torch.distributed, all_gather fallback when FlagCX is disabled, all_gatherv
shape checks, reduce_scatter sized variants, destroy/checkpoint lifecycle,
and the constructor's device-error path on NPU/AMD.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.supplementary_tests.unit.distributed._flagcx_stubs import _OnceLogger

pytest.importorskip("torch")


class _Base:
    """Stub of vLLM's DeviceCommunicatorBase with the fields the FL
    communicator reads and writes."""

    def __init__(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name="",
        global_ranks=None,
        global_world_size=None,
        use_all2all=False,
    ):
        self.cpu_group = cpu_group
        self.device = device if device is not None else torch.device("cpu")
        self.device_group = device_group
        self.unique_name = unique_name
        self.global_ranks = global_ranks
        self.global_world_size = global_world_size
        self.world_size = 1
        self.rank_in_group = 0
        self.ranks = [0]
        self.use_all2all = use_all2all
        self.all2all_backend = "naive"
        self.all2all_manager = None

    def all_gather(self, input_, dim=-1):
        return ("base_all_gather", input_, dim)

    def broadcast(self, tensor, src=0):
        return ("base_broadcast", tensor, src)


class _CudaCommunicator(_Base):
    """Stub of vLLM's CudaCommunicator (bridge base for graph capture)."""


@pytest.fixture
def comm_env(monkeypatch):
    """Import the real communicator modules against vLLM boundary stubs.

    The ``PyFlagcxCommunicator`` leaf is the native FlagCX boundary; each
    test controls its instances through ``make_flagcx``.  The modules under
    test are dropped from sys.modules and re-imported per test so their
    import-time bindings always refer to this environment's stubs.
    """
    created = []

    class PyFlagcxCommunicator:
        def __init__(self, group=None, device=None):
            self.group = group
            self.device = device
            self.disabled = False
            self.init_error = None
            created.append(self)

    def make_flagcx(**attrs):
        instance = PyFlagcxCommunicator()
        for key, value in attrs.items():
            setattr(instance, key, value)
        return instance

    def leaf(name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    def chain(name, **attrs):
        # Parents go through monkeypatch.setitem so every stub is removed on
        # teardown — a leaked non-package 'vllm' stub breaks later tests that
        # import on-disk fake vllm trees.
        setitem = monkeypatch.setitem
        parts = name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                setitem(sys.modules, parent, ModuleType(parent))
        return leaf(name, **attrs)

    # Import the real vllm_fl.distributed packages BEFORE stubbing so the
    # package objects (and their __path__) stay intact.
    import vllm_fl.distributed.device_communicators  # noqa: F401

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.base_device_communicator",
        chain(
            "vllm.distributed.device_communicators.base_device_communicator",
            DeviceCommunicatorBase=_Base,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.cuda_communicator",
        chain(
            "vllm.distributed.device_communicators.cuda_communicator",
            CudaCommunicator=_CudaCommunicator,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.utils",
        chain("vllm.distributed.utils", StatelessProcessGroup=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        leaf("vllm.platforms", current_platform=SimpleNamespace(vendor_name=None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.logger",
        leaf("vllm.logger", init_logger=_OnceLogger),
    )

    # Drop previously imported modules under test so they re-import against
    # this environment's stubs; monkeypatch restores the prior entries.  The
    # parent package attributes are cleared too: a stale attribute survives
    # a sys.modules pop and would hand the import statement's ``as`` binding
    # the previous test's module object.  ``package.__dict__`` membership is
    # used instead of hasattr because the package's ``__getattr__`` imports
    # the submodule as a side effect.
    package = sys.modules.get("vllm_fl.distributed")
    for name in (
        "vllm_fl.distributed.communicator",
        "vllm_fl.distributed.cuda_flagcx_communicator",
        "vllm_fl.distributed.nvidia_communicator",
    ):
        attr = name.rsplit(".", 1)[1]
        if package is not None and attr in package.__dict__:
            monkeypatch.setitem(package.__dict__, attr, package.__dict__[attr])
            del package.__dict__[attr]
        if name in sys.modules:
            monkeypatch.setitem(sys.modules, name, None)
            sys.modules.pop(name, None)

    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.distributed.device_communicators.flagcx",
        leaf(
            "vllm_fl.distributed.device_communicators.flagcx",
            PyFlagcxCommunicator=PyFlagcxCommunicator,
        ),
    )

    # importlib.import_module always resolves through sys.modules, unlike
    # ``import a.b.c as c`` whose attribute-chain binding can resolve a
    # stale parent attribute to the previous test's module object.
    import importlib

    comm = importlib.import_module("vllm_fl.distributed.communicator")
    cuda_fl = importlib.import_module("vllm_fl.distributed.cuda_flagcx_communicator")
    nvidia_fl = importlib.import_module("vllm_fl.distributed.nvidia_communicator")

    return SimpleNamespace(
        comm=comm,
        cuda_fl=cuda_fl,
        nvidia_fl=nvidia_fl,
        created=created,
        make_flagcx=make_flagcx,
    )


def _make(env, cls=None, world_size=1, flagcx=None, **kwargs):
    """Construct a communicator with a controlled world size and FlagCX.

    The base stub's __init__ is wrapped briefly to force ``world_size``
    after the base initialization, mirroring what a real process group
    would report; the wrapper is always restored.
    """
    cls = cls if cls is not None else env.comm.CommunicatorFL
    original_init = _Base.__init__

    def recording_init(self, *args, **kw):
        original_init(self, *args, **kw)
        self.world_size = world_size

    _Base.__init__ = recording_init
    try:
        if flagcx is not None:
            env.comm.PyFlagcxCommunicator = lambda group=None, device=None: flagcx
        return cls(cpu_group=object(), **kwargs)
    finally:
        _Base.__init__ = original_init


# ---------------------------------------------------------------------------
# Constructor and FlagCX wiring
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_world_size_one_skips_flagcx(comm_env):
    """A single-rank group never builds a FlagCX communicator."""
    comm = _make(comm_env, world_size=1)
    assert comm.pyflagcx_comm is None


@pytest.mark.supplementary
def test_world_size_gt_one_builds_flagcx(comm_env):
    """Multi-rank groups construct PyFlagcxCommunicator on the cpu group."""
    comm = _make(comm_env, world_size=2)
    assert comm.pyflagcx_comm is comm_env.created[-1]
    assert comm_env.created[-1].group is comm.cpu_group
    assert comm_env.created[-1].device is comm.device


@pytest.mark.supplementary
def test_disabled_flagcx_raises_on_npu(comm_env):
    """A disabled FlagCX communicator raises RuntimeError on NPU devices,
    chained from the recorded init error."""
    error = OSError("no native library")
    flagcx = comm_env.make_flagcx(disabled=True, init_error=error)
    device = SimpleNamespace(type="npu")

    with pytest.raises(RuntimeError, match="FlagCX requested on NPU") as excinfo:
        _make(comm_env, world_size=2, flagcx=flagcx, device=device)
    assert excinfo.value.__cause__ is error


@pytest.mark.supplementary
def test_disabled_flagcx_raises_on_amd(comm_env, monkeypatch):
    """A disabled FlagCX communicator raises on AMD vendor platforms."""
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(vendor_name="amd")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    flagcx = comm_env.make_flagcx(disabled=True, init_error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="FlagCX requested on AMD"):
        _make(comm_env, world_size=2, flagcx=flagcx)


@pytest.mark.supplementary
def test_disabled_flagcx_tolerated_on_other_vendors(comm_env):
    """Other platforms tolerate a disabled FlagCX (graceful degradation)."""
    flagcx = comm_env.make_flagcx(disabled=True)
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    assert comm.pyflagcx_comm is flagcx


@pytest.mark.supplementary
def test_cuda_bridge_sets_disabled_native_stacks(comm_env):
    """The CUDA graph bridge disables every native comm stack so FlagCX
    owns communication alone."""
    flagcx = comm_env.make_flagcx()
    comm = _make(
        comm_env,
        cls=comm_env.cuda_fl.CudaFlagcxCommunicatorFL,
        world_size=2,
        flagcx=flagcx,
    )
    assert isinstance(comm, comm_env.cuda_fl.CudaCommunicator)
    assert comm.use_custom_allreduce is False
    assert comm.use_torch_symm_mem is False
    assert comm.use_flashinfer_allreduce is False
    assert comm.use_flashinfer_pcie_ipc_allreduce is False
    assert comm.use_aiter_allreduce is False
    assert comm.pynccl_comm is None
    assert comm.ca_comm is None
    assert comm.qr_comm is None
    assert comm.symm_mem_comm is None
    assert comm.fi_ar_comm is None
    assert comm.fi_pcie_ipc_ar_comm is None
    assert comm.aiter_ar_comm is None


@pytest.mark.supplementary
def test_nvidia_communicator_is_cuda_bridge(comm_env):
    """NvidiaCommunicatorFL subclasses the CUDA bridge unchanged."""
    comm = _make(comm_env, cls=comm_env.nvidia_fl.NvidiaCommunicatorFL, world_size=1)
    assert isinstance(comm, comm_env.cuda_fl.CudaFlagcxCommunicatorFL)


# ---------------------------------------------------------------------------
# all2all manager selection
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
@pytest.mark.parametrize(
    ("backend", "manager_name"),
    [
        ("naive", "AgRsAll2AllManager"),
        ("allgather_reducescatter", "AgRsAll2AllManager"),
        ("deepep_high_throughput", "DeepEPHTAll2AllManager"),
        ("deepep_low_latency", "DeepEPLLAll2AllManager"),
        ("mori_high_throughput", "MoriAll2AllManager"),
        ("mori_low_latency", "MoriAll2AllManager"),
        ("deepep_v2", "DeepEPV2All2AllManager"),
        ("nixl_ep", "NixlEPAll2AllManager"),
        ("flashinfer_all2allv", "FlashInferNVLinkTwoSidedManager"),
        ("flashinfer_nvlink_two_sided", "FlashInferNVLinkTwoSidedManager"),
        ("flashinfer_nvlink_one_sided", "FlashInferNVLinkOneSidedManager"),
    ],
)
def test_all2all_manager_selection(comm_env, monkeypatch, backend, manager_name):
    """Each configured backend instantiates its matching manager class with
    the documented constructor arguments."""
    manager_names = (
        "AgRsAll2AllManager",
        "DeepEPHTAll2AllManager",
        "DeepEPLLAll2AllManager",
        "MoriAll2AllManager",
        "DeepEPV2All2AllManager",
        "NixlEPAll2AllManager",
        "FlashInferNVLinkTwoSidedManager",
        "FlashInferNVLinkOneSidedManager",
    )
    managers = {name: Mock(name=name) for name in manager_names}
    all2all = ModuleType("vllm.distributed.device_communicators.all2all")
    for name, mock in managers.items():
        setattr(all2all, name, mock)
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.device_communicators.all2all", all2all
    )

    comm = comm_env.comm.CommunicatorFL.__new__(comm_env.comm.CommunicatorFL)
    comm.all2all_backend = backend
    comm.cpu_group = object()
    comm.device_group = object()
    comm.device = torch.device("cpu")
    tcp_store_group = object()
    comm._init_all2all_manager(tcp_store_group)

    expected = managers[manager_name]
    assert comm.all2all_manager is expected.return_value
    if manager_name == "MoriAll2AllManager":
        expected.assert_called_once_with(comm.cpu_group, backend)
    elif manager_name == "FlashInferNVLinkOneSidedManager":
        expected.assert_called_once_with(comm.cpu_group)
    elif manager_name == "DeepEPV2All2AllManager":
        expected.assert_called_once_with(
            comm.cpu_group, tcp_store_group, device_group=comm.device_group
        )
    else:
        expected.assert_called_once_with(comm.cpu_group, tcp_store_group)


@pytest.mark.supplementary
def test_all2all_unknown_backend_rejected(comm_env):
    """An unknown all2all backend raises a descriptive ValueError."""
    comm = comm_env.comm.CommunicatorFL.__new__(comm_env.comm.CommunicatorFL)
    comm.all2all_backend = "does_not_exist"
    comm.cpu_group = object()
    with pytest.raises(ValueError, match="Unknown all2all backend"):
        comm._init_all2all_manager(None)


@pytest.mark.supplementary
def test_all2all_dispatch_and_combine_delegate(comm_env):
    """dispatch/combine/dispatch_router_logits forward every argument to the
    manager unchanged."""
    comm = comm_env.comm.CommunicatorFL.__new__(comm_env.comm.CommunicatorFL)
    comm.all2all_manager = Mock()
    args = (object(), object(), object(), True, [object()])
    comm.dispatch(*args)
    comm.all2all_manager.dispatch.assert_called_once_with(*args)
    comm.combine(args[0], args[3])
    comm.all2all_manager.combine.assert_called_once_with(args[0], args[3])
    comm.dispatch_router_logits(args[0], args[1], args[3], args[4])
    comm.all2all_manager.dispatch_router_logits.assert_called_once_with(
        args[0], args[1], args[3], args[4]
    )


# ---------------------------------------------------------------------------
# Collective fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_all_reduce_falls_back_to_torch_when_flagcx_returns_none(comm_env, monkeypatch):
    """When FlagCX all_reduce returns None (e.g. during testing), the
    communicator clones and falls back to torch.distributed.all_reduce."""
    flagcx = comm_env.make_flagcx()
    flagcx.all_reduce = lambda tensor: None
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, group=None: calls.append((tensor, group)),
    )
    tensor = torch.ones(4)
    result = comm.all_reduce(tensor)
    assert result is not tensor
    assert torch.equal(result, tensor)
    assert calls == [(result, comm.device_group)]


@pytest.mark.supplementary
def test_all_reduce_uses_flagcx_result(comm_env):
    """A non-None FlagCX result is returned as-is."""
    expected = torch.zeros(3)
    flagcx = comm_env.make_flagcx()
    flagcx.all_reduce = lambda tensor: expected
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    assert comm.all_reduce(torch.ones(3)) is expected


@pytest.mark.supplementary
def test_all_gather_world_size_one_passthrough(comm_env):
    """A single-rank all_gather returns the input untouched."""
    comm = _make(comm_env, world_size=1)
    tensor = torch.ones(2, 3)
    assert comm.all_gather(tensor, dim=1) is tensor


@pytest.mark.supplementary
def test_all_gather_disabled_flagcx_uses_base(comm_env):
    """A disabled FlagCX communicator falls back to the base class with the
    negative dim already normalized."""
    flagcx = comm_env.make_flagcx(disabled=True)
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    tensor = torch.ones(2, 2)
    assert comm.all_gather(tensor, dim=-1) == ("base_all_gather", tensor, 1)


@pytest.mark.supplementary
def test_all_gather_reshapes_and_moves_dim(comm_env):
    """The FlagCX all_gather gathers along dim 0 then moves it into place."""
    flagcx = comm_env.make_flagcx()
    gathered = []

    def fake_all_gather(output, inp):
        gathered.append((tuple(output.shape), tuple(inp.shape)))
        # world_size=2: first half of the rows from rank 0, rest from rank 1.
        output[:2].fill_(1.0)
        output[2:].fill_(2.0)

    flagcx.all_gather = fake_all_gather
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    tensor = torch.arange(4).reshape(2, 2).float()
    result = comm.all_gather(tensor, dim=1)
    assert gathered == [((4, 2), (2, 2))]
    # dim=1: gathered [4, 2] -> moved to [2, 4].
    assert result.shape == (2, 4)
    assert result.tolist() == [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]


@pytest.mark.supplementary
def test_broadcast_disabled_flagcx_uses_base(comm_env):
    """A disabled FlagCX falls back to the base broadcast."""
    flagcx = comm_env.make_flagcx(disabled=True)
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    tensor = torch.ones(2)
    assert comm.broadcast(tensor, src=0) == ("base_broadcast", tensor, 0)


@pytest.mark.supplementary
def test_broadcast_world_size_one_passthrough(comm_env):
    """A single-rank broadcast returns the tensor."""
    comm = _make(comm_env, world_size=1)
    tensor = torch.ones(1)
    assert comm.broadcast(tensor) is tensor


@pytest.mark.supplementary
def test_send_defaults_to_next_rank_and_uses_flagcx(comm_env):
    """send() defaults dst to (rank+1) % world_size and hands the tensor to
    FlagCX when enabled."""
    flagcx = comm_env.make_flagcx()
    sent = []
    flagcx.send = lambda tensor, dst: sent.append((tensor, dst))
    comm = _make(comm_env, world_size=4, flagcx=flagcx)
    comm.rank_in_group = 1
    tensor = torch.zeros(2)
    comm.send(tensor)
    assert sent == [(tensor, 2)]
    comm.send(tensor, dst=0)
    assert sent[-1] == (tensor, 0)


@pytest.mark.supplementary
def test_recv_defaults_to_previous_rank(comm_env):
    """recv() defaults src to (rank-1) % world_size and returns the buffer
    FlagCX filled."""
    flagcx = comm_env.make_flagcx()
    received = []

    def fake_recv(tensor, src):
        received.append((tuple(tensor.shape), src))
        tensor.fill_(7.0)

    flagcx.recv = fake_recv
    comm = _make(comm_env, world_size=4, flagcx=flagcx)
    comm.rank_in_group = 2
    result = comm.recv((3,), torch.float32)
    assert received == [((3,), 1)]
    assert torch.equal(result, torch.full((3,), 7.0))


@pytest.mark.supplementary
def test_send_recv_fallback_to_torch_when_disabled(comm_env, monkeypatch):
    """With FlagCX disabled, send/recv use torch.distributed with global
    ranks mapped through self.ranks."""
    flagcx = comm_env.make_flagcx(disabled=True)
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    comm.rank_in_group = 1
    comm.ranks = [10, 20]
    torch_calls = []
    monkeypatch.setattr(
        torch.distributed, "send", lambda t, r, g: torch_calls.append(("send", r))
    )
    monkeypatch.setattr(
        torch.distributed, "recv", lambda t, r, g: torch_calls.append(("recv", r))
    )
    comm.send(torch.zeros(1), dst=0)
    comm.recv((1,), torch.float32, src=0)
    assert torch_calls == [("send", 10), ("recv", 10)]


# ---------------------------------------------------------------------------
# reduce_scatter / reduce_scatterv / all_gatherv shapes
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_reduce_scatter_splits_and_moves_dim(comm_env):
    """reduce_scatter slices dim 0 into world_size chunks and returns the
    local chunk moved back to the original dim."""
    flagcx = comm_env.make_flagcx()
    seen = []

    def fake_reduce_scatter(output, inp):
        seen.append((tuple(output.shape), tuple(inp.shape)))
        output.copy_(inp[: output.shape[0]])

    flagcx.reduce_scatter = fake_reduce_scatter
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    tensor = torch.arange(8).float().reshape(4, 2)
    result = comm.reduce_scatter(tensor, dim=0)
    assert seen == [((2, 2), (4, 2))]
    assert torch.equal(result, tensor[:2])


@pytest.mark.supplementary
def test_reduce_scatter_normalizes_negative_dim(comm_env):
    """A negative dim is normalized before slicing (dim -1 of a (4, 2)
    tensor scatters the 2-wide axis into 1-wide local chunks)."""
    flagcx = comm_env.make_flagcx()
    seen = []

    def fake_reduce_scatter(output, inp):
        seen.append((tuple(output.shape), tuple(inp.shape)))
        output.fill_(0.0)

    flagcx.reduce_scatter = fake_reduce_scatter
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    result = comm.reduce_scatter(torch.zeros(4, 2), dim=-1)
    # movedim(0, 1) makes the input (2, 4); chunk 1 of dim 0 -> (1, 4);
    # moving back gives the (4, 1) local slice.
    assert seen == [((1, 4), (2, 4))]
    assert result.shape == (4, 1)


@pytest.mark.supplementary
def test_reduce_scatterv_uses_rank_slice_from_sizes(comm_env):
    """reduce_scatterv with explicit sizes allocates this rank's chunk and
    forwards the sizes vector to the native call."""
    flagcx = comm_env.make_flagcx()
    seen = []

    def fake_scatterv(output, inp, sizes=None):
        seen.append((tuple(output.shape), tuple(inp.shape), sizes))
        output.fill_(-1.0)

    flagcx.reduce_scatterv = fake_scatterv
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    comm.rank_in_group = 1
    tensor = torch.arange(6).float().reshape(6, 1)
    result = comm.reduce_scatterv(tensor, dim=0, sizes=[2, 4])
    assert seen == [((4, 1), (6, 1), [2, 4])]
    assert result.shape == (4, 1)
    assert torch.equal(result, torch.full((4, 1), -1.0))


@pytest.mark.supplementary
def test_reduce_scatterv_even_split_without_sizes(comm_env):
    """Without sizes, reduce_scatterv falls back to an even split."""
    flagcx = comm_env.make_flagcx()
    flagcx.reduce_scatter = lambda out, inp: out.copy_(inp[: out.shape[0]])
    comm = _make(comm_env, world_size=3, flagcx=flagcx)
    tensor = torch.arange(9).float()
    result = comm.reduce_scatterv(tensor, dim=0)
    assert result.shape == (3,)
    assert torch.equal(result, tensor[:3])


@pytest.mark.supplementary
def test_reduce_scatterv_validates_sizes(comm_env):
    """reduce_scatterv asserts the sizes vector length, its total against
    the input length, and the even divisibility when sizes is omitted."""
    flagcx = comm_env.make_flagcx()
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    tensor = torch.zeros(4)
    with pytest.raises(AssertionError):
        comm.reduce_scatterv(tensor, dim=0, sizes=[2])
    with pytest.raises(AssertionError):
        comm.reduce_scatterv(tensor, dim=0, sizes=[1, 1])
    with pytest.raises(AssertionError):
        comm.reduce_scatterv(torch.zeros(3), dim=0, sizes=None)


@pytest.mark.supplementary
def test_all_gatherv_rejects_nonzero_dim(comm_env):
    """all_gatherv only supports dim 0."""
    flagcx = comm_env.make_flagcx()
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    with pytest.raises(NotImplementedError, match="only dim 0"):
        comm.all_gatherv(torch.zeros(2), dim=1)


@pytest.mark.supplementary
def test_all_gatherv_uniform_sizes_drop_to_all_gather(comm_env):
    """Uniform sizes degrade to plain all_gather."""
    flagcx = comm_env.make_flagcx()
    calls = []
    flagcx.all_gather = lambda out, inp: (calls.append("gather"), out.fill_(1.0))
    flagcx.all_gatherv = lambda out, inp, sizes=None: calls.append("gatherv")
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    tensor = torch.ones(2, 2)
    result = comm.all_gatherv(tensor, sizes=[2, 2])
    assert calls == ["gather"]
    assert result.shape == (4, 2)
    assert torch.equal(result, torch.ones(4, 2))


@pytest.mark.supplementary
def test_all_gatherv_sized_output_layout(comm_env):
    """With per-rank sizes the output is the concatenation in rank order and
    the sizes vector is forwarded to the native call."""
    flagcx = comm_env.make_flagcx()
    recorded = []

    def fake_gatherv(output, inp, sizes=None):
        recorded.append(sizes)
        output.fill_(9.0)

    flagcx.all_gatherv = fake_gatherv
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    comm.rank_in_group = 1
    result = comm.all_gatherv(torch.zeros(4, 1), sizes=[1, 4])
    assert recorded == [[1, 4]]
    assert result.shape == (5, 1)
    assert torch.equal(result, torch.full((5, 1), 9.0))


@pytest.mark.supplementary
def test_all_gatherv_list_input_groups_calls(comm_env):
    """A list of tensors is gathered under group_start/group_end."""
    flagcx = comm_env.make_flagcx()
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    calls = []
    flagcx.group_start = lambda: calls.append("start")
    flagcx.group_end = lambda: calls.append("end")
    flagcx.all_gather = lambda out, inp: out.fill_(1.0)
    result = comm.all_gatherv([torch.zeros(2), torch.zeros(3)])
    assert calls == ["start", "end"]
    assert [tuple(t.shape) for t in result] == [(4,), (6,)]
    assert all(torch.equal(t, torch.ones_like(t)) for t in result)


@pytest.mark.supplementary
def test_all_gatherv_checks_input_shape_against_sizes(comm_env):
    """The local input's dim-0 must equal this rank's size."""
    flagcx = comm_env.make_flagcx()
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    comm.rank_in_group = 0
    with pytest.raises(AssertionError, match="!="):
        comm.all_gatherv(torch.zeros(3), sizes=[2, 4])


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_destroy_txda_calls_flagcx_destroy(comm_env):
    """destroy() tears the FlagCX communicator down on txda devices."""
    flagcx = comm_env.make_flagcx()
    flagcx.destroy = Mock()
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    comm.device = SimpleNamespace(type="txda")
    comm.destroy()
    flagcx.destroy.assert_called_once_with()
    assert comm.pyflagcx_comm is None


@pytest.mark.supplementary
def test_destroy_other_devices_keep_flagcx(comm_env):
    """Non-txda devices skip the FlagCX destroy but still detach it."""
    flagcx = comm_env.make_flagcx()
    flagcx.destroy = Mock()
    comm = _make(comm_env, world_size=2, flagcx=flagcx)
    comm.device = SimpleNamespace(type="cpu")
    comm.destroy()
    flagcx.destroy.assert_not_called()
    assert comm.pyflagcx_comm is None


@pytest.mark.supplementary
def test_destroy_tears_down_all2all_manager(comm_env):
    """destroy() forwards to the manager's destroy and detaches it."""
    comm = _make(comm_env, world_size=1)
    manager = Mock()
    comm.all2all_manager = manager
    comm.destroy()
    manager.destroy.assert_called_once_with()
    assert comm.all2all_manager is None


@pytest.mark.supplementary
def test_destroy_without_all2all_manager_is_noop(comm_env):
    """destroy() with no manager attached is a safe no-op."""
    comm = _make(comm_env, world_size=1)
    comm.all2all_manager = None
    comm.destroy()  # must not raise


@pytest.mark.supplementary
def test_checkpoint_prepare_and_restore_delegate(comm_env):
    """checkpoint_prepare/checkpoint_restore forward to the manager."""
    comm = _make(comm_env, world_size=1)
    comm.all2all_manager = Mock()
    comm.checkpoint_prepare()
    comm.all2all_manager.checkpoint_prepare.assert_called_once_with()
    comm.checkpoint_restore()
    comm.all2all_manager.checkpoint_restore.assert_called_once_with()

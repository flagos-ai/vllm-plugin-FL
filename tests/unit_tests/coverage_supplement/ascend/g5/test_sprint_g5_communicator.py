# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""814b coverage for ``vllm_fl/distributed/communicator.py``.

``CommunicatorFL`` in 814b differs from ``main``: it calls
``DeviceCommunicatorBase.__init__`` directly (extra ``use_all2all`` kwarg),
raises on NPU/AMD when FlagCX fails to initialize, and selects one of many
``all2all_backend`` managers.  The tests below never build a real process
group: the base initializer is replaced, ``PyFlagcxCommunicator`` is faked, and
every vLLM all2all manager class is monkeypatched to a recorder, so the
dispatch logic is driven purely on CPU.

Self-contained: no imports from the shared ``coverage_tests`` helpers.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

MODULE_NAME = "vllm_fl.distributed.communicator"


def _load():
    try:
        return importlib.import_module(MODULE_NAME)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"communicator module not importable: {exc!r}")


try:
    _MODULE = importlib.import_module(MODULE_NAME)
except Exception:  # pragma: no cover - environment guard
    _MODULE = None


def _mod():
    if _MODULE is None:  # pragma: no cover - environment guard
        pytest.skip("communicator module not importable")
    return _MODULE


def _base_module():
    try:
        return importlib.import_module(
            "vllm.distributed.device_communicators.base_device_communicator"
        )
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"DeviceCommunicatorBase unavailable: {exc!r}")


def _all2all_module():
    try:
        return importlib.import_module(
            "vllm.distributed.device_communicators.all2all"
        )
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"vllm all2all module unavailable: {exc!r}")


def _has_npu() -> bool:
    try:
        probe = torch.zeros(1, device="npu")
        del probe
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _Group:
    def __init__(self, world_size=2):
        self.world_size = world_size


class _FakeLogger:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


def _fake_base_init(
    self,
    cpu_group,
    device=None,
    device_group=None,
    unique_name="",
    global_ranks=None,
    global_world_size=None,
    use_all2all=False,
    **kwargs,
):
    self.cpu_group = cpu_group
    self.device = device if device is not None else torch.device("cpu")
    self.device_group = device_group
    self.world_size = cpu_group.world_size
    self.rank_in_group = 0
    self.rank = 0
    self.ranks = list(range(self.world_size))
    self.use_all2all = use_all2all
    self.all2all_manager = None
    self.all2all_backend = "naive"


def _install_init(monkeypatch, mod):
    base_mod = _base_module()
    monkeypatch.setattr(
        base_mod.DeviceCommunicatorBase, "__init__", _fake_base_init
    )
    return base_mod


def _make_comm(mod, *, world_size=2, rank=0, pyflagcx=None, device="cpu"):
    comm = object.__new__(mod.CommunicatorFL)
    comm.world_size = world_size
    comm.rank_in_group = rank
    comm.rank = rank
    comm.device = torch.device(device)
    comm.device_group = object()
    comm.cpu_group = object()
    comm.ranks = list(range(world_size))
    comm.pyflagcx_comm = pyflagcx
    comm.all2all_manager = None
    return comm


class _FakePyFlagcx:
    def __init__(self, *, disabled=False):
        self.disabled = disabled
        self.all_gather_calls = []
        self.broadcast_calls = []

    def all_gather(self, output, input_):
        self.all_gather_calls.append((output, input_))

    def broadcast(self, tensor, src):
        self.broadcast_calls.append((tensor, src))


# ---------------------------------------------------------------------------
# Constructor: NPU/AMD failure escalation (target 50, 54, 55, 57-60)
# ---------------------------------------------------------------------------


class _DisabledCtor:
    def __init__(self, group=None, device=None):
        self.group = group
        self.device = device
        self.disabled = True
        self.init_error = RuntimeError("flagcx missing")


def _install_ctor(monkeypatch, mod, *, disabled=True):
    monkeypatch.setattr(mod, "PyFlagcxCommunicator", _DisabledCtor, raising=False)
    _install_init(monkeypatch, mod)


def test_constructor_raises_on_npu_when_flagcx_disabled(monkeypatch):
    """A disabled FlagCX on an NPU device escalates to RuntimeError('NPU')."""
    if not _has_npu():  # pragma: no cover - hardware guard
        pytest.skip("torch.npu is unavailable on this host")
    mod = _mod()
    _install_ctor(monkeypatch, mod)
    try:
        import vllm.platforms as platforms_mod
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"vllm.platforms unavailable: {exc!r}")
    monkeypatch.setattr(
        platforms_mod,
        "current_platform",
        SimpleNamespace(vendor_name="nvidia"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="NPU"):
        mod.CommunicatorFL(_Group(2), torch.device("npu"), None)


def test_constructor_raises_on_amd_when_flagcx_disabled(monkeypatch):
    """A disabled FlagCX with the AMD vendor escalates to RuntimeError('AMD')."""
    mod = _mod()
    _install_ctor(monkeypatch, mod)
    try:
        import vllm.platforms as platforms_mod
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"vllm.platforms unavailable: {exc!r}")
    monkeypatch.setattr(
        platforms_mod,
        "current_platform",
        SimpleNamespace(vendor_name="amd"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="AMD"):
        mod.CommunicatorFL(_Group(2), torch.device("cpu"), None)


def test_constructor_keeps_disabled_flagcx_on_other_vendors(monkeypatch):
    """On a CPU/NVIDIA host a disabled FlagCX is kept instead of raising."""
    mod = _mod()
    _install_ctor(monkeypatch, mod)
    try:
        import vllm.platforms as platforms_mod
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"vllm.platforms unavailable: {exc!r}")
    monkeypatch.setattr(
        platforms_mod,
        "current_platform",
        SimpleNamespace(vendor_name="nvidia"),
        raising=False,
    )

    comm = mod.CommunicatorFL(_Group(2), torch.device("cpu"), None)
    assert isinstance(comm.pyflagcx_comm, _DisabledCtor)
    assert comm.pyflagcx_comm.disabled is True


# ---------------------------------------------------------------------------
# _init_all2all_manager backend dispatch (target 77-144)
# ---------------------------------------------------------------------------

_ALL2ALL_NAMES = (
    "AgRsAll2AllManager",
    "DeepEPHTAll2AllManager",
    "DeepEPLLAll2AllManager",
    "MoriAll2AllManager",
    "DeepEPV2All2AllManager",
    "NixlEPAll2AllManager",
    "FlashInferNVLinkTwoSidedManager",
    "FlashInferNVLinkOneSidedManager",
)

_BACKENDS = [
    ("naive", "AgRsAll2AllManager", 2),
    ("allgather_reducescatter", "AgRsAll2AllManager", 2),
    ("deepep_high_throughput", "DeepEPHTAll2AllManager", 2),
    ("deepep_low_latency", "DeepEPLLAll2AllManager", 2),
    ("mori_high_throughput", "MoriAll2AllManager", 2),
    ("mori_low_latency", "MoriAll2AllManager", 2),
    ("deepep_v2", "DeepEPV2All2AllManager", 2),
    ("nixl_ep", "NixlEPAll2AllManager", 2),
    ("flashinfer_all2allv", "FlashInferNVLinkTwoSidedManager", 2),
    ("flashinfer_nvlink_two_sided", "FlashInferNVLinkTwoSidedManager", 2),
    ("flashinfer_nvlink_one_sided", "FlashInferNVLinkOneSidedManager", 1),
]


def _manager_factory(name, recorder):
    class _Manager:
        def __init__(self, *args, **kwargs):
            recorder.append((name, args, kwargs))

        def destroy(self):
            return None

        def checkpoint_prepare(self):
            return None

        def checkpoint_restore(self):
            return None

    _Manager.__name__ = name
    return _Manager


@pytest.mark.parametrize("backend,expected_arg,min_args", _BACKENDS)
def test_init_all2all_manager_dispatch(monkeypatch, backend, expected_arg, min_args):
    """Each backend string constructs only its matching manager."""
    mod = _mod()
    all2all_mod = _all2all_module()
    recorder = []
    for name in _ALL2ALL_NAMES:
        monkeypatch.setattr(
            all2all_mod, name, _manager_factory(name, recorder), raising=False
        )
    monkeypatch.setattr(mod, "logger", _FakeLogger(), raising=False)

    comm = object.__new__(mod.CommunicatorFL)
    comm.cpu_group = _Group(2)
    comm.device_group = object()
    comm.all2all_backend = backend

    comm._init_all2all_manager(object())

    assert recorder, f"{backend} constructed no manager"
    used, args, kwargs = recorder[0]
    assert used == expected_arg
    assert len(args) >= min_args
    assert args[0] is comm.cpu_group
    if backend in ("mori_high_throughput", "mori_low_latency"):
        assert args[1] == backend
    if backend == "deepep_v2":
        assert kwargs.get("device_group") is comm.device_group
    assert comm.all2all_manager is not None


def test_init_all2all_manager_unknown_backend(monkeypatch):
    """An unrecognized backend raises instead of silently picking a manager."""
    mod = _mod()
    monkeypatch.setattr(mod, "logger", _FakeLogger(), raising=False)

    comm = object.__new__(mod.CommunicatorFL)
    comm.cpu_group = _Group(2)
    comm.device_group = object()
    comm.all2all_backend = "does_not_exist"

    with pytest.raises(ValueError, match="Unknown all2all backend"):
        comm._init_all2all_manager(object())


# ---------------------------------------------------------------------------
# all_gather (target 165-186)
# ---------------------------------------------------------------------------


def test_all_gather_single_rank_returns_input():
    """world_size == 1 returns the input tensor untouched."""
    mod = _mod()
    comm = _make_comm(mod, world_size=1, pyflagcx=_FakePyFlagcx())
    tensor = torch.arange(3)
    assert comm.all_gather(tensor) is tensor


def test_all_gather_disabled_falls_back_to_base(monkeypatch):
    """A missing/disabled FlagCX delegates to DeviceCommunicatorBase."""
    mod = _mod()
    base_mod = _base_module()
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        base_mod.DeviceCommunicatorBase,
        "all_gather",
        lambda self, input_, dim: calls.append((input_, dim)) or sentinel,
        raising=False,
    )
    for pyflagcx in (None, _FakePyFlagcx(disabled=True)):
        comm = _make_comm(mod, pyflagcx=pyflagcx)
        assert comm.all_gather(torch.arange(3), dim=0) is sentinel
    assert len(calls) == 2


@pytest.mark.parametrize(
    "dim,expected_shape",
    [(0, (4, 3)), (-1, (2, 6))],
)
def test_all_gather_enabled_builds_output(dim, expected_shape):
    """An enabled FlagCX gathers into a world-scaled output buffer."""
    mod = _mod()
    fake = _FakePyFlagcx()
    comm = _make_comm(mod, world_size=2, pyflagcx=fake)
    base = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    inp = base[::2]  # (2, 3), non-contiguous

    out = comm.all_gather(inp, dim=dim)

    assert out.shape == expected_shape
    assert len(fake.all_gather_calls) == 1
    passed_out, passed_in = fake.all_gather_calls[0]
    assert passed_out.shape[0] == 4
    assert passed_in.is_contiguous()
    assert torch.equal(passed_in, inp.contiguous())


# ---------------------------------------------------------------------------
# broadcast (target 279, 280, 282-286)
# ---------------------------------------------------------------------------


def test_broadcast_single_rank_returns_input():
    """world_size == 1 returns the tensor without dispatching."""
    mod = _mod()
    fake = _FakePyFlagcx()
    comm = _make_comm(mod, world_size=1, pyflagcx=fake)
    tensor = torch.zeros(2)
    assert comm.broadcast(tensor, src=0) is tensor
    assert fake.broadcast_calls == []


def test_broadcast_enabled_delegates_and_returns_tensor():
    """An enabled FlagCX broadcasts and returns the same tensor."""
    mod = _mod()
    fake = _FakePyFlagcx()
    comm = _make_comm(mod, world_size=2, pyflagcx=fake)
    tensor = torch.zeros(2)

    result = comm.broadcast(tensor, src=1)

    assert result is tensor
    assert fake.broadcast_calls == [(tensor, 1)]


def test_broadcast_disabled_falls_back_to_base(monkeypatch):
    """A disabled FlagCX falls back to DeviceCommunicatorBase.broadcast."""
    mod = _mod()
    base_mod = _base_module()
    sentinel = object()
    monkeypatch.setattr(
        base_mod.DeviceCommunicatorBase,
        "broadcast",
        lambda self, tensor, src: sentinel,
        raising=False,
    )
    comm = _make_comm(mod, world_size=2, pyflagcx=_FakePyFlagcx(disabled=True))
    assert comm.broadcast(torch.zeros(2), src=0) is sentinel


# ---------------------------------------------------------------------------
# checkpoint hooks (target 298, 299, 302, 303)
# ---------------------------------------------------------------------------


class _CheckpointManager:
    def __init__(self):
        self.prepared = 0
        self.restored = 0

    def checkpoint_prepare(self):
        self.prepared += 1

    def checkpoint_restore(self):
        self.restored += 1


def test_checkpoint_hooks_delegate_to_manager():
    """checkpoint_prepare/restore forward to the all2all manager."""
    mod = _mod()
    manager = _CheckpointManager()
    comm = _make_comm(mod, pyflagcx=None)
    comm.all2all_manager = manager

    comm.checkpoint_prepare()
    comm.checkpoint_restore()

    assert manager.prepared == 1
    assert manager.restored == 1


def test_checkpoint_hooks_noop_without_manager():
    """Without an all2all manager the hooks are no-ops."""
    mod = _mod()
    comm = _make_comm(mod, pyflagcx=None)
    comm.all2all_manager = None
    assert comm.checkpoint_prepare() is None
    assert comm.checkpoint_restore() is None

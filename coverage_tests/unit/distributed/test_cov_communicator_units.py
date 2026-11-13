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

"""CPU unit tests for ``CommunicatorFL``'s collective wrappers.

The real device communicators need GPU/NCCL/flagcx, so every external
boundary is faked here: the flagcx communicator, the process groups, and
the ``torch.distributed`` fallbacks.  The fake flagcx implements the
documented reductions over independently supplied per-rank payloads, so
each assertion checks the communicator's own chunking, padding/trimming,
dtype/device propagation and call order against values derived in the
test, not against the code under test.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)

# Imported dynamically so the shim above always runs first.
DeviceCommunicatorBase = importlib.import_module(
    "vllm.distributed.device_communicators.base_device_communicator"
).DeviceCommunicatorBase
Mod = importlib.import_module("vllm_fl.distributed.communicator")
CommunicatorFL = Mod.CommunicatorFL


class _FakeFlagcx:
    """Python-side stand-in for ``PyFlagcxCommunicator``.

    ``payloads`` is a list of per-collective payloads; each payload is the
    list of resident row-blocks for ranks ``0..world_size-1``.  The fake
    concatenates (all-gather) or reduces (reduce-scatter) them exactly as
    the collective contract specifies.
    """

    def __init__(
        self,
        *,
        world_size=2,
        rank=0,
        disabled=False,
        payloads=None,
        reduce_peers=None,
        reducev_peers=None,
    ):
        self.world_size = world_size
        self.rank = rank
        self.disabled = disabled
        self.events = []
        self.calls = []
        self._payloads = list(payloads or [])
        self.reduce_peers = reduce_peers
        self.reducev_peers = reducev_peers
        self.all_reduce_result = None
        self.recv_fill = None
        self.error = None

    def _next_payload(self):
        return self._payloads.pop(0)

    def _maybe_raise(self):
        if self.error is not None:
            raise self.error

    def all_reduce(self, input_):
        self.events.append("all_reduce")
        self.calls.append(("all_reduce", input_))
        self._maybe_raise()
        return self.all_reduce_result

    def all_gather(self, output, input_):
        self.events.append("all_gather")
        self.calls.append(("all_gather", output, input_, None))
        self._maybe_raise()
        rows = self._next_payload()
        offset = 0
        for row in rows:
            output[offset : offset + row.shape[0]].copy_(row)
            offset += row.shape[0]

    def all_gatherv(self, output, input_, sizes=None):
        self.events.append("all_gatherv")
        self.calls.append(("all_gatherv", output, input_, sizes))
        self._maybe_raise()
        rows = self._next_payload()
        offset = 0
        for row in rows:
            output[offset : offset + row.shape[0]].copy_(row)
            offset += row.shape[0]

    def reduce_scatter(self, output, input_):
        self.events.append("reduce_scatter")
        self.calls.append(("reduce_scatter", output, input_, None))
        self._maybe_raise()
        chunk = output.shape[0]
        start = self.rank * chunk
        reduced = input_[start : start + chunk].clone()
        for peer in (self.reduce_peers or [input_])[1:]:
            reduced += peer[start : start + chunk]
        output.copy_(reduced)

    def reduce_scatterv(self, output, input_, sizes=None):
        self.events.append("reduce_scatterv")
        self.calls.append(("reduce_scatterv", output, input_, sizes))
        self._maybe_raise()
        offset = sum(sizes[: self.rank])
        size = sizes[self.rank]
        reduced = torch.zeros(
            (size,) + input_.shape[1:], dtype=input_.dtype, device=input_.device
        )
        peers = self.reducev_peers or [input_]
        for peer in peers:
            reduced += peer[offset : offset + size]
        output.copy_(reduced)

    def send(self, tensor, dst):
        self.events.append("send")
        self.calls.append(("send", tensor, dst))

    def recv(self, tensor, src):
        self.events.append("recv")
        self.calls.append(("recv", tensor, src))
        if self.recv_fill is not None:
            tensor.copy_(self.recv_fill)

    def group_start(self):
        self.events.append("group_start")

    def group_end(self):
        self.events.append("group_end")


class _FakeAll2All:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []
        self.destroyed = False

    def dispatch(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        is_sequence_parallel=False,
        extra_tensors=None,
    ):
        self.calls.append(
            (
                "dispatch",
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
            )
        )
        return "dispatched"

    def combine(self, hidden_states, is_sequence_parallel=False):
        self.calls.append(("combine", hidden_states, is_sequence_parallel))
        return "combined"

    def destroy(self):
        self.destroyed = True


def _make_comm(*, world_size=2, rank=0, pyflagcx=None, all2all=None, device="cpu"):
    comm = object.__new__(CommunicatorFL)
    comm.world_size = world_size
    comm.rank = rank
    comm.rank_in_group = rank
    comm.device = torch.device(device)
    comm.device_group = object()
    comm.cpu_group = object()
    comm.ranks = list(range(world_size))
    comm.pyflagcx_comm = pyflagcx
    comm.all2all_manager = all2all
    comm.use_all2all = all2all is not None
    return comm


# ---------------------------------------------------------------------------
# all_reduce
# ---------------------------------------------------------------------------


def test_all_reduce_returns_flagcx_result():
    """Checks that all_reduce returns the FlagCX result and passes the input through."""
    expected = torch.tensor([1.0, 2.0])
    fake = _FakeFlagcx()
    fake.all_reduce_result = expected
    comm = _make_comm(pyflagcx=fake)
    inp = torch.tensor([0.5, 0.5])

    assert comm.all_reduce(inp) is expected
    assert fake.calls == [("all_reduce", inp)]


def test_all_reduce_falls_back_to_torch_when_flagcx_returns_none(monkeypatch):
    """Checks that all_reduce falls back to torch.distributed on a copy when FlagCX returns None."""
    fake = _FakeFlagcx()
    fake.all_reduce_result = None
    comm = _make_comm(pyflagcx=fake)
    inp = torch.tensor([1.0, -2.0, 3.0])
    recorded = []

    def fake_all_reduce(tensor, group=None):
        recorded.append((tensor, group))
        tensor.add_(10.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    out = comm.all_reduce(inp)

    assert out is not inp
    assert torch.equal(out, torch.tensor([11.0, 8.0, 13.0]))
    assert torch.equal(inp, torch.tensor([1.0, -2.0, 3.0]))
    assert recorded == [(out, comm.device_group)]


def test_all_reduce_requires_flagcx():
    """Checks that all_reduce asserts when no FlagCX communicator exists."""
    comm = _make_comm(pyflagcx=None)
    with pytest.raises(AssertionError):
        comm.all_reduce(torch.zeros(2))


# ---------------------------------------------------------------------------
# reduce_scatter / reduce_scatterv
# ---------------------------------------------------------------------------


def _expected_reduce_scatter(input_, dim, world_size, rank):
    transformed = input_.movedim(0, dim).contiguous()
    chunk = transformed.shape[0] // world_size
    start = rank * chunk
    return (
        transformed[start : start + chunk].movedim(0, dim).contiguous(),
        (transformed),
        chunk,
    )


def test_reduce_scatter_negative_dim_matches_closed_form():
    """Checks reduce_scatter with a negative dim against the closed-form chunk reduction and buffer layout."""
    world_size, rank = 2, 1
    inp = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    transformed_peers = [
        inp.movedim(0, 1).contiguous(),
        (inp + 1000.0).movedim(0, 1).contiguous(),
    ]
    fake = _FakeFlagcx(world_size=world_size, rank=rank, reduce_peers=transformed_peers)
    comm = _make_comm(world_size=world_size, rank=rank, pyflagcx=fake)

    out = comm.reduce_scatter(inp, dim=-1)

    chunk = transformed_peers[0].shape[0] // world_size
    expected_chunk = transformed_peers[0][rank * chunk : (rank + 1) * chunk]
    expected_chunk = (
        expected_chunk + transformed_peers[1][rank * chunk : (rank + 1) * chunk]
    )
    expected = expected_chunk.movedim(0, 1).contiguous()
    assert torch.equal(out, expected)
    assert out.shape == (4, 3)
    assert out.dtype == inp.dtype
    assert out.device == inp.device
    _, passed_out, passed_in, _ = fake.calls[-1]
    assert passed_out is not out  # pre-reshape buffer was handed to flagcx
    assert torch.equal(passed_in, transformed_peers[0])
    assert passed_in.is_contiguous()
    assert passed_in.shape == (6, 4)
    assert passed_out.shape == (chunk, 4)


def test_reduce_scatter_dim_zero_uses_leading_axis():
    """Checks that reduce_scatter with dim zero chunks along the leading axis."""
    world_size, rank = 3, 0
    inp = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    fake = _FakeFlagcx(world_size=world_size, rank=rank)
    comm = _make_comm(world_size=world_size, rank=rank, pyflagcx=fake)

    out = comm.reduce_scatter(inp, dim=0)

    expected, _, _ = _expected_reduce_scatter(inp, 0, world_size, rank)
    assert torch.equal(out, expected)
    assert out.shape == (2, 2)
    assert out.dtype == torch.float64


def test_reduce_scatter_contiguous_copy_for_noncontiguous_input():
    """Checks that reduce_scatter copies non-contiguous input into a contiguous buffer."""
    world_size, rank = 2, 0
    inp = torch.arange(48, dtype=torch.float32).reshape(4, 12)[:, ::2]
    assert inp.shape == (4, 6)
    assert not inp.is_contiguous()
    fake = _FakeFlagcx(world_size=world_size, rank=rank)
    comm = _make_comm(world_size=world_size, rank=rank, pyflagcx=fake)

    out = comm.reduce_scatter(inp, dim=-1)

    expected, transformed, _ = _expected_reduce_scatter(inp, 1, world_size, rank)
    assert torch.equal(out, expected)
    _, _, passed_in, _ = fake.calls[-1]
    assert passed_in.is_contiguous()
    assert torch.equal(passed_in, transformed)


def test_reduce_scatter_rejects_indivisible_first_axis():
    """Checks that reduce_scatter asserts when the first axis is not divisible by world size."""
    fake = _FakeFlagcx(world_size=2, rank=0)
    comm = _make_comm(pyflagcx=fake)
    with pytest.raises(AssertionError):
        comm.reduce_scatter(torch.zeros(5, 3), dim=0)


def test_reduce_scatter_requires_flagcx():
    """Checks that reduce_scatter asserts when no FlagCX communicator exists."""
    comm = _make_comm(pyflagcx=None)
    with pytest.raises(AssertionError):
        comm.reduce_scatter(torch.zeros(4, 3))


def test_reduce_scatter_propagates_flagcx_errors():
    """Checks that reduce_scatter propagates errors raised by FlagCX."""
    fake = _FakeFlagcx(world_size=2, rank=0)
    fake.error = RuntimeError("flagcx failure")
    comm = _make_comm(pyflagcx=fake)
    with pytest.raises(RuntimeError, match="flagcx failure"):
        comm.reduce_scatter(torch.zeros(4, 4))


def test_reduce_scatterv_uses_rank_chunk_sizes():
    """Checks that reduce_scatterv reduces per-rank chunk sizes and forwards the sizes list."""
    world_size, rank = 2, 1
    sizes = [2, 3]
    inp = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    peers = [inp, inp + 100, inp + 200]
    fake = _FakeFlagcx(world_size=world_size, rank=rank, reducev_peers=peers)
    comm = _make_comm(world_size=world_size, rank=rank, pyflagcx=fake)

    out = comm.reduce_scatterv(inp, dim=0, sizes=sizes)

    expected = (peers[0] + peers[1] + peers[2])[2:5]
    assert torch.equal(out, expected)
    assert out.shape == (3, 4)
    _, _, passed_in, passed_sizes = fake.calls[-1]
    assert passed_sizes is sizes
    assert passed_in.is_contiguous()
    assert torch.equal(passed_in, inp)


def test_reduce_scatterv_negative_dim_and_even_split_fallback():
    """Checks that reduce_scatterv with no sizes falls back to an even reduce_scatter along a negative dim."""
    world_size, rank = 2, 0
    inp = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    fake = _FakeFlagcx(world_size=world_size, rank=rank)
    comm = _make_comm(world_size=world_size, rank=rank, pyflagcx=fake)

    out = comm.reduce_scatterv(inp, dim=-1, sizes=None)

    expected, _, _ = _expected_reduce_scatter(inp, 1, world_size, rank)
    assert torch.equal(out, expected)
    assert fake.calls[-1][0] == "reduce_scatter"
    assert fake.calls[-1][3] is None


def test_reduce_scatterv_validates_sizes():
    """Checks that reduce_scatterv rejects a sizes list of the wrong length or total."""
    fake = _FakeFlagcx(world_size=2, rank=0)
    comm = _make_comm(pyflagcx=fake)
    with pytest.raises(AssertionError):
        comm.reduce_scatterv(torch.zeros(4, 3), dim=0, sizes=[2])
    with pytest.raises(AssertionError):
        comm.reduce_scatterv(torch.zeros(4, 3), dim=0, sizes=[1, 2])


# ---------------------------------------------------------------------------
# all_gatherv
# ---------------------------------------------------------------------------


def test_all_gatherv_uneven_sizes_pads_and_trims():
    """Checks that all_gatherv gathers uneven sizes into one padded output and forwards local input and sizes."""
    world_size, rank = 3, 1
    sizes = [2, 3, 1]
    payload = [
        torch.arange(8, dtype=torch.float64).reshape(2, 4),
        torch.arange(100, 112, dtype=torch.float64).reshape(3, 4),
        torch.tensor([[500.0, 501.0, 502.0, 503.0]], dtype=torch.float64),
    ]
    local = payload[rank]
    fake = _FakeFlagcx(world_size=world_size, rank=rank, payloads=[payload])
    comm = _make_comm(world_size=world_size, rank=rank, pyflagcx=fake)

    out = comm.all_gatherv(local, dim=0, sizes=sizes)

    assert torch.equal(out, torch.cat(payload, dim=0))
    assert out.shape == (6, 4)
    assert out.dtype == torch.float64
    assert out.device == local.device
    name, passed_out, passed_in, passed_sizes = fake.calls[0]
    assert name == "all_gatherv"
    assert passed_out is out
    assert passed_in is local
    assert passed_sizes is sizes


def test_all_gatherv_equal_sizes_falls_back_to_all_gather():
    """Checks that all_gatherv uses plain all_gather when all sizes are equal."""
    world_size, rank = 3, 2
    sizes = [2, 2, 2]
    payload = [
        torch.full((2, 2), float(r), dtype=torch.float32) for r in range(world_size)
    ]
    fake = _FakeFlagcx(world_size=world_size, rank=rank, payloads=[payload])
    comm = _make_comm(world_size=world_size, rank=rank, pyflagcx=fake)

    out = comm.all_gatherv(payload[rank], dim=0, sizes=sizes)

    assert torch.equal(out, torch.cat(payload, dim=0))
    assert out.shape == (6, 2)
    assert fake.calls[0][0] == "all_gather"
    assert fake.calls[0][3] is None


def test_all_gatherv_without_sizes_multiplies_first_axis():
    """Checks that all_gatherv without sizes uses all_gather and scales the first axis by world size."""
    world_size, rank = 2, 0
    payload = [
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(6, 12, dtype=torch.float32).reshape(2, 3),
    ]
    fake = _FakeFlagcx(world_size=world_size, rank=rank, payloads=[payload])
    comm = _make_comm(world_size=world_size, rank=rank, pyflagcx=fake)

    out = comm.all_gatherv(payload[rank])

    assert torch.equal(out, torch.cat(payload, dim=0))
    assert out.shape == (4, 3)
    assert fake.calls[0][0] == "all_gather"


def test_all_gatherv_list_uses_group_and_preserves_order():
    """Checks that all_gatherv over a list brackets the group and preserves list order."""
    world_size, rank = 2, 1
    first = [
        torch.full((1, 2), 1.0),
        torch.full((1, 2), 2.0),
    ]
    second = [
        torch.full((3, 2), 3.0),
        torch.full((3, 2), 4.0),
    ]
    fake = _FakeFlagcx(world_size=world_size, rank=rank, payloads=[first, second])
    comm = _make_comm(world_size=world_size, rank=rank, pyflagcx=fake)

    outputs = comm.all_gatherv([first[rank], second[rank]])

    assert isinstance(outputs, list)
    assert torch.equal(outputs[0], torch.cat(first, dim=0))
    assert torch.equal(outputs[1], torch.cat(second, dim=0))
    assert outputs[0].shape == (2, 2)
    assert outputs[1].shape == (6, 2)
    assert fake.events == [
        "group_start",
        "all_gather",
        "all_gather",
        "group_end",
    ]


def test_all_gatherv_empty_list_still_brackets_group():
    """Checks that all_gatherv over an empty list returns empty but still brackets the group."""
    fake = _FakeFlagcx(world_size=2, rank=0)
    comm = _make_comm(pyflagcx=fake)
    assert comm.all_gatherv([]) == []
    assert fake.events == ["group_start", "group_end"]


@pytest.mark.parametrize("dim", [1, -1])
def test_all_gatherv_rejects_nonzero_dim(dim):
    """Checks that all_gatherv rejects any dim other than zero."""
    fake = _FakeFlagcx(world_size=2, rank=0)
    comm = _make_comm(pyflagcx=fake)
    with pytest.raises(NotImplementedError, match="only dim 0"):
        comm.all_gatherv(torch.zeros(2, 2), dim=dim)


def test_all_gatherv_validates_sizes_against_world_and_rank():
    """Checks that all_gatherv asserts on a sizes list that does not match world size and rank."""
    fake = _FakeFlagcx(world_size=2, rank=0)
    comm = _make_comm(pyflagcx=fake)
    with pytest.raises(AssertionError):
        comm.all_gatherv(torch.zeros(2, 2), sizes=[2, 3, 2])
    with pytest.raises(AssertionError, match="2 != 3"):
        comm.all_gatherv(torch.zeros(2, 2), sizes=[3, 2])


def test_all_gatherv_requires_enabled_flagcx():
    """Checks that all_gatherv asserts when FlagCX is missing or disabled."""
    comm = _make_comm(pyflagcx=None)
    with pytest.raises(AssertionError):
        comm.all_gatherv(torch.zeros(2, 2))

    comm = _make_comm(pyflagcx=_FakeFlagcx(disabled=True))
    with pytest.raises(AssertionError):
        comm.all_gatherv(torch.zeros(2, 2))


# ---------------------------------------------------------------------------
# send / recv
# ---------------------------------------------------------------------------


def test_send_uses_flagcx_and_explicit_dst():
    """Checks that send forwards the tensor and explicit destination to FlagCX."""
    fake = _FakeFlagcx(world_size=3, rank=0)
    comm = _make_comm(world_size=3, rank=0, pyflagcx=fake)
    tensor = torch.arange(4)

    comm.send(tensor, dst=2)

    assert fake.calls == [("send", tensor, 2)]


def test_send_default_dst_wraps_to_zero():
    """Checks that send without a destination wraps from the last rank to rank zero."""
    fake = _FakeFlagcx(world_size=3, rank=2)
    comm = _make_comm(world_size=3, rank=2, pyflagcx=fake)
    tensor = torch.zeros(1)
    comm.send(tensor)
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "send"
    assert fake.calls[0][1] is tensor
    assert fake.calls[0][2] == 0


def test_send_default_dst_advances_to_next_rank():
    """Checks that send without a destination advances to the next rank."""
    fake = _FakeFlagcx(world_size=4, rank=1)
    comm = _make_comm(world_size=4, rank=1, pyflagcx=fake)
    tensor = torch.zeros(1)
    comm.send(tensor)
    assert len(fake.calls) == 1
    assert fake.calls[0][1] is tensor
    assert fake.calls[0][2] == 2


@pytest.mark.parametrize("fake_flagcx", ["none", "disabled"])
def test_send_falls_back_to_torch_with_global_rank(monkeypatch, fake_flagcx):
    """Checks that send falls back to torch.distributed with the global destination when FlagCX is missing or disabled."""
    if fake_flagcx == "none":
        pyflagcx = None
    else:
        pyflagcx = _FakeFlagcx(world_size=3, rank=1, disabled=True)
    comm = _make_comm(world_size=3, rank=1, pyflagcx=pyflagcx)
    comm.ranks = [10, 11, 12]
    recorded = []
    monkeypatch.setattr(
        torch.distributed,
        "send",
        lambda tensor, dst, group=None: recorded.append((tensor, dst, group)),
    )
    tensor = torch.zeros(2)

    comm.send(tensor, dst=2)

    assert recorded == [(tensor, 12, comm.device_group)]


def test_send_fallback_default_dst_uses_ranks_table(monkeypatch):
    """Checks that the send fallback default destination comes from the ranks table."""
    comm = _make_comm(world_size=2, rank=1, pyflagcx=None)
    comm.ranks = [7, 9]
    recorded = []
    monkeypatch.setattr(
        torch.distributed,
        "send",
        lambda tensor, dst, group=None: recorded.append(dst),
    )
    comm.send(torch.zeros(1))
    assert recorded == [7]


def test_recv_allocates_and_fills_tensor_via_flagcx():
    """Checks that recv allocates a tensor of the requested shape, dtype and device and fills it via FlagCX."""
    fake = _FakeFlagcx(world_size=2, rank=1)
    fake.recv_fill = torch.tensor([4.0, 5.0])
    comm = _make_comm(world_size=2, rank=1, pyflagcx=fake)

    got = comm.recv((2,), torch.float32, src=0)

    assert torch.equal(got, fake.recv_fill)
    assert got.dtype == torch.float32
    assert got.device == comm.device
    assert fake.calls[0][0] == "recv"
    assert fake.calls[0][2] == 0


def test_recv_default_src_wraps_to_last_rank():
    """Checks that recv without a source wraps from rank zero to the last rank."""
    fake = _FakeFlagcx(world_size=3, rank=0)
    comm = _make_comm(world_size=3, rank=0, pyflagcx=fake)
    comm.recv((1,), torch.float32)
    assert fake.calls[0][2] == 2


def test_recv_default_src_steps_back_one_rank():
    """Checks that recv without a source steps back one rank."""
    fake = _FakeFlagcx(world_size=4, rank=2)
    comm = _make_comm(world_size=4, rank=2, pyflagcx=fake)
    comm.recv((1,), torch.float32)
    assert fake.calls[0][2] == 1


@pytest.mark.parametrize("fake_flagcx", ["none", "disabled"])
def test_recv_falls_back_to_torch_with_global_rank(monkeypatch, fake_flagcx):
    """Checks that recv falls back to torch.distributed with the global source when FlagCX is missing or disabled."""
    if fake_flagcx == "none":
        pyflagcx = None
    else:
        pyflagcx = _FakeFlagcx(world_size=2, rank=0, disabled=True)
    comm = _make_comm(world_size=2, rank=0, pyflagcx=pyflagcx)
    comm.ranks = [20, 21]
    recorded = []

    def fake_recv(tensor, src, group=None):
        recorded.append((tensor, src, group))
        tensor.fill_(6.0)

    monkeypatch.setattr(torch.distributed, "recv", fake_recv)

    got = comm.recv((3,), torch.float64, src=1)

    assert got.dtype == torch.float64
    assert torch.equal(got, torch.full((3,), 6.0, dtype=torch.float64))
    assert recorded == [(got, 21, comm.device_group)]


# ---------------------------------------------------------------------------
# destroy / all2all delegation
# ---------------------------------------------------------------------------


def test_destroy_clears_flagcx_and_all2all_manager():
    """Checks that destroy clears both communicators and destroys the all2all manager."""
    fake_flagcx = _FakeFlagcx()
    manager = _FakeAll2All()
    comm = _make_comm(pyflagcx=fake_flagcx, all2all=manager)

    comm.destroy()

    assert comm.pyflagcx_comm is None
    assert comm.all2all_manager is None
    assert manager.destroyed is True


def test_destroy_is_safe_without_components():
    """Checks that destroy is a no-op when neither component exists."""
    comm = _make_comm(pyflagcx=None, all2all=None)
    comm.destroy()
    assert comm.pyflagcx_comm is None
    assert comm.all2all_manager is None


def test_dispatch_and_combine_delegate_to_manager():
    """Checks that dispatch and combine forward their arguments to the all2all manager."""
    manager = _FakeAll2All()
    comm = _make_comm(all2all=manager)
    hidden = torch.zeros(2, 2)
    weights = torch.zeros(2, 1)
    ids = torch.zeros(2, 1, dtype=torch.int64)
    extras = [torch.zeros(1)]

    assert comm.dispatch(hidden, weights, ids, True, extras) == "dispatched"
    assert manager.calls[0] == (
        "dispatch",
        hidden,
        weights,
        ids,
        True,
        extras,
    )
    assert comm.combine(hidden, True) == "combined"
    assert manager.calls[1] == ("combine", hidden, True)


@pytest.mark.parametrize("method", ["dispatch", "combine"])
def test_dispatch_combine_require_manager(method):
    """Checks that dispatch and combine assert when no all2all manager exists."""
    comm = _make_comm(all2all=None)
    if method == "dispatch":
        args = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    else:
        args = (torch.zeros(1),)
    with pytest.raises(AssertionError):
        getattr(comm, method)(*args)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class _PyFlagcxCtor:
    instances = []

    def __init__(self, *, group=None, device=None):
        self.group = group
        self.device = device
        self.disabled = False
        _PyFlagcxCtor.instances.append(self)


class _AgRsCtor:
    instances = []

    def __init__(self, cpu_group, tcp_store_group=None):
        self.cpu_group = cpu_group
        self.tcp_store_group = tcp_store_group
        _AgRsCtor.instances.append(self)


def _patch_init_dependencies(monkeypatch, *, use_all2all):
    def fake_base_init(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name="",
        global_ranks=None,
        global_world_size=None,
    ):
        self.cpu_group = cpu_group
        self.device = device or torch.device("cpu")
        self.device_group = device_group
        self.world_size = cpu_group.world_size
        self.rank_in_group = 0
        self.ranks = list(range(self.world_size))
        self.use_all2all = use_all2all
        self.all2all_manager = None

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base_init)
    _PyFlagcxCtor.instances = []
    _AgRsCtor.instances = []
    monkeypatch.setattr(Mod, "PyFlagcxCommunicator", _PyFlagcxCtor)
    import vllm.distributed.device_communicators.all2all as all2all_mod

    monkeypatch.setattr(all2all_mod, "AgRsAll2AllManager", _AgRsCtor)


def test_init_creates_flagcx_only_for_multirank_groups(monkeypatch):
    """Single-rank groups skip FlagCX; multirank groups build it with the group."""
    _patch_init_dependencies(monkeypatch, use_all2all=False)
    group = SimpleNamespace(world_size=1)
    device = torch.device("cpu")

    single = CommunicatorFL(group, device, None)
    assert single.world_size == 1
    assert single.pyflagcx_comm is None
    assert _PyFlagcxCtor.instances == []

    group2 = SimpleNamespace(world_size=2)
    multi = CommunicatorFL(group2, device, None)
    assert multi.world_size == 2
    assert multi.pyflagcx_comm is _PyFlagcxCtor.instances[0]
    assert multi.pyflagcx_comm.group is group2
    assert multi.pyflagcx_comm.device is device
    assert multi.all2all_manager is None


def test_init_builds_ag_rs_all2all_manager_when_enabled(monkeypatch):
    """When enabled, the all2all manager receives the CPU group and TCP store."""
    _patch_init_dependencies(monkeypatch, use_all2all=True)
    group = SimpleNamespace(world_size=2)
    store = object()

    comm = CommunicatorFL(group, torch.device("cpu"), None, tcp_store_group=store)

    assert len(_AgRsCtor.instances) == 1
    assert comm.all2all_manager is _AgRsCtor.instances[0]
    assert comm.all2all_manager.cpu_group is group
    assert comm.all2all_manager.tcp_store_group is store

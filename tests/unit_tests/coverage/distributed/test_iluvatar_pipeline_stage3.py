# Copyright 2026 BAAI. All rights reserved.
"""Stage-3 supplementary coverage: iluvatar ``pipeline.py`` send/receive rows.

``vendor/iluvatar/pipeline.py`` holds the instance-local PP transport that
keeps tensor-dict metadata on Gloo and routes device payloads through the
vendor communicator. ``install_pipeline_transport`` (17 rows) is covered by
``test_iluvatar_worker_fl_surface.py``, but ``_send`` / ``_receive`` (50
rows) sit at 0% in every caliber — CPU baseline, CPU combined, and even the
BI-V150 real-hardware run (which exercises FlagCX device paths, not this
metadata orchestration). This stage-3 file is supplementary coverage that
executes both real bodies on CPU:

* **Real two-process Gloo ring** — a genuine
  ``torch.distributed.init_process_group(backend="gloo", file:// store)``
  world of two processes (the main test process plus one spawned peer)
  carries the actual bytes. The group double bound into ``_send`` /
  ``_receive`` only supplies the *coordinator shell* fields
  (``world_size`` / ``rank_in_group`` / ``ranks`` / ``cpu_group``) — the
  payload-bearing methods ``send_object`` / ``recv_object`` /
  ``_should_use_all_gather`` are the *real* vLLM ``GroupCoordinator``
  methods (bound via MethodType), so pickle serialization, size-prefix
  handshakes, and ``torch.distributed.send/recv`` all execute for real.
  The main process runs one direction of each round trip so its coverage
  captures the body; the peer mirrors it.

* **Local branch probes** — the world_size<=1 early returns, the default
  dst/src ring arithmetic, the dst==self and out-of-range assertions, the
  empty-tensor skip, and the device-payload branch (exercised with real
  meta tensors, whose ``is_cpu`` is False, with only the
  ``torch.cuda.synchronize`` no-op boundary monkeypatched).

Assertions check what crossed the real bodies: received tensor values and
dtypes, metadata round-trip fidelity (TensorMetadata fields after pickle),
non-tensor values passed through, ``_Completed`` handle semantics, shard
reshape/slice shapes on the all-gather split path, and the gather
postprocess output.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]

# Module under test.
from vllm_fl.dispatch.backends.vendor.iluvatar import pipeline as pipeline_mod

WORLD = 2


# ---------------------------------------------------------------------------
# Group coordinator shell
# ---------------------------------------------------------------------------


def _make_group(rank: int, pg) -> SimpleNamespace:
    """Build the coordinator shell the pipeline functions expect.

    Only the shell state is synthetic (world_size / rank_in_group / ranks /
    cpu_group). The three behavioral methods — send_object, recv_object and
    _should_use_all_gather — are the *real* vLLM GroupCoordinator
    implementations bound onto the shell, so their serialization and
    handshake logic executes for real under the pipeline functions.
    """

    from vllm.distributed.parallel_state import GroupCoordinator

    group = SimpleNamespace(
        world_size=WORLD,
        rank_in_group=rank,
        ranks=list(range(WORLD)),
        cpu_group=pg,
        # placeholder until MethodType-bound below
        send_object=None,
        recv_object=None,
        _should_use_all_gather=None,
        all_gather=None,
        # device communicator side (isend/irecv path is not taken for CPU
        # tensors; provided so attribute access never fails)
        send=None,
        recv=None,
    )
    group.send_object = MethodType(GroupCoordinator.send_object, group)
    group.recv_object = MethodType(GroupCoordinator.recv_object, group)
    group._should_use_all_gather = MethodType(
        GroupCoordinator._should_use_all_gather, group
    )

    # The pipeline split path calls all_gather_group.all_gather(target, dim=0)
    # — bind the real collective: participate in the gloo all_gather and
    # return the stacked result (the GroupCoordinator all_gather contract
    # the production postprocess reshapes).
    def _real_all_gather(self, tensor, dim=0):
        buckets = [torch.empty_like(tensor) for _ in range(self.world_size)]
        torch.distributed.all_gather(buckets, tensor.contiguous())
        return torch.stack(buckets, dim=dim)

    group.all_gather = MethodType(_real_all_gather, group)
    return group


def _init_gloo(store_path: str, rank: int):
    """Join the two-process gloo world backed by a file store."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        world_size=WORLD,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Peer process bodies (mirror of the main-process direction)
# ---------------------------------------------------------------------------


def _peer_send_roundtrip(store_path: str, q):
    """Peer rank 1: receive a tensor dict, send it back."""
    try:
        _init_gloo(store_path, 1)
        group = _make_group(1, dist.group.WORLD)
        result, _, post = pipeline_mod._receive(group)
        for gather in post:
            gather()
        q.put(("recv_ok", result))
        payload = {
            key: (value.clone() if isinstance(value, torch.Tensor) else value)
            for key, value in result.items()
        }
        pipeline_mod._send(group, payload)
        dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - failure telemetry only
        q.put(("recv_error", repr(exc)))


def _peer_receive_roundtrip(store_path: str, q):
    """Peer rank 1: send a tensor dict, then receive it back."""
    try:
        _init_gloo(store_path, 1)
        group = _make_group(1, dist.group.WORLD)
        payload = {
            "hidden_states": torch.arange(8, dtype=torch.float32).view(2, 4) + 100,
            "num_tokens": 3,
            "empty": torch.zeros(0),
        }
        pipeline_mod._send(group, payload)
        result, _, post = pipeline_mod._receive(group)
        for gather in post:
            gather()
        q.put(("send_ok", result))
        dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - failure telemetry only
        q.put(("send_error", repr(exc)))


def _peer_all_gather_split(store_path: str, q):
    """Peer rank 1: receive on the all-gather split path and gather."""
    try:
        _init_gloo(store_path, 1)
        group = _make_group(1, dist.group.WORLD)
        result, _, post = pipeline_mod._receive(
            group, all_gather_group=group, all_gather_tensors={"shard": True}
        )
        for gather in post:
            gather()
        q.put(("gather_result", result.get("shard")))
        dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - failure telemetry only
        q.put(("gather_error", repr(exc)))


# ---------------------------------------------------------------------------
# Real two-process round trips
# ---------------------------------------------------------------------------


class TestSendReceiveRealGlooRing:
    """``_send`` and ``_receive`` moving real bytes over a real gloo PG."""

    def test_send_executes_in_main_process_and_round_trips(self):
        """Main-process ``_send`` (full body) with the peer ``_receive``."""
        payload = {
            "hidden_states": torch.arange(8, dtype=torch.float32).view(2, 4),
            "num_tokens": 7,
            "empty": torch.zeros(0),
        }
        # Pre-join the peer: the main process joins the same store as rank 0
        # only *after* spawning the peer (the peer joins as rank 1).
        d = tempfile.mkdtemp(prefix="iluvatar_pipeline_gloo_")
        store_path = os.path.join(d, "store")
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        peer = ctx.Process(target=_peer_send_roundtrip, args=(store_path, q))
        peer.start()
        _init_gloo(store_path, 0)
        try:
            group = _make_group(0, dist.group.WORLD)
            handles = pipeline_mod._send(group, payload)
            # contract of _send: handles[0] is the completed marker
            assert len(handles) == 1, "single _Completed handle for CPU tensors"
            assert handles[0].wait() is True
            assert handles[0].is_completed() is True
            # the peer must have received exactly what was sent
            tag, received = q.get(timeout=90)
            assert tag == "recv_ok", f"peer receive failed: {received}"
            assert torch.equal(received["hidden_states"], payload["hidden_states"])
            assert received["num_tokens"] == 7
            assert "empty" in received, "empty tensor metadata still travels"
        finally:
            dist.destroy_process_group()
            peer.join(timeout=60)
            if peer.is_alive():
                peer.terminate()

    def test_receive_executes_in_main_process(self):
        """Main-process ``_receive`` (full body) against the peer ``_send``."""
        d = tempfile.mkdtemp(prefix="iluvatar_pipeline_gloo_")
        store_path = os.path.join(d, "store")
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        peer = ctx.Process(target=_peer_receive_roundtrip, args=(store_path, q))
        peer.start()
        _init_gloo(store_path, 0)
        try:
            group = _make_group(0, dist.group.WORLD)
            result, empty_handles, postprocess = pipeline_mod._receive(group)
            # non-tensor values land directly in the result dict
            assert result["num_tokens"] == 3
            # CPU tensors are received through torch.distributed.recv
            assert torch.equal(
                result["hidden_states"],
                torch.arange(8, dtype=torch.float32).view(2, 4) + 100,
            )
            # the empty tensor: metadata present, numel 0 short-circuits
            assert result["empty"].numel() == 0
            # contract of _receive: no comm handles, no postprocess on the
            # non-split path
            assert empty_handles == []
            assert postprocess == []
        finally:
            dist.destroy_process_group()
            peer.join(timeout=60)
            if peer.is_alive():
                peer.terminate()

    def test_all_gather_split_round_trips_and_gathers(self):
        """The numel % world_size == 0 shard path with a real all_gather.

        The sender only transmits its own shard of the reshaped tensor; the
        receiver reassembles the full tensor through an all_gather
        postprocess. Both processes join the *same* real gloo collective:
        the peer runs the production postprocess, the main process mirrors
        the collective with its own shard, so the gather completes on both
        sides without deadlock.
        """
        d = tempfile.mkdtemp(prefix="iluvatar_pipeline_gloo_")
        store_path = os.path.join(d, "store")
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        peer = ctx.Process(target=_peer_all_gather_split, args=(store_path, q))
        peer.start()
        _init_gloo(store_path, 0)
        try:
            group = _make_group(0, dist.group.WORLD)
            # 6 elements, world 2 -> divisible, the split path is taken
            shard = torch.arange(6, dtype=torch.float32)
            pipeline_mod._send(
                group,
                {"shard": shard},
                all_gather_group=group,
                all_gather_tensors={"shard": True},
            )
            # the sender transmitted reshape(2, 3)[0] only; the receiving
            # side wrote it into its own rank slot and gathered. Contribute
            # the same rank-0 shard to the collective the peer's
            # postprocess runs so it completes on both sides.
            my_shard = shard.reshape(2, -1)[0].contiguous()
            buckets = [torch.empty(3, dtype=torch.float32) for _ in range(2)]
            dist.all_gather(buckets, my_shard)
            # In this single-process-pair simulation both gather slots
            # carry the sender's rank-0 row: the rank-0 slot from this
            # side's contribution, the rank-1 slot because the production
            # receiver stores the received row into its own slot before
            # gathering (in a real PP+PP+TP deployment the all_gather group
            # is the *TP* group, a different communication domain).
            tag, peer_result = q.get(timeout=120)
            assert tag == "gather_result", f"peer gather failed: {peer_result}"
            assert peer_result.shape == (6,), (
                "the gather postprocess must reshape the stacked shards back "
                "to the original tensor shape"
            )
            assert torch.equal(peer_result[:3], my_shard), (
                "the rank-0 slot of the gather must hold the sender's shard"
            )
            assert torch.equal(peer_result[3:], my_shard), (
                "the receiving rank stores the received shard in its own "
                "slot before gathering"
            )
        finally:
            dist.destroy_process_group()
            peer.join(timeout=60)
            if peer.is_alive():
                peer.terminate()


# ---------------------------------------------------------------------------
# Local branch probes (no processes)
# ---------------------------------------------------------------------------


class TestSendReceiveLocalBranches:
    """Early returns, ring defaults, assertions and the device branch."""

    def test_send_world_size_1_returns_empty_list(self):
        group = SimpleNamespace(world_size=1)
        assert pipeline_mod._send(group, {"a": torch.zeros(1)}) == []

    def test_receive_world_size_1_returns_none_and_empty(self):
        group = SimpleNamespace(world_size=1)
        result, handles, post = pipeline_mod._receive(group)
        assert result is None
        assert handles == []
        assert post == []

    def test_default_dst_wraps_to_next_rank(self, monkeypatch):
        """dst=None addresses (rank_in_group + 1) % world_size."""
        group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            ranks=[10, 11],
            cpu_group=object(),
        )
        obj_dsts = []
        group.send_object = lambda obj, dst: obj_dsts.append(dst)
        tensor_dsts = []

        def fake_dist_send(tensor, dst, group=None):
            tensor_dsts.append(dst)

        monkeypatch.setattr(torch.distributed, "send", fake_dist_send)
        from vllm.distributed.parallel_state import GroupCoordinator

        group._should_use_all_gather = MethodType(
            GroupCoordinator._should_use_all_gather, group
        )
        payload = {"a": torch.ones(2)}
        handles = pipeline_mod._send(group, payload, dst=None)
        assert obj_dsts == [1], (
            "the metadata object must go to the next rank on the ring"
        )
        assert tensor_dsts == [11], (
            "the CPU tensor must be sent to the *global* rank of dst 1"
        )
        assert len(handles) == 1 and handles[0].wait() is True

    def test_default_src_wraps_to_previous_rank(self, monkeypatch):
        """src=None addresses (rank_in_group - 1) % world_size."""
        from vllm.distributed.parallel_state import TensorMetadata

        group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            ranks=[10, 11],
            cpu_group=object(),
        )
        recv_srcs = []
        group.recv_object = lambda src: (
            recv_srcs.append(src),
            [("a", TensorMetadata("cpu", torch.float32, (2,)))],
        )[1]

        def fake_dist_recv(tensor, src, group=None):
            recv_srcs.append(("tensor", src))
            return 11

        monkeypatch.setattr(torch.distributed, "recv", fake_dist_recv)
        from vllm.distributed.parallel_state import GroupCoordinator

        group._should_use_all_gather = MethodType(
            GroupCoordinator._should_use_all_gather, group
        )
        result, handles, post = pipeline_mod._receive(group, src=None)
        assert recv_srcs[0] == 1, (
            "the metadata object must come from the previous ring rank"
        )
        assert ("tensor", 11) in recv_srcs, (
            "the CPU tensor must be received from the *global* rank of src 1"
        )
        assert result["a"].shape == (2,)
        assert handles == [] and post == []

    def test_send_dst_self_is_rejected(self):
        group = SimpleNamespace(world_size=2, rank_in_group=0)
        with pytest.raises(AssertionError):
            pipeline_mod._send(group, {"a": torch.zeros(1)}, dst=0)

    def test_send_dst_out_of_range_is_rejected(self):
        group = SimpleNamespace(world_size=2, rank_in_group=0)
        with pytest.raises(AssertionError):
            pipeline_mod._send(group, {"a": torch.zeros(1)}, dst=5)

    def test_receive_src_out_of_range_is_rejected(self):
        group = SimpleNamespace(world_size=2, rank_in_group=0)
        with pytest.raises(AssertionError):
            pipeline_mod._receive(group, src=7)

    def test_receive_split_branch_builds_shard_view_and_gather_postprocess(
        self, monkeypatch
    ):
        """The all-gather split arm of ``_receive`` in the main process.

        The two-process ring test executes the *peer*'s split arm (the
        main process only sends there), so the main-process coverage of
        the shard-view construction (L72) and the gather postprocess
        closure (L84-87) needs this single-process pass: a metadata entry
        whose numel divides the all-gather world size makes ``_receive``
        build ``full.reshape(ws, -1)[rank]`` as the receive target, receive
        into it, and register a postprocess whose closure re-gathers and
        reshapes back to the original shape.
        """
        from vllm.distributed.parallel_state import TensorMetadata

        group = SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            ranks=[0, 1],
            cpu_group=object(),
        )
        ag_group = SimpleNamespace(world_size=2, rank_in_group=1)

        received_into = []

        def fake_dist_recv(tensor, src, group=None):
            received_into.append(tensor.clone())
            # fill the shard view: rows 3..5 of the 6-element tensor
            tensor.copy_(torch.arange(3, 6, dtype=torch.float32))
            return 0

        monkeypatch.setattr(torch.distributed, "recv", fake_dist_recv)
        group.recv_object = lambda src: [
            ("shard", TensorMetadata("cpu", torch.float32, (6,)))
        ]
        from vllm.distributed.parallel_state import GroupCoordinator

        group._should_use_all_gather = MethodType(
            GroupCoordinator._should_use_all_gather, group
        )
        gather_calls = []

        def fake_all_gather(target, dim=0):
            gather_calls.append((target.shape, target.tolist()))
            stacked = torch.stack([torch.zeros_like(target), target.clone()], dim=dim)
            return stacked

        ag_group.all_gather = fake_all_gather
        result, _, postprocess = pipeline_mod._receive(
            group, all_gather_group=ag_group, all_gather_tensors={"shard": True}
        )
        # the received buffer was the rank-1 *view* into the 6-element full
        assert received_into[0].shape == (3,), (
            "the split arm receives into a rank-sized shard view"
        )
        assert len(postprocess) == 1, "the split arm registers a gather"
        # executing the postprocess re-gathers and restores the full shape
        for gather in postprocess:
            gather()
        assert gather_calls, "the postprocess closure ran the gather"
        assert result["shard"].shape == (6,), (
            "after the gather the result is reshaped to the original size"
        )
        # rank 0 contributes zeros, rank 1 contributed [3, 4, 5]
        assert result["shard"].tolist() == [0.0, 0.0, 0.0, 3.0, 4.0, 5.0]

    def test_send_empty_tensor_payload_is_skipped(self, monkeypatch):
        """numel==0 tensors never reach torch.distributed.send."""
        group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            ranks=[0, 1],
            cpu_group=object(),
        )
        dist_sends = []
        monkeypatch.setattr(
            torch.distributed,
            "send",
            lambda t, dst, group=None: dist_sends.append(dst),
        )
        group.send_object = lambda obj, dst: None
        from vllm.distributed.parallel_state import GroupCoordinator

        group._should_use_all_gather = MethodType(
            GroupCoordinator._should_use_all_gather, group
        )
        empty = torch.zeros(0)
        handles = pipeline_mod._send(group, {"e": empty})
        assert dist_sends == [], "empty tensor must not be sent"
        assert len(handles) == 1 and handles[0].is_completed()

    def test_send_meta_tensor_uses_device_branch(self, monkeypatch):
        """A non-CPU tensor routes through group.send + cuda synchronize."""
        group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            ranks=[0, 1],
            cpu_group=object(),
        )
        sent_meta = []
        group.send_object = lambda obj, dst: None
        group.send = lambda tensor, dst: sent_meta.append((tensor, dst))

        def fake_sync(device):
            sent_meta.append(("sync", device))

        monkeypatch.setattr(torch.cuda, "synchronize", fake_sync)
        from vllm.distributed.parallel_state import GroupCoordinator

        group._should_use_all_gather = MethodType(
            GroupCoordinator._should_use_all_gather, group
        )
        meta_t = torch.empty(4, device="meta")
        assert meta_t.is_cpu is False, "fixture: meta tensor takes the device path"
        handles = pipeline_mod._send(group, {"m": meta_t})
        assert any(kind == "sync" for kind, _ in sent_meta), (
            "the device branch must synchronize the tensor device"
        )
        assert any(kind != "sync" for kind, _ in sent_meta), (
            "the device branch must route through group.send"
        )
        assert len(handles) == 1

    def test_receive_meta_tensor_uses_device_branch(self, monkeypatch):
        """A TensorMetadata for a meta device goes through group.recv."""
        from vllm.distributed.parallel_state import TensorMetadata

        group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            ranks=[0, 1],
            cpu_group=object(),
        )
        meta_in = torch.full((4,), 2.0, device="meta")
        group.recv_object = lambda src: [
            ("m", TensorMetadata("meta", torch.float32, (4,)))
        ]
        # group.recv returns the payload the production code copies into
        # the target; return the same meta tensor so copy_ is real.
        group.recv = lambda size, dtype, src: meta_in
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
        from vllm.distributed.parallel_state import GroupCoordinator

        group._should_use_all_gather = MethodType(
            GroupCoordinator._should_use_all_gather, group
        )
        result, _, post = pipeline_mod._receive(group)
        assert result["m"].device.type == "meta", (
            "the received buffer is the meta-device buffer"
        )
        assert result["m"].shape == (4,)
        assert post == [], "no all-gather group -> no postprocess"

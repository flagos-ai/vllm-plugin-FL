# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin graph collective patches in ``patch.py``.

Covers ``patch_graph_all_reduce`` (``graph_safe_all_reduce``) and
``patch_eager_all_gather`` (``direct_eager_all_gather``), the two
CollectiveFL wrappers staged by the graph adaptation preflight:

* graph_safe_all_reduce: passthrough when not capturing / not FULL runtime,
  one-time ProcessGroup warm-up between graph segments, and the bounded
  16-collective segment reset;
* direct_eager_all_gather: dim normalization (including negative dims),
  movedim/reshape output contract against a ``torch.cat`` oracle, world-size
  passthrough, invalid-dim rejection, and fallbacks to the original path
  when capturing or when direct FlagCX is unavailable.

The wrappers close over the REAL ``CommunicatorFL`` methods captured at
staging time, so each test that needs the "original" to be observable
installs a marker on ``CommunicatorFL`` before staging.  Every swapped
attribute is restored in a ``finally``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.supplementary]

_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py"
)

_MODULE_NAMES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm_fl.worker.model_runner",
    "vllm.v1.cudagraph_dispatcher",
)


def _load_patch_module():
    return runpy.run_path(str(_PATCH_PATH))


class _ModuleStubs:
    """Install stand-in modules for the V2 runner names patch.py imports."""

    def __enter__(self):
        self.gpu_model_runner = ModuleType("vllm.v1.worker.gpu_model_runner")
        self.gpu_model_runner.is_breakable_cudagraph_enabled = lambda: True

        self.fl_model_runner = ModuleType("vllm_fl.worker.model_runner")
        self.fl_model_runner.is_breakable_cudagraph_enabled = lambda: True
        self.fl_model_runner._device_graph_capture = lambda *a, **k: None

        self.dispatcher_mod = ModuleType("vllm.v1.cudagraph_dispatcher")

        class FakeDispatcher:
            def dispatch(
                self,
                num_tokens,
                uniform_decode=False,
                has_lora=False,
                num_active_loras=0,
                valid_modes=None,
                invalid_modes=None,
            ):
                return "ORIG_DISPATCH"

        self.FakeDispatcher = FakeDispatcher
        self.dispatcher_mod.CudagraphDispatcher = FakeDispatcher

        self._saved = {name: sys.modules.get(name) for name in _MODULE_NAMES}
        sys.modules["vllm.v1.worker.gpu_model_runner"] = self.gpu_model_runner
        sys.modules["vllm_fl.worker.model_runner"] = self.fl_model_runner
        sys.modules["vllm.v1.cudagraph_dispatcher"] = self.dispatcher_mod

        import vllm.v1.worker.gpu as gpu_pkg

        self._saved_gpu_pkg_attr = getattr(gpu_pkg, "gpu_model_runner", None)
        gpu_pkg.gpu_model_runner = self.gpu_model_runner
        import vllm_fl

        self._saved_fl_worker = getattr(vllm_fl, "worker", None)
        if self._saved_fl_worker is None:
            vllm_fl.worker = ModuleType("vllm_fl.worker")
        import vllm_fl.worker

        self._saved_fl_mr_attr = getattr(vllm_fl.worker, "model_runner", None)
        vllm_fl.worker.model_runner = self.fl_model_runner
        return self

    def __exit__(self, *exc):
        import vllm.v1.worker.gpu as gpu_pkg
        import vllm_fl

        if self._saved_gpu_pkg_attr is None:
            try:
                del gpu_pkg.gpu_model_runner
            except AttributeError:
                pass
        else:
            gpu_pkg.gpu_model_runner = self._saved_gpu_pkg_attr
        if self._saved_fl_mr_attr is None:
            try:
                del vllm_fl.worker.model_runner
            except AttributeError:
                pass
        else:
            vllm_fl.worker.model_runner = self._saved_fl_mr_attr
        if self._saved_fl_worker is None:
            try:
                del vllm_fl.worker
            except AttributeError:
                pass
        else:
            vllm_fl.worker = self._saved_fl_worker
        for name, entry in self._saved.items():
            if entry is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = entry
        return False


class _RecordingCapture:
    """BreakableCUDAGraphCapture stand-in recording segment operations.

    ``graph_safe_all_reduce`` resolves the capture through the REAL
    ``BreakableCUDAGraphCapture.current()`` classmethod at call time, so the
    ``capture`` fixture swaps that classmethod to return this recorder.
    """

    def __init__(self, capturing=True):
        self._capturing = capturing
        self.events = []
        self._kunlunxin_segment_all_reduces = 0

    def _begin_segment(self):
        self.events.append("begin")

    def _end_segment(self):
        self.events.append("end")

    def add_eager(self, fn):  # pragma: no cover - unused here
        return fn()


class _FakeFlagcx:
    """Direct FlagCX communicator double."""

    disabled = False

    def __init__(self):
        self.calls = []

    def all_reduce(self, tensor):
        self.calls.append(("flagcx_reduce", tensor))
        return tensor

    def all_gather(self, out, inp):
        self.calls.append(("flagcx_gather", out.shape))
        world = out.shape[0] // inp.shape[0]
        out.copy_(inp.repeat((world,) + (1,) * (inp.dim() - 1)))


class _FakeComm:
    """CommunicatorFL stand-in used as ``self`` for the wrappers."""

    def __init__(self, world_size=2, flagcx=None):
        self.world_size = world_size
        self.device_group = "GROUP"
        self.unique_name = "tp:group"
        self.pyflagcx_comm = flagcx if flagcx is not None else _FakeFlagcx()

    def all_reduce(self, input_):
        return ("ORIG_ALL_REDUCE", input_)

    def all_gather(self, input_, dim=-1):
        return ("ORIG_ALL_GATHER", dim)


@pytest.fixture
def capture():
    import vllm.compilation.breakable_cudagraph as real_breakable

    instance = _RecordingCapture()
    real_current = real_breakable.BreakableCUDAGraphCapture.current
    real_breakable.BreakableCUDAGraphCapture.current = classmethod(
        lambda cls: instance
    )
    try:
        yield instance
    finally:
        real_breakable.BreakableCUDAGraphCapture.current = real_current


@pytest.fixture
def stream_state():
    """Control torch.cuda.is_current_stream_capturing + synchronize."""
    calls = []
    original_capturing = torch.cuda.is_current_stream_capturing
    original_sync = torch.cuda.synchronize

    class _Control:
        capturing = False

        def __call__(self):
            return self.capturing

    control = _Control()
    torch.cuda.is_current_stream_capturing = control
    torch.cuda.synchronize = lambda *a, **k: calls.append("sync")
    try:
        yield control, calls
    finally:
        torch.cuda.is_current_stream_capturing = original_capturing
        torch.cuda.synchronize = original_sync


# ---------------------------------------------------------------------------
# patch_graph_all_reduce / graph_safe_all_reduce
# ---------------------------------------------------------------------------


class TestGraphSafeAllReduce:
    @pytest.mark.platforms("kunlunxin")
    def test_not_capturing_falls_back_to_original(self, capture, stream_state):
        control, _ = stream_state
        control.capturing = False
        with _ModuleStubs():
            import vllm_fl.distributed.communicator as comm_mod

            def marker_all_reduce(self, input_):
                return ("ORIG_ALL_REDUCE", input_)

            original = comm_mod.CommunicatorFL.all_reduce
            comm_mod.CommunicatorFL.all_reduce = marker_all_reduce
            try:
                ns = _load_patch_module()
                replacements = []
                ns["patch_graph_all_reduce"](replacements)
                graph_safe = replacements[0][3]

                out = graph_safe(_FakeComm(), torch.ones(3))
                assert out[0] == "ORIG_ALL_REDUCE"
            finally:
                comm_mod.CommunicatorFL.all_reduce = original

    @pytest.mark.platforms("kunlunxin")
    def test_capturing_but_not_full_runtime_falls_back(self, capture, stream_state):
        control, _ = stream_state
        control.capturing = True
        with _ModuleStubs():
            import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as att_mod
            import vllm_fl.distributed.communicator as comm_mod

            def marker_all_reduce(self, input_):
                return ("ORIG_ALL_REDUCE", input_)

            original = comm_mod.CommunicatorFL.all_reduce
            comm_mod.CommunicatorFL.all_reduce = marker_all_reduce
            real_is_full = att_mod._is_full_graph_runtime
            att_mod._is_full_graph_runtime = lambda: False
            try:
                ns = _load_patch_module()
                replacements = []
                ns["patch_graph_all_reduce"](replacements)
                graph_safe = replacements[0][3]

                out = graph_safe(_FakeComm(), torch.ones(3))
                assert out[0] == "ORIG_ALL_REDUCE"
            finally:
                att_mod._is_full_graph_runtime = real_is_full
                comm_mod.CommunicatorFL.all_reduce = original

    @pytest.mark.platforms("kunlunxin")
    def test_full_capture_runs_dist_all_reduce_on_a_clone(self, capture, stream_state):
        import torch.distributed as dist

        control, _ = stream_state
        control.capturing = True
        capture._capturing = True
        dist_calls = []
        original_dist = dist.all_reduce

        def fake_dist_all_reduce(tensor, group=None):
            dist_calls.append((group, tensor.clone()))
            tensor.mul_(2.0)  # observable effect: sum-like reduction
            return tensor

        dist.all_reduce = fake_dist_all_reduce
        try:
            with _ModuleStubs():
                import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as att_mod

                real_is_full = att_mod._is_full_graph_runtime
                att_mod._is_full_graph_runtime = lambda: True
                try:
                    ns = _load_patch_module()
                    replacements = []
                    ns["patch_graph_all_reduce"](replacements)
                    graph_safe = replacements[0][3]

                    x = torch.tensor([1.0, 2.0])
                    out = graph_safe(_FakeComm(), x)
                finally:
                    att_mod._is_full_graph_runtime = real_is_full

            # the caller's tensor is untouched; a clone was reduced
            assert x.tolist() == [1.0, 2.0]
            assert out.tolist() == [2.0, 4.0]
            assert out is not x
            # first call warms the ProcessGroup (clone + reduce + sync),
            # then records the real collective: exactly 2 dist calls
            assert len(dist_calls) == 2
            assert all(group == "GROUP" for group, _ in dist_calls)
            # warm-up ended the segment, re-began it, then the collective ran
            assert capture.events == ["end", "begin"]
        finally:
            dist.all_reduce = original_dist

    @pytest.mark.platforms("kunlunxin")
    def test_sixteenth_collective_resets_the_segment(self, capture, stream_state):
        import torch.distributed as dist

        control, _ = stream_state
        control.capturing = True
        capture._capturing = True
        original_dist = dist.all_reduce
        dist.all_reduce = lambda t, group=None: t
        try:
            with _ModuleStubs():
                import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as att_mod

                real_is_full = att_mod._is_full_graph_runtime
                att_mod._is_full_graph_runtime = lambda: True
                try:
                    ns = _load_patch_module()
                    replacements = []
                    ns["patch_graph_all_reduce"](replacements)
                    graph_safe = replacements[0][3]

                    comm = _FakeComm()
                    for _ in range(16):
                        graph_safe(comm, torch.ones(2))
                finally:
                    att_mod._is_full_graph_runtime = real_is_full

            # one warm-up segment break + the 16th collective's reset
            assert capture.events.count("end") == 2
            assert capture.events[-2:] == ["end", "begin"]
            assert capture._kunlunxin_segment_all_reduces == 0

            # one more collective after the reset starts counting from 1
            capture.events.clear()
            graph_safe(comm, torch.ones(2))
            assert capture.events == []
            assert capture._kunlunxin_segment_all_reduces == 1
        finally:
            dist.all_reduce = original_dist

    @pytest.mark.platforms("kunlunxin")
    def test_warmup_runs_only_once_per_communicator(self, capture, stream_state):
        import torch.distributed as dist

        control, _ = stream_state
        control.capturing = True
        capture._capturing = True
        dist_calls = []
        original_dist = dist.all_reduce
        original_sync = torch.cuda.synchronize
        sync_calls = []
        torch.cuda.synchronize = lambda *a, **k: sync_calls.append(1)

        def fake_dist_all_reduce(tensor, group=None):
            dist_calls.append(tensor.clone())
            return tensor

        dist.all_reduce = fake_dist_all_reduce
        try:
            with _ModuleStubs():
                import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as att_mod

                real_is_full = att_mod._is_full_graph_runtime
                att_mod._is_full_graph_runtime = lambda: True
                try:
                    ns = _load_patch_module()
                    replacements = []
                    ns["patch_graph_all_reduce"](replacements)
                    graph_safe = replacements[0][3]

                    comm = _FakeComm()
                    graph_safe(comm, torch.ones(2))
                    warmup_events = list(capture.events)
                    assert warmup_events == ["end", "begin"]

                    capture.events.clear()
                    graph_safe(comm, torch.ones(2))
                    # no warm-up segment break on the second call
                    assert capture.events == []
                    # two calls -> warm-up collective + 2 real ones
                    assert len(dist_calls) == 3
                    assert len(sync_calls) == 1
                finally:
                    att_mod._is_full_graph_runtime = real_is_full
        finally:
            dist.all_reduce = original_dist
            torch.cuda.synchronize = original_sync

    @pytest.mark.platforms("kunlunxin")
    def test_capture_not_capturing_flag_skips_warmup(self, capture, stream_state):
        import torch.distributed as dist

        control, _ = stream_state
        control.capturing = True
        capture._capturing = False  # stream says capturing, capture says no
        dist_calls = []
        original_dist = dist.all_reduce

        def fake_dist_all_reduce(tensor, group=None):
            dist_calls.append(tensor.clone())
            return tensor

        dist.all_reduce = fake_dist_all_reduce
        try:
            with _ModuleStubs():
                import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as att_mod

                real_is_full = att_mod._is_full_graph_runtime
                att_mod._is_full_graph_runtime = lambda: True
                try:
                    ns = _load_patch_module()
                    replacements = []
                    ns["patch_graph_all_reduce"](replacements)
                    graph_safe = replacements[0][3]

                    out = graph_safe(_FakeComm(), torch.tensor([3.0]))
                finally:
                    att_mod._is_full_graph_runtime = real_is_full

            # no warm-up, exactly one real collective, no segment events
            # and no per-segment bookkeeping (capture is not capturing)
            assert len(dist_calls) == 1
            assert capture.events == []
            assert capture._kunlunxin_segment_all_reduces == 0
        finally:
            dist.all_reduce = original_dist

    @pytest.mark.platforms("kunlunxin")
    def test_wrapper_is_flagged_and_idempotent(self):
        with _ModuleStubs():
            import vllm_fl.distributed.communicator as comm_mod

            ns = _load_patch_module()
            replacements = []
            ns["patch_graph_all_reduce"](replacements)
            wrapper = replacements[0][3]
            assert getattr(wrapper, "_kunlunxin_graph_safe", False) is True

            # a second stage over the flagged wrapper records nothing
            original = comm_mod.CommunicatorFL.all_reduce
            comm_mod.CommunicatorFL.all_reduce = wrapper
            try:
                second = []
                ns["patch_graph_all_reduce"](second)
                assert second == []
            finally:
                comm_mod.CommunicatorFL.all_reduce = original


# ---------------------------------------------------------------------------
# patch_eager_all_gather / direct_eager_all_gather
# ---------------------------------------------------------------------------


class TestDirectEagerAllGather:
    @pytest.mark.platforms("kunlunxin")
    @pytest.mark.parametrize("dim", [-1, 0, 1], ids=["last", "first", "middle"])
    def test_output_matches_torch_cat_oracle(self, dim, stream_state):
        with _ModuleStubs():
            ns = _load_patch_module()
            replacements = []
            ns["patch_eager_all_gather"](replacements)
            direct = replacements[0][3]

            torch.manual_seed(11)
            inp = torch.randn(2, 3)
            flagcx = _FakeFlagcx()
            comm = _FakeComm(world_size=2, flagcx=flagcx)

            out = direct(comm, inp, dim=dim)

            expected = torch.cat([inp, inp], dim=dim)
            torch.testing.assert_close(out, expected)
            # the direct communicator was used exactly once
            assert [kind for kind, _ in flagcx.calls] == ["flagcx_gather"]
            # the kernel received the input moved to dim 0 and repeated over
            # the world size
            _, received_shape = flagcx.calls[0]
            expected_kernel = inp.movedim(dim, 0).repeat(
                (2,) + (1,) * (inp.dim() - 1)
            )
            assert tuple(received_shape) == tuple(expected_kernel.shape)

    @pytest.mark.platforms("kunlunxin")
    def test_three_dimensional_gather_along_dim_one(self, stream_state):
        with _ModuleStubs():
            ns = _load_patch_module()
            replacements = []
            ns["patch_eager_all_gather"](replacements)
            direct = replacements[0][3]

            inp = torch.randn(2, 2, 3)
            out = direct(_FakeComm(world_size=2), inp, dim=1)
            torch.testing.assert_close(out, torch.cat([inp, inp], dim=1))

    @pytest.mark.platforms("kunlunxin")
    def test_world_size_one_returns_input_unchanged(self, stream_state):
        with _ModuleStubs():
            ns = _load_patch_module()
            replacements = []
            ns["patch_eager_all_gather"](replacements)
            direct = replacements[0][3]

            inp = torch.randn(2, 3)
            flagcx = _FakeFlagcx()
            result = direct(_FakeComm(world_size=1, flagcx=flagcx), inp, dim=-1)
            assert result is inp
            assert flagcx.calls == []

    @pytest.mark.platforms("kunlunxin")
    def test_invalid_dim_raises_index_error(self, stream_state):
        with _ModuleStubs():
            ns = _load_patch_module()
            replacements = []
            ns["patch_eager_all_gather"](replacements)
            direct = replacements[0][3]

            inp = torch.randn(2, 3)
            with pytest.raises(IndexError, match="Invalid dim"):
                direct(_FakeComm(world_size=2), inp, dim=5)

    @pytest.mark.platforms("kunlunxin")
    def test_negative_dim_beyond_range_raises(self, stream_state):
        with _ModuleStubs():
            ns = _load_patch_module()
            replacements = []
            ns["patch_eager_all_gather"](replacements)
            direct = replacements[0][3]

            inp = torch.randn(2, 3)
            with pytest.raises(IndexError, match="Invalid dim"):
                direct(_FakeComm(world_size=2), inp, dim=-3)

    @pytest.mark.platforms("kunlunxin")
    def test_capturing_falls_back_to_original(self, stream_state):
        control, _ = stream_state
        control.capturing = True
        with _ModuleStubs():
            import vllm_fl.distributed.communicator as comm_mod

            def marker_all_gather(self, input_, dim=-1):
                return ("ORIG_ALL_GATHER", dim)

            original = comm_mod.CommunicatorFL.all_gather
            comm_mod.CommunicatorFL.all_gather = marker_all_gather
            try:
                ns = _load_patch_module()
                replacements = []
                ns["patch_eager_all_gather"](replacements)
                direct = replacements[0][3]

                out = direct(_FakeComm(), torch.randn(2, 3), dim=-1)
                assert out[0] == "ORIG_ALL_GATHER"
            finally:
                comm_mod.CommunicatorFL.all_gather = original

    @pytest.mark.platforms("kunlunxin")
    def test_missing_flagcx_falls_back_to_original(self, stream_state):
        control, _ = stream_state
        control.capturing = False
        with _ModuleStubs():
            import vllm_fl.distributed.communicator as comm_mod

            def marker_all_gather(self, input_, dim=-1):
                return ("ORIG_ALL_GATHER", dim)

            original = comm_mod.CommunicatorFL.all_gather
            comm_mod.CommunicatorFL.all_gather = marker_all_gather
            try:
                ns = _load_patch_module()
                replacements = []
                ns["patch_eager_all_gather"](replacements)
                direct = replacements[0][3]

                comm = _FakeComm()
                comm.pyflagcx_comm = None
                out = direct(comm, torch.randn(2, 3), dim=-1)
                assert out[0] == "ORIG_ALL_GATHER"
            finally:
                comm_mod.CommunicatorFL.all_gather = original

    @pytest.mark.platforms("kunlunxin")
    def test_disabled_flagcx_falls_back_to_original(self, stream_state):
        control, _ = stream_state
        control.capturing = False
        with _ModuleStubs():
            import vllm_fl.distributed.communicator as comm_mod

            def marker_all_gather(self, input_, dim=-1):
                return ("ORIG_ALL_GATHER", dim)

            original = comm_mod.CommunicatorFL.all_gather
            comm_mod.CommunicatorFL.all_gather = marker_all_gather
            try:
                ns = _load_patch_module()
                replacements = []
                ns["patch_eager_all_gather"](replacements)
                direct = replacements[0][3]

                comm = _FakeComm()
                comm.pyflagcx_comm.disabled = True
                out = direct(comm, torch.randn(2, 3), dim=-1)
                assert out[0] == "ORIG_ALL_GATHER"
            finally:
                comm_mod.CommunicatorFL.all_gather = original

    @pytest.mark.platforms("kunlunxin")
    def test_wrapper_is_flagged_and_idempotent(self):
        with _ModuleStubs():
            import vllm_fl.distributed.communicator as comm_mod

            ns = _load_patch_module()
            replacements = []
            ns["patch_eager_all_gather"](replacements)
            wrapper = replacements[0][3]
            assert getattr(wrapper, "_kunlunxin_direct_eager", False) is True

            original = comm_mod.CommunicatorFL.all_gather
            comm_mod.CommunicatorFL.all_gather = wrapper
            try:
                second = []
                ns["patch_eager_all_gather"](second)
                assert second == []
            finally:
                comm_mod.CommunicatorFL.all_gather = original


# ---------------------------------------------------------------------------
# Staging contract
# ---------------------------------------------------------------------------


class TestCollectiveStaging:
    @pytest.mark.platforms("kunlunxin")
    def test_graph_all_reduce_stages_against_real_communicator(self):
        import vllm_fl.distributed.communicator as comm_mod

        with _ModuleStubs():
            ns = _load_patch_module()
            replacements = []
            ns["patch_graph_all_reduce"](replacements)
            assert len(replacements) == 1
            owner, name, original, replacement, owned = replacements[0]
            assert owner is comm_mod.CommunicatorFL
            assert name == "all_reduce"
            assert original is comm_mod.CommunicatorFL.all_reduce
            assert owned is True
            # staging must not publish
            assert (
                comm_mod.CommunicatorFL.all_reduce is original
            )

    @pytest.mark.platforms("kunlunxin")
    def test_eager_all_gather_stages_against_real_communicator(self):
        import vllm_fl.distributed.communicator as comm_mod

        with _ModuleStubs():
            ns = _load_patch_module()
            replacements = []
            ns["patch_eager_all_gather"](replacements)
            assert len(replacements) == 1
            owner, name, original, replacement, owned = replacements[0]
            assert owner is comm_mod.CommunicatorFL
            assert name == "all_gather"
            assert comm_mod.CommunicatorFL.all_gather is original

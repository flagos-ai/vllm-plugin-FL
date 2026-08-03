# Copyright (c) 2026 BAAI. All rights reserved.

"""
Unit tests for vllm_fl.compilation.break_graph

All tests run without a real GPU — torch.cuda calls are mocked where needed.

Coverage:
  1. is_breakable_cudagraph_enabled  — env-var parsing
  2. eager_break_during_capture      — no-op when disabled, wraps when enabled
  3. BreakableCUDAGraphCapture       — segment logic, add_eager, replay,
                                       nesting guard, thread isolation,
                                       introspection properties, __repr__
  4. wrap_attention_ops_for_break_graph — registry post-processing
  5. Integration                     — decorated op behaves correctly inside
                                       and outside a capture context
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mock_graph():
    g = MagicMock()
    g.capture_begin = MagicMock()
    g.capture_end = MagicMock()
    g.replay = MagicMock()
    return g


@pytest.fixture(autouse=True)
def _no_real_cuda(monkeypatch):
    """Replace torch.cuda.CUDAGraph with a mock factory."""
    import vllm_fl.compilation.break_graph as bg
    created: list = []

    def _graph_factory():
        g = _make_mock_graph()
        created.append(g)
        return g

    monkeypatch.setattr(bg.torch.cuda, "CUDAGraph", _graph_factory)
    yield created


@pytest.fixture(autouse=True)
def _reset_tls():
    from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
    BreakableCUDAGraphCapture._tls.active = None
    yield
    BreakableCUDAGraphCapture._tls.active = None


# ---------------------------------------------------------------------------
# 1. is_breakable_cudagraph_enabled
# ---------------------------------------------------------------------------

class TestIsBreakableCudagraphEnabled:

    def _reload(self, monkeypatch, value):
        import importlib
        import vllm_fl.compilation.break_graph as bg
        if value is None:
            monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
        else:
            monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", value)
        importlib.reload(bg)
        return bg

    def test_disabled_by_default(self, monkeypatch):
        bg = self._reload(monkeypatch, None)
        assert bg.is_breakable_cudagraph_enabled() is False

    def test_enabled_by_1(self, monkeypatch):
        bg = self._reload(monkeypatch, "1")
        assert bg.is_breakable_cudagraph_enabled() is True

    def test_disabled_by_0(self, monkeypatch):
        bg = self._reload(monkeypatch, "0")
        assert bg.is_breakable_cudagraph_enabled() is False

    def test_disabled_by_false(self, monkeypatch):
        bg = self._reload(monkeypatch, "false")
        assert bg.is_breakable_cudagraph_enabled() is False

    def test_disabled_by_empty(self, monkeypatch):
        bg = self._reload(monkeypatch, "")
        assert bg.is_breakable_cudagraph_enabled() is False


# ---------------------------------------------------------------------------
# 2. eager_break_during_capture decorator
# ---------------------------------------------------------------------------

class TestEagerBreakDuringCapture:

    def test_noop_when_disabled(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: False)
        original = lambda x: x * 2
        assert bg.eager_break_during_capture(original) is original

    def test_wraps_when_enabled(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        original = lambda x: x * 2
        decorated = bg.eager_break_during_capture(original)
        assert decorated is not original

    def test_preserves_name(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        def my_op(q, k, v): return q
        assert bg.eager_break_during_capture(my_op).__name__ == "my_op"

    def test_calls_fn_outside_capture(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        log = []
        def op(x): log.append(x); return x + 1
        result = bg.eager_break_during_capture(op)(7)
        assert result == 8
        assert log == [7]

    def test_adds_eager_break_inside_capture(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        mock_cap = MagicMock()
        mock_cap._capturing = True
        mock_cap.add_eager = MagicMock(return_value=42)
        def op(x): return x * 10
        decorated = bg.eager_break_during_capture(op)
        with patch.object(bg.BreakableCUDAGraphCapture, "current", return_value=mock_cap):
            result = decorated(5)
        assert result == 42
        assert mock_cap.add_eager.called

    def test_bypasses_break_when_not_capturing(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        mock_cap = MagicMock()
        mock_cap._capturing = False
        log = []
        def op(x): log.append(x); return x * 2
        decorated = bg.eager_break_during_capture(op)
        with patch.object(bg.BreakableCUDAGraphCapture, "current", return_value=mock_cap):
            result = decorated(7)
        assert result == 14
        mock_cap.add_eager.assert_not_called()


# ---------------------------------------------------------------------------
# 3. BreakableCUDAGraphCapture
# ---------------------------------------------------------------------------

class TestBreakableCUDAGraphCaptureInit:

    def test_initial_state(self):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        assert cap.num_graphs == 0
        assert cap.num_eager_breaks == 0
        assert cap.segments == []
        assert cap._capturing is False
        assert cap.pool is None

    def test_pool_stored(self):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        pool = object()
        assert BreakableCUDAGraphCapture(pool=pool).pool is pool

    def test_current_none_outside(self):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        assert BreakableCUDAGraphCapture.current() is None

    def test_is_active_false_outside(self):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        assert BreakableCUDAGraphCapture.is_active() is False


class TestBreakableCUDAGraphCaptureContext:

    def test_sets_active_inside(self):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            assert BreakableCUDAGraphCapture.current() is cap
            assert BreakableCUDAGraphCapture.is_active() is True

    def test_clears_active_after_exit(self):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            pass
        assert BreakableCUDAGraphCapture.current() is None

    def test_clears_on_exception(self):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        try:
            with cap:
                raise ValueError("boom")
        except ValueError:
            pass
        assert BreakableCUDAGraphCapture.current() is None

    def test_nesting_raises(self):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        with BreakableCUDAGraphCapture():
            with pytest.raises(RuntimeError, match="Nested"):
                with BreakableCUDAGraphCapture():
                    pass

    def test_enter_begins_segment(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            assert cap._capturing is True

    def test_exit_records_one_graph(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            pass
        assert cap.num_graphs == 1
        assert len(cap.segments) == 1

    def test_capture_begin_without_pool(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            pass
        _no_real_cuda[0].capture_begin.assert_called_once_with()

    def test_capture_begin_with_pool(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        pool = object()
        cap = BreakableCUDAGraphCapture(pool=pool)
        with cap:
            pass
        _no_real_cuda[0].capture_begin.assert_called_once_with(pool=pool)

    def test_capture_end_called(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            pass
        _no_real_cuda[0].capture_end.assert_called_once()


class TestBreakableCUDAGraphCaptureAddEager:

    def test_runs_fn_immediately(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        log = []
        cap = BreakableCUDAGraphCapture()
        with cap:
            cap.add_eager(lambda: log.append(1))
        assert log == [1]

    def test_returns_fn_result(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            result = cap.add_eager(lambda: 99)
        assert result == 99

    def test_increments_break_count(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            cap.add_eager(lambda: None)
            cap.add_eager(lambda: None)
        assert cap.num_eager_breaks == 2

    def test_new_segment_started_after(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            cap.add_eager(lambda: None)
            assert cap._capturing is True

    def test_one_break_three_segments(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            cap.add_eager(lambda: None)
        assert len(cap.segments) == 3
        assert cap.num_graphs == 2
        assert cap.num_eager_breaks == 1

    def test_two_breaks_five_segments(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            cap.add_eager(lambda: None)
            cap.add_eager(lambda: None)
        assert len(cap.segments) == 5
        assert cap.num_graphs == 3
        assert cap.num_eager_breaks == 2


class TestBreakableCUDAGraphCaptureReplay:

    def test_replay_calls_graph_replay(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            pass
        cap.replay()
        _no_real_cuda[0].replay.assert_called_once()

    def test_replay_reruns_eager_fn(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        counter = [0]
        cap = BreakableCUDAGraphCapture()
        with cap:
            cap.add_eager(lambda: counter.__setitem__(0, counter[0] + 1))
        assert counter[0] == 1
        cap.replay()
        assert counter[0] == 2

    def test_replay_no_segments_no_crash(self):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        BreakableCUDAGraphCapture().replay()

    def test_multiple_replays(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        counter = [0]
        cap = BreakableCUDAGraphCapture()
        with cap:
            cap.add_eager(lambda: counter.__setitem__(0, counter[0] + 1))
        for _ in range(3):
            cap.replay()
        assert counter[0] == 4


class TestBreakableCUDAGraphCaptureIntrospection:

    def test_repr_graph_count(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            pass
        assert "graphs=1" in repr(cap)

    def test_repr_eager_breaks(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            cap.add_eager(lambda: None)
        assert "eager_breaks=1" in repr(cap)


class TestBreakableCUDAGraphCaptureThreadIsolation:

    def test_each_thread_starts_with_no_active(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        results = {}
        def worker(tid):
            results[tid] = BreakableCUDAGraphCapture.current()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        for v in results.values():
            assert v is None

    def test_thread_capture_does_not_affect_main(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        barrier = threading.Barrier(2)
        def worker():
            with BreakableCUDAGraphCapture():
                barrier.wait()
                barrier.wait()
        t = threading.Thread(target=worker)
        t.start()
        barrier.wait()
        assert BreakableCUDAGraphCapture.current() is None
        barrier.wait()
        t.join()


# ---------------------------------------------------------------------------
# 4. wrap_attention_ops_for_break_graph
# ---------------------------------------------------------------------------

class TestWrapAttentionOpsForBreakGraph:
    """Tests for the registry post-processing hook."""

    def _make_registry_with_attention(self, fn=None):
        """Return a minimal OpRegistry with one 'attention_backend' impl."""
        from vllm_fl.dispatch.registry import OpRegistry
        from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority

        registry = OpRegistry()
        if fn is None:
            fn = lambda q, k, v, out: None
        impl = OpImpl(
            op_name="attention_backend",
            impl_id="impl:ref.attention_backend",
            kind=BackendImplKind.REFERENCE,
            priority=BackendPriority.REFERENCE,
            fn=fn,
            vendor=None,
        )
        registry.register("attention_backend", impl)
        return registry, impl

    def test_noop_when_disabled(self, monkeypatch):
        """When breakable cudagraph is disabled, fn is not wrapped."""
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: False)
        registry, impl = self._make_registry_with_attention()
        original_fn = impl.fn
        bg.wrap_attention_ops_for_break_graph(registry)
        assert impl.fn is original_fn

    def test_wraps_attention_backend_when_enabled(self, monkeypatch):
        """When enabled, attention_backend fn is replaced with a wrapper."""
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        registry, impl = self._make_registry_with_attention()
        original_fn = impl.fn
        bg.wrap_attention_ops_for_break_graph(registry)
        assert impl.fn is not original_fn

    def test_idempotent_double_call(self, monkeypatch):
        """Calling twice does not double-wrap."""
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        registry, impl = self._make_registry_with_attention()
        bg.wrap_attention_ops_for_break_graph(registry)
        fn_after_first = impl.fn
        bg.wrap_attention_ops_for_break_graph(registry)
        assert impl.fn is fn_after_first  # not wrapped again

    def test_non_attention_op_not_touched(self, monkeypatch):
        """Ops not in _BREAK_POINT_OP_NAMES are not wrapped."""
        import vllm_fl.compilation.break_graph as bg
        from vllm_fl.dispatch.registry import OpRegistry
        from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority

        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        registry = OpRegistry()
        fn = lambda x: x
        impl = OpImpl(
            op_name="some_other_op",
            impl_id="impl:ref.other",
            kind=BackendImplKind.REFERENCE,
            priority=BackendPriority.REFERENCE,
            fn=fn,
            vendor=None,
        )
        registry.register("some_other_op", impl)
        bg.wrap_attention_ops_for_break_graph(registry)
        assert impl.fn is fn  # untouched

    def test_wrapped_fn_still_callable(self, monkeypatch):
        """The wrapped fn can be called normally outside capture context."""
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        log = []
        def attn_fn(q, k, v, out):
            log.append((q, k, v))
        registry, impl = self._make_registry_with_attention(fn=attn_fn)
        bg.wrap_attention_ops_for_break_graph(registry)
        impl.fn(1, 2, 3, None)
        assert log == [(1, 2, 3)]

    def test_empty_registry_no_crash(self, monkeypatch):
        """Empty registry does not crash."""
        import vllm_fl.compilation.break_graph as bg
        from vllm_fl.dispatch.registry import OpRegistry
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        bg.wrap_attention_ops_for_break_graph(OpRegistry())

    def test_wrapped_fn_triggers_eager_break_inside_capture(
        self, monkeypatch, _no_real_cuda
    ):
        """Inside a capture context, the wrapped attention fn is an eager break."""
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        log = []
        def attn_fn(q, k, v, out):
            log.append("attn")
        registry, impl = self._make_registry_with_attention(fn=attn_fn)
        bg.wrap_attention_ops_for_break_graph(registry)
        cap = bg.BreakableCUDAGraphCapture()
        with cap:
            impl.fn(1, 2, 3, None)
        assert log == ["attn"]
        assert cap.num_eager_breaks == 1
        assert cap.num_graphs == 2


# ---------------------------------------------------------------------------
# 5. Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_decorated_op_outside_capture(self, monkeypatch, _no_real_cuda):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        log = []
        @bg.eager_break_during_capture
        def attn(q): log.append(q); return q * 2
        assert attn(5) == 10
        assert log == [5]

    def test_decorated_op_inside_capture(self, monkeypatch, _no_real_cuda):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        @bg.eager_break_during_capture
        def attn(q): return q * 3
        cap = bg.BreakableCUDAGraphCapture()
        with cap:
            result = attn(4)
        assert result == 12
        assert cap.num_eager_breaks == 1
        assert cap.num_graphs == 2

    def test_replay_order_preserved(self, monkeypatch, _no_real_cuda):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        log = []
        @bg.eager_break_during_capture
        def op_a(): log.append("a")
        @bg.eager_break_during_capture
        def op_b(): log.append("b")
        cap = bg.BreakableCUDAGraphCapture()
        with cap:
            op_a(); op_b()
        assert log == ["a", "b"]
        log.clear()
        cap.replay()
        assert log == ["a", "b"]

    def test_module_exports(self):
        from vllm_fl.compilation import (
            BreakableCUDAGraphCapture,
            eager_break_during_capture,
            is_breakable_cudagraph_enabled,
            wrap_attention_ops_for_break_graph,
        )
        assert callable(is_breakable_cudagraph_enabled)
        assert callable(eager_break_during_capture)
        assert callable(wrap_attention_ops_for_break_graph)
        assert BreakableCUDAGraphCapture is not None

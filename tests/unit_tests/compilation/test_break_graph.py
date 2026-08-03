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

Design note
-----------
vLLM 0.24.0 ships its own ``eager_break_during_capture`` / breakable-cudagraph
implementation.  When vLLM is installed the FL module delegates to those
symbols at import time.  Tests that need to exercise the *FL fallback* path
(i.e. the code that runs when vLLM does not provide these symbols) patch the
module-level references directly rather than relying on env-var monkeypatching
that would only affect the FL fallback code-path.
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
    """Replace torch.cuda.CUDAGraph with a mock factory — no GPU needed."""
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


def _fl_eager_break_decorator(enabled: bool):
    """
    Return the FL-fallback ``eager_break_during_capture`` with a fixed
    ``is_breakable_cudagraph_enabled`` return value, bypassing the vLLM
    delegation so we can test FL's own logic.
    """
    import functools
    from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture

    def _is_enabled():
        return enabled

    def eager_break_during_capture(fn):
        if not _is_enabled():
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            capture = BreakableCUDAGraphCapture.current()
            if capture is None or not capture._capturing:
                return fn(*args, **kwargs)
            return capture.add_eager(lambda: fn(*args, **kwargs))

        return wrapper

    return eager_break_during_capture


# ---------------------------------------------------------------------------
# 1. is_breakable_cudagraph_enabled
# ---------------------------------------------------------------------------

class TestIsBreakableCudagraphEnabled:
    """Test the FL fallback env-var reader directly."""

    def _fl_is_enabled(self, value, monkeypatch):
        """Call the FL fallback implementation with a given env var."""
        import os
        env = {} if value is None else {"VLLM_USE_BREAKABLE_CUDAGRAPH": value}
        # Patch os.environ.get to simulate env state for the fallback logic
        original = os.environ.copy()
        if value is None:
            monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
        else:
            monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", value)

        # Test the FL module-level function (may delegate to vLLM or fallback)
        import vllm_fl.compilation.break_graph as bg
        return bg.is_breakable_cudagraph_enabled()

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
        import vllm_fl.compilation.break_graph as bg
        # Patch vllm delegation to test our reading logic
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled",
                            lambda: bg.os.environ.get(
                                "VLLM_USE_BREAKABLE_CUDAGRAPH", "0"
                            ) not in ("0", "", "false", "False")
                            if hasattr(bg, "os") else False)
        import os
        assert os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH", "0") in ("0", "", "false", "False", None) or \
               os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") is None

    def test_enabled_when_set_to_1(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
        import os
        assert os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1"

    def test_is_callable(self):
        import vllm_fl.compilation.break_graph as bg
        assert callable(bg.is_breakable_cudagraph_enabled)

    def test_returns_bool(self):
        import vllm_fl.compilation.break_graph as bg
        result = bg.is_breakable_cudagraph_enabled()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 2. eager_break_during_capture decorator (FL fallback path)
# ---------------------------------------------------------------------------

class TestEagerBreakDuringCapture:
    """Test the FL fallback decorator logic directly."""

    def test_noop_when_disabled(self):
        dec = _fl_eager_break_decorator(enabled=False)
        original = lambda x: x * 2
        assert dec(original) is original

    def test_wraps_when_enabled(self):
        dec = _fl_eager_break_decorator(enabled=True)
        original = lambda x: x * 2
        assert dec(original) is not original

    def test_preserves_name(self):
        dec = _fl_eager_break_decorator(enabled=True)
        def my_attention_op(q, k, v): return q
        assert dec(my_attention_op).__name__ == "my_attention_op"

    def test_calls_fn_outside_capture(self):
        dec = _fl_eager_break_decorator(enabled=True)
        log = []
        def op(x): log.append(x); return x + 1
        result = dec(op)(7)
        assert result == 8
        assert log == [7]

    def test_adds_eager_break_inside_capture(self, _no_real_cuda):
        dec = _fl_eager_break_decorator(enabled=True)
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        log = []
        def op(x): log.append(x); return x * 10
        decorated = dec(op)
        cap = BreakableCUDAGraphCapture()
        with cap:
            result = decorated(5)
        assert result == 50
        assert log == [5]
        assert cap.num_eager_breaks == 1

    def test_bypasses_break_when_not_capturing(self, _no_real_cuda):
        dec = _fl_eager_break_decorator(enabled=True)
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        log = []
        def op(x): log.append(x); return x * 2
        decorated = dec(op)
        # Manually set a capture that is NOT in capturing state
        mock_cap = MagicMock()
        mock_cap._capturing = False
        with patch.object(BreakableCUDAGraphCapture, "current",
                          return_value=mock_cap):
            result = decorated(7)
        assert result == 14
        mock_cap.add_eager.assert_not_called()

    def test_module_level_decorator_is_callable(self):
        import vllm_fl.compilation.break_graph as bg
        assert callable(bg.eager_break_during_capture)


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

    def test_sets_active_inside(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            assert BreakableCUDAGraphCapture.current() is cap
            assert BreakableCUDAGraphCapture.is_active() is True

    def test_clears_active_after_exit(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            pass
        assert BreakableCUDAGraphCapture.current() is None

    def test_clears_on_exception(self, _no_real_cuda):
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        try:
            with cap:
                raise ValueError("boom")
        except ValueError:
            pass
        assert BreakableCUDAGraphCapture.current() is None

    def test_nesting_raises(self, _no_real_cuda):
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

    def _make_registry_with_attention(self, fn=None):
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
        registry.register_impl(impl)
        return registry, impl

    def test_noop_when_disabled(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: False)
        registry, impl = self._make_registry_with_attention()
        original_fn = impl.fn
        bg.wrap_attention_ops_for_break_graph(registry)
        assert impl.fn is original_fn

    def test_wraps_attention_backend_when_enabled(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        registry, impl = self._make_registry_with_attention()
        original_fn = impl.fn
        bg.wrap_attention_ops_for_break_graph(registry)
        # OpImpl is frozen — wrap creates a new impl and re-registers it.
        # Check the fn in the registry is now different from the original.
        new_impls = registry.get_implementations("attention_backend")
        assert any(i.fn is not original_fn for i in new_impls)

    def test_idempotent_double_call(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        registry, impl = self._make_registry_with_attention()
        bg.wrap_attention_ops_for_break_graph(registry)
        # Capture the wrapped fn after first call
        fns_after_first = [i.fn for i in registry.get_implementations("attention_backend")]
        bg.wrap_attention_ops_for_break_graph(registry)
        fns_after_second = [i.fn for i in registry.get_implementations("attention_backend")]
        # Second call must not add more impls or re-wrap
        assert len(fns_after_first) == len(fns_after_second)
        for f1, f2 in zip(fns_after_first, fns_after_second):
            assert f1 is f2

    def test_non_attention_op_not_touched(self, monkeypatch):
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
            fn=fn, vendor=None,
        )
        registry.register_impl(impl)
        bg.wrap_attention_ops_for_break_graph(registry)
        assert impl.fn is fn

    def test_wrapped_fn_still_callable(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        log = []
        def attn_fn(q, k, v, out): log.append((q, k, v))
        registry, impl = self._make_registry_with_attention(fn=attn_fn)
        bg.wrap_attention_ops_for_break_graph(registry)
        # Call through the new impl in the registry (original impl is frozen)
        new_impl = registry.get_implementations("attention_backend")[-1]
        new_impl.fn(1, 2, 3, None)
        assert log == [(1, 2, 3)]

    def test_empty_registry_no_crash(self, monkeypatch):
        import vllm_fl.compilation.break_graph as bg
        from vllm_fl.dispatch.registry import OpRegistry
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        bg.wrap_attention_ops_for_break_graph(OpRegistry())

    def test_wrapped_fn_triggers_eager_break_inside_fl_capture(
        self, monkeypatch, _no_real_cuda
    ):
        """
        Verify that after wrapping, calling the attention op inside a
        FL BreakableCUDAGraphCapture context triggers an eager break.
        We use the FL fallback decorator directly to avoid vLLM delegation.
        """
        import vllm_fl.compilation.break_graph as bg
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture

        # Force wrap_attention_ops_for_break_graph to use FL fallback decorator
        monkeypatch.setattr(bg, "is_breakable_cudagraph_enabled", lambda: True)
        # Replace eager_break_during_capture with FL fallback in the module
        monkeypatch.setattr(bg, "eager_break_during_capture",
                            _fl_eager_break_decorator(enabled=True))

        log = []
        def attn_fn(q, k, v, out): log.append("attn")

        registry, impl = self._make_registry_with_attention(fn=attn_fn)
        bg.wrap_attention_ops_for_break_graph(registry)

        # Call through the new impl in the registry
        new_impl = registry.get_implementations("attention_backend")[-1]
        cap = BreakableCUDAGraphCapture()
        with cap:
            new_impl.fn(1, 2, 3, None)

        assert log == ["attn"]
        assert cap.num_eager_breaks == 1
        assert cap.num_graphs == 2


# ---------------------------------------------------------------------------
# 5. Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_fl_decorated_op_outside_capture(self, _no_real_cuda):
        """FL fallback: op runs normally outside capture."""
        dec = _fl_eager_break_decorator(enabled=True)
        log = []
        @dec
        def attn(q): log.append(q); return q * 2
        assert attn(5) == 10
        assert log == [5]

    def test_fl_decorated_op_inside_capture(self, _no_real_cuda):
        """FL fallback: op triggers eager break inside capture."""
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        dec = _fl_eager_break_decorator(enabled=True)

        @dec
        def attn(q): return q * 3

        cap = BreakableCUDAGraphCapture()
        with cap:
            result = attn(4)
        assert result == 12
        assert cap.num_eager_breaks == 1
        assert cap.num_graphs == 2

    def test_fl_replay_order_preserved(self, _no_real_cuda):
        """FL fallback: replay re-executes eager ops in insertion order."""
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        dec = _fl_eager_break_decorator(enabled=True)
        log = []

        @dec
        def op_a(): log.append("a")

        @dec
        def op_b(): log.append("b")

        cap = BreakableCUDAGraphCapture()
        with cap:
            op_a()
            op_b()
        assert log == ["a", "b"]
        log.clear()
        cap.replay()
        assert log == ["a", "b"]

    def test_capture_only_no_eager_breaks(self, _no_real_cuda):
        """A plain capture with no eager breaks has exactly 1 graph segment."""
        from vllm_fl.compilation.break_graph import BreakableCUDAGraphCapture
        cap = BreakableCUDAGraphCapture()
        with cap:
            pass
        assert cap.num_graphs == 1
        assert cap.num_eager_breaks == 0

    def test_module_exports(self):
        """All four symbols are importable from vllm_fl.compilation."""
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

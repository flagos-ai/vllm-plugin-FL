# Copyright (c) 2026 BAAI. All rights reserved.

"""
Unit tests for vllm_fl.compilation.break_graph

For the vLLM 0.28.0 target, break_graph.py remains a thin
re-export of upstream symbols.  These tests verify:

  1. All expected symbols are re-exported correctly from upstream.
  2. The re-exported symbols are identical to vLLM's originals.
  3. is_breakable_cudagraph_enabled responds to env var.
  4. eager_break_during_capture decorator preserves function behavior.
  5. BreakableCUDAGraphWrapper is importable for model_runner usage.
  6. wrap_attention_ops_for_break_graph is removed (no longer exported).

Import strategy: tests import vllm_fl.compilation.break_graph directly (not
via vllm_fl.compilation) to avoid triggering vllm_fl.__init__ which pulls in
FlagGems and requires GPU device detection.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Re-export identity — FL symbols ARE upstream symbols
# ---------------------------------------------------------------------------


class TestReExportIdentity:
    """Verify that vllm_fl.compilation.break_graph re-exports upstream symbols."""

    def test_is_breakable_cudagraph_enabled_is_upstream(self):
        from vllm.compilation.breakable_cudagraph import (
            is_breakable_cudagraph_enabled as upstream,
        )

        import vllm_fl.compilation.break_graph as bg

        assert bg.is_breakable_cudagraph_enabled is upstream

    def test_eager_break_during_capture_is_upstream(self):
        from vllm.compilation.breakable_cudagraph import (
            eager_break_during_capture as upstream,
        )

        import vllm_fl.compilation.break_graph as bg

        assert bg.eager_break_during_capture is upstream

    def test_breakable_cuda_graph_capture_is_upstream(self):
        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphCapture as upstream,
        )

        import vllm_fl.compilation.break_graph as bg

        assert bg.BreakableCUDAGraphCapture is upstream

    def test_breakable_cuda_graph_wrapper_is_upstream(self):
        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphWrapper as upstream,
        )

        import vllm_fl.compilation.break_graph as bg

        assert bg.BreakableCUDAGraphWrapper is upstream


# ---------------------------------------------------------------------------
# 2. Module-level __all__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """All expected symbols are defined in __all__."""

    def test_dunder_all(self):
        import vllm_fl.compilation.break_graph as bg

        expected = {
            "is_breakable_cudagraph_enabled",
            "eager_break_during_capture",
            "BreakableCUDAGraphCapture",
            "BreakableCUDAGraphWrapper",
        }
        assert set(bg.__all__) == expected

    def test_wrap_attention_ops_not_in_module(self):
        """wrap_attention_ops_for_break_graph was removed."""
        import vllm_fl.compilation.break_graph as bg

        assert not hasattr(bg, "wrap_attention_ops_for_break_graph")


# ---------------------------------------------------------------------------
# 3. is_breakable_cudagraph_enabled env-var behavior
# ---------------------------------------------------------------------------


class TestIsBreakableCudagraphEnabled:
    def test_returns_bool(self):
        import vllm_fl.compilation.break_graph as bg

        result = bg.is_breakable_cudagraph_enabled()
        assert isinstance(result, bool)

    def test_enabled_when_set(self, monkeypatch):
        import vllm.envs

        monkeypatch.setattr(vllm.envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", 1)
        import vllm_fl.compilation.break_graph as bg

        assert bg.is_breakable_cudagraph_enabled() is True

    def test_disabled_when_unset(self, monkeypatch):
        import vllm.envs

        monkeypatch.setattr(vllm.envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", 0)
        import vllm_fl.compilation.break_graph as bg

        assert bg.is_breakable_cudagraph_enabled() is False


# ---------------------------------------------------------------------------
# 4. eager_break_during_capture decorator basic behavior
# ---------------------------------------------------------------------------


class TestEagerBreakDecorator:
    """Test that the upstream decorator preserves function semantics."""

    def test_decorated_fn_callable_outside_capture(self):
        import vllm_fl.compilation.break_graph as bg

        @bg.eager_break_during_capture
        def my_op(x):
            return x * 2

        # Outside capture context, should execute normally
        assert my_op(5) == 10

    def test_preserves_function_name(self):
        import vllm_fl.compilation.break_graph as bg

        @bg.eager_break_during_capture
        def my_attention_op(q, k, v):
            return q

        assert my_attention_op.__name__ == "my_attention_op"

    def test_preserves_return_value(self):
        import vllm_fl.compilation.break_graph as bg

        @bg.eager_break_during_capture
        def add(a, b):
            return a + b

        assert add(3, 4) == 7


# ---------------------------------------------------------------------------
# 5. BreakableCUDAGraphCapture upstream class properties
# ---------------------------------------------------------------------------


class TestBreakableCUDAGraphCapture:
    """Verify upstream BreakableCUDAGraphCapture is usable from FL."""

    def test_current_none_outside_context(self):
        import vllm_fl.compilation.break_graph as bg

        assert bg.BreakableCUDAGraphCapture.current() is None

    def test_is_active_false_outside(self):
        import vllm_fl.compilation.break_graph as bg

        assert bg.BreakableCUDAGraphCapture.is_active() is False

    def test_has_expected_interface(self):
        import vllm_fl.compilation.break_graph as bg

        cls = bg.BreakableCUDAGraphCapture
        assert hasattr(cls, "current")
        assert hasattr(cls, "is_active")
        assert hasattr(cls, "__enter__")
        assert hasattr(cls, "__exit__")


# ---------------------------------------------------------------------------
# 6. BreakableCUDAGraphWrapper interface
# ---------------------------------------------------------------------------


class TestBreakableCUDAGraphWrapper:
    """Verify upstream BreakableCUDAGraphWrapper has unwrap method."""

    def test_has_unwrap_method(self):
        import vllm_fl.compilation.break_graph as bg

        assert hasattr(bg.BreakableCUDAGraphWrapper, "unwrap")

    def test_class_is_not_nn_module(self):
        import torch.nn as nn

        import vllm_fl.compilation.break_graph as bg

        assert not issubclass(bg.BreakableCUDAGraphWrapper, nn.Module)


# ---------------------------------------------------------------------------
# 7. builtin_ops no longer calls wrap_attention_ops_for_break_graph
# ---------------------------------------------------------------------------


class TestBuiltinOpsNoWrap:
    """Verify the wrap_attention_ops_for_break_graph code is removed."""

    def test_no_break_graph_import_in_builtin_ops(self):
        import pathlib

        # Read the source file directly to avoid triggering vllm_fl.__init__
        bo_path = (
            pathlib.Path(__file__).resolve().parents[3]
            / "vllm_fl"
            / "dispatch"
            / "builtin_ops.py"
        )
        source = bo_path.read_text()
        assert "wrap_attention_ops_for_break_graph" not in source

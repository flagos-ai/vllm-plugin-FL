# Copyright (c) 2026 BAAI. All rights reserved.
#
# Break-graph support for vllm-plugin-FL.
#
# vLLM ≥ 0.24.0 introduced ``breakable_cudagraph`` — a mode where a single
# forward-pass stream capture is split at attention / kv-cache boundaries
# into alternating graph-segments and eager-segments, avoiding the need for
# torch.compile FX-graph splitting.
#
# This module provides:
#
#   1.  ``is_breakable_cudagraph_enabled()``           — thin proxy to the vLLM
#       env var (falls back to reading it directly on older vLLM builds).
#   2.  ``eager_break_during_capture(fn)``             — decorator that turns a
#       custom attention / kv-cache op into a break-point during capture.
#       Delegates to vLLM's implementation when available; otherwise uses the
#       self-contained fallback built on :class:`BreakableCUDAGraphCapture`.
#   3.  ``BreakableCUDAGraphCapture``                  — self-contained capture
#       context re-implemented against plain ``torch.cuda``.  Used by unit
#       tests and by OOT backends that drive capture directly.
#   4.  ``wrap_attention_ops_for_break_graph(registry)`` — post-registration
#       hook called by ``builtin_ops.register_builtins()``.  Wraps every
#       ``attention_backend`` OpImpl in the registry with
#       ``eager_break_during_capture`` so that OOT vendor attention ops
#       participate correctly in breakable CUDA graph capture.
#
# Import strategy
# ---------------
# We always try to import the symbols from vLLM first (canonical production
# implementation).  If the running vLLM version does not yet expose them
# (pre-0.24.0 or stripped build) we fall back to our own implementations so
# that the rest of the plugin keeps working transparently.
#
# TODO: once the minimum supported vLLM version is >= 0.24.0 and the upstream
#       symbols are stable, the fallback implementations can be removed and
#       this module can become a thin re-export.

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import torch

__all__ = [
    "is_breakable_cudagraph_enabled",
    "eager_break_during_capture",
    "BreakableCUDAGraphCapture",
    "wrap_attention_ops_for_break_graph",
]

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# is_breakable_cudagraph_enabled
# ---------------------------------------------------------------------------

try:
    from vllm.compilation.breakable_cudagraph import (  # type: ignore[import]
        is_breakable_cudagraph_enabled as _vllm_is_breakable_enabled,
    )

    def is_breakable_cudagraph_enabled() -> bool:
        """Return True when ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` is set."""
        return _vllm_is_breakable_enabled()

except ImportError:
    import os

    def is_breakable_cudagraph_enabled() -> bool:  # type: ignore[misc]
        """Return True when ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` is set."""
        return os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH", "0") not in (
            "0", "", "false", "False",
        )


# ---------------------------------------------------------------------------
# BreakableCUDAGraphCapture  (self-contained, no vLLM internal imports)
# ---------------------------------------------------------------------------

class BreakableCUDAGraphCapture:
    """Stream-capture context that supports eager break-points.

    A *break-point* ends the current CUDA graph segment, executes a callable
    eagerly on the capture stream, records that callable for replay, and
    starts a fresh segment.  The resulting capture artifact is an ordered
    list of zero-arg callables — ``CUDAGraph.replay`` for graph segments and
    the original callable for eager segments — that are executed in order
    during replay.

    This implementation depends only on ``torch.cuda`` so that it can be
    used (and unit-tested) without the full vLLM internal stack.

    Thread safety
    -------------
    Only one capture may be active **per thread** at any time.  A
    ``RuntimeError`` is raised when nesting is attempted.

    Example::

        cap = BreakableCUDAGraphCapture()
        with cap:
            output = model(*static_inputs)
        # Later, after copying new data into static input buffers:
        cap.replay()
    """

    _tls: threading.local = threading.local()

    @classmethod
    def current(cls) -> "BreakableCUDAGraphCapture | None":
        """Return the active capture for the current thread, or ``None``."""
        return getattr(cls._tls, "active", None)

    @classmethod
    def is_active(cls) -> bool:
        """Return ``True`` if a capture is currently active on this thread."""
        return cls.current() is not None

    def __init__(self, pool: Any | None = None) -> None:
        self.pool = pool
        self.segments: list[Callable[[], Any]] = []
        self._num_graphs: int = 0
        self._num_eager_breaks: int = 0
        self._current_graph: torch.cuda.CUDAGraph | None = None
        self._capturing: bool = False

    def __enter__(self) -> "BreakableCUDAGraphCapture":
        if getattr(BreakableCUDAGraphCapture._tls, "active", None) is not None:
            raise RuntimeError(
                "Nested BreakableCUDAGraphCapture is not supported."
            )
        BreakableCUDAGraphCapture._tls.active = self
        self._begin_segment()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self._end_segment()
        finally:
            BreakableCUDAGraphCapture._tls.active = None

    def _begin_segment(self) -> None:
        assert not self._capturing, "Already capturing a segment."
        g = torch.cuda.CUDAGraph()
        if self.pool is not None:
            g.capture_begin(pool=self.pool)
        else:
            g.capture_begin()
        self._current_graph = g
        self._capturing = True

    def _end_segment(self) -> None:
        if not self._capturing:
            return
        assert self._current_graph is not None
        self._current_graph.capture_end()
        self.segments.append(self._current_graph.replay)
        self._num_graphs += 1
        self._current_graph = None
        self._capturing = False

    def add_eager(self, fn: Callable[[], Any]) -> Any:
        """Insert an eager break-point.

        Ends the current graph segment, executes *fn* eagerly, records it
        for replay, then starts a new segment.

        Args:
            fn: Zero-arg callable (typically a lambda closing over static
                tensor buffers).

        Returns:
            The return value of ``fn()`` from capture-time execution.
        """
        self._end_segment()
        result = fn()
        self.segments.append(fn)
        self._num_eager_breaks += 1
        self._begin_segment()
        return result

    def replay(self) -> None:
        """Replay all captured segments in order."""
        for segment in self.segments:
            segment()

    @property
    def num_graphs(self) -> int:
        """Number of captured CUDA graph segments."""
        return self._num_graphs

    @property
    def num_eager_breaks(self) -> int:
        """Number of eager break-points inserted during capture."""
        return self._num_eager_breaks

    def __repr__(self) -> str:
        return (
            f"BreakableCUDAGraphCapture("
            f"graphs={self.num_graphs}, "
            f"eager_breaks={self.num_eager_breaks})"
        )


# ---------------------------------------------------------------------------
# eager_break_during_capture
# ---------------------------------------------------------------------------

try:
    from vllm.compilation.breakable_cudagraph import (  # type: ignore[import]
        eager_break_during_capture as _vllm_eager_break,
    )

    def eager_break_during_capture(fn: F) -> F:
        """Decorator: turn a custom op into a break-point during capture.

        Delegates to the vLLM implementation when available so that the
        official vLLM capture machinery cooperates with this decorator.
        When ``VLLM_USE_BREAKABLE_CUDAGRAPH`` is not set the decorator is a
        no-op and the original function is returned unchanged.
        """
        return _vllm_eager_break(fn)

except ImportError:
    def eager_break_during_capture(fn: F) -> F:  # type: ignore[misc]
        """Decorator: turn a custom op into a break-point during capture.

        Fallback used when vLLM < 0.24.0 or the symbol is not available.

        When ``VLLM_USE_BREAKABLE_CUDAGRAPH`` is not set, this is a no-op
        and the original function is returned unchanged.

        When active, the decorated function:
          * Runs normally outside a capture context.
          * Inside a :class:`BreakableCUDAGraphCapture` context, ends the
            current segment, runs eagerly, and starts a new segment.

        **Important:** Decorated ops must write results into caller-provided
        output buffers so that downstream graph segments read from stable
        memory addresses on replay.
        """
        if not is_breakable_cudagraph_enabled():
            return fn

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            capture = BreakableCUDAGraphCapture.current()
            if capture is None or not capture._capturing:
                return fn(*args, **kwargs)
            return capture.add_eager(lambda: fn(*args, **kwargs))

        return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# wrap_attention_ops_for_break_graph
# ---------------------------------------------------------------------------

#: Op names that act as break-points during CUDA graph capture.
#: ``attention_backend`` is the primary one; extend this set if kv-cache
#: ops are ever registered through the dispatch system as well.
_BREAK_POINT_OP_NAMES: frozenset[str] = frozenset({
    "attention_backend",
})


def wrap_attention_ops_for_break_graph(registry: Any) -> None:
    """Wrap registered attention ops with ``eager_break_during_capture``.

    Called by ``builtin_ops.register_builtins()`` **after** all vendor ops
    have been registered.  For every :class:`~vllm_fl.dispatch.types.OpImpl`
    in *registry* whose ``op_name`` is in :data:`_BREAK_POINT_OP_NAMES`,
    the ``fn`` attribute is replaced with a break-point-aware wrapper.

    This is a post-registration hook rather than a per-vendor change so that
    the break-graph behaviour is applied uniformly to **all** backends
    (FlagGems, Reference, and every OOT vendor) without modifying each
    individual ``register_ops.py``.

    The function is idempotent: if break-graph capture is not enabled
    (``VLLM_USE_BREAKABLE_CUDAGRAPH != 1``) it returns immediately without
    touching the registry.

    Args:
        registry: An :class:`~vllm_fl.dispatch.registry.OpRegistry` instance.
    """
    if not is_breakable_cudagraph_enabled():
        return

    try:
        from vllm_fl.dispatch.logger_manager import get_logger
        _log = get_logger()
    except Exception:
        import logging
        _log = logging.getLogger(__name__)

    wrapped_count = 0
    for op_name in _BREAK_POINT_OP_NAMES:
        try:
            impls = registry.get_implementations(op_name)
        except Exception:
            continue
        if not impls:
            continue
        for impl in impls:
            original_fn = impl.fn
            # Guard: skip if already wrapped (idempotency across repeated calls)
            if getattr(original_fn, "_break_graph_wrapped", False):
                continue
            wrapped_fn = eager_break_during_capture(original_fn)
            # Mark so we don't double-wrap on repeated calls
            try:
                wrapped_fn._break_graph_wrapped = True
            except (AttributeError, TypeError):
                pass
            impl.fn = wrapped_fn
            wrapped_count += 1

    if wrapped_count:
        _log.info(
            "wrap_attention_ops_for_break_graph: wrapped %d impl(s) "
            "as eager break-points for CUDA graph capture.",
            wrapped_count,
        )

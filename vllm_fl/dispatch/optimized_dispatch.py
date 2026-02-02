# Copyright (c) 2026 BAAI. All rights reserved.

"""
Optimized Dispatch Integration Layer.

This module integrates all optimization modules:
1. Adaptive Learning - Runtime backend selection learning
2. Decode Optimizer - Fast paths for decode phase
3. Dynamic Thresholds - Shape-aware dispatch decisions

Usage:
    from vllm_fl.dispatch.optimized_dispatch import optimized_call

    # Automatically uses all optimizations
    result = optimized_call('silu_and_mul', x)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .manager import get_default_manager
from .adaptive_learning import get_adaptive_learner, record_performance, get_optimal_backend
from .decode_optimizer import get_decode_optimizer, DecodeMode
from .dynamic_threshold import get_threshold_manager, should_use_flaggems

logger = logging.getLogger(__name__)


class OptimizedDispatcher:
    """
    Unified dispatcher with all optimizations integrated.

    Optimization flow:
    1. Check decode optimizer (fast path bypass)
    2. Check adaptive learning (learned preferences)
    3. Check dynamic thresholds (shape-aware)
    4. Fall back to default manager

    Performance tracking:
    - Records execution time for adaptive learning
    - Tracks dispatch decisions for debugging
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Get component instances
        self._manager = get_default_manager()
        self._adaptive = get_adaptive_learner()
        self._decode_opt = get_decode_optimizer()
        self._threshold_mgr = get_threshold_manager()

        # Configuration
        self._enable_adaptive = os.environ.get("ENABLE_ADAPTIVE_LEARNING", "1") == "1"
        self._enable_decode_opt = os.environ.get("ENABLE_DECODE_OPT", "1") == "1"
        self._enable_dynamic_threshold = os.environ.get("ENABLE_DYNAMIC_THRESHOLD", "1") == "1"
        self._enable_perf_tracking = os.environ.get("ENABLE_PERF_TRACKING", "0") == "1"

        # Statistics
        self._stats = {
            'total_calls': 0,
            'fast_path_hits': 0,
            'adaptive_hits': 0,
            'threshold_hits': 0,
            'default_fallback': 0,
        }

        # Thread-local context
        self._local = threading.local()

        logger.info(
            f"OptimizedDispatcher initialized "
            f"(adaptive={self._enable_adaptive}, decode_opt={self._enable_decode_opt}, "
            f"dynamic_threshold={self._enable_dynamic_threshold})"
        )

    def _extract_shape_info(self, args, kwargs) -> Tuple[int, int, int]:
        """Extract batch_size, seq_len, hidden from arguments."""
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shape = arg.shape
                if len(shape) == 2:
                    return shape[0], 1, shape[1]
                elif len(shape) == 3:
                    return shape[0], shape[1], shape[2]
                elif len(shape) == 1:
                    return 1, 1, shape[0]
        return 1, 1, 4096  # Default

    def call(self, op_name: str, *args, **kwargs) -> Any:
        """
        Optimized operator call with all optimizations.

        Args:
            op_name: Operator name
            *args, **kwargs: Operator arguments

        Returns:
            Operator result
        """
        self._stats['total_calls'] += 1

        # Extract shape info for decision making
        batch_size, seq_len, hidden_size = self._extract_shape_info(args, kwargs)
        is_decode = self._decode_opt.is_decode_phase

        # 1. Check decode fast path (highest priority)
        if self._enable_decode_opt and is_decode:
            fast_path = self._decode_opt.get_fast_path(op_name)
            if fast_path is not None:
                self._stats['fast_path_hits'] += 1
                return fast_path(*args, **kwargs)

        # 2. Check adaptive learning
        if self._enable_adaptive:
            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break

            if tensor is not None:
                preferred = get_optimal_backend(op_name, tensor)
                if preferred is not None:
                    self._stats['adaptive_hits'] += 1
                    # Use preferred backend
                    return self._call_with_backend(op_name, preferred, args, kwargs, tensor)

        # 3. Check dynamic thresholds
        if self._enable_dynamic_threshold:
            use_flaggems = should_use_flaggems(
                op_name, batch_size, seq_len, hidden_size, is_decode
            )
            if use_flaggems:
                self._stats['threshold_hits'] += 1
                return self._call_with_backend(op_name, 'gems', args, kwargs, None)

        # 4. Fall back to default manager
        self._stats['default_fallback'] += 1
        return self._manager.call(op_name, *args, **kwargs)

    def _call_with_backend(
        self,
        op_name: str,
        backend: str,
        args: tuple,
        kwargs: dict,
        tensor: Optional[torch.Tensor],
    ) -> Any:
        """Call operator with specific backend and optionally record performance."""
        start_time = time.perf_counter() if self._enable_perf_tracking else None

        # Get implementation based on backend preference
        try:
            # For now, use default manager - could be extended to force specific backend
            result = self._manager.call(op_name, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Backend {backend} failed for {op_name}: {e}")
            raise

        # Record performance for adaptive learning
        if self._enable_perf_tracking and start_time and tensor is not None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            record_performance(op_name, tensor, duration_ms, backend)

        return result

    # ==================== Context Management ====================

    def enter_decode_phase(self, batch_size: int = 1):
        """Enter decode phase for optimizations."""
        self._decode_opt.enter_decode(batch_size, DecodeMode.ULTRA_LOW_LATENCY)

    def exit_decode_phase(self):
        """Exit decode phase."""
        self._decode_opt.exit_decode()

    @contextmanager
    def decode_context(self, batch_size: int = 1):
        """Context manager for decode phase."""
        self.enter_decode_phase(batch_size)
        try:
            yield
        finally:
            self.exit_decode_phase()

    @contextmanager
    def prefill_context(self, batch_size: int = 1, seq_len: int = 1):
        """Context manager for prefill phase (no decode optimizations)."""
        # Ensure we're not in decode mode
        was_decode = self._decode_opt.is_decode_phase
        if was_decode:
            self._decode_opt.exit_decode()
        try:
            yield
        finally:
            if was_decode:
                self._decode_opt.enter_decode(1, DecodeMode.NORMAL)

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, int]:
        """Get dispatch statistics."""
        return self._stats.copy()

    def print_report(self):
        """Print comprehensive optimization report."""
        print("\n" + "=" * 80)
        print("OPTIMIZED DISPATCHER REPORT")
        print("=" * 80)

        total = self._stats['total_calls']
        if total == 0:
            print("\nNo calls recorded.")
            return

        print(f"\nTotal calls: {total}")
        print(f"\nOptimization breakdown:")
        print(f"  Fast path hits:     {self._stats['fast_path_hits']:>8} ({self._stats['fast_path_hits']/total*100:.1f}%)")
        print(f"  Adaptive hits:      {self._stats['adaptive_hits']:>8} ({self._stats['adaptive_hits']/total*100:.1f}%)")
        print(f"  Threshold hits:     {self._stats['threshold_hits']:>8} ({self._stats['threshold_hits']/total*100:.1f}%)")
        print(f"  Default fallback:   {self._stats['default_fallback']:>8} ({self._stats['default_fallback']/total*100:.1f}%)")

        # Component reports
        print("\n" + "-" * 80)
        self._adaptive.print_report()

        print("\n" + "-" * 80)
        self._decode_opt.print_report()

        print("\n" + "-" * 80)
        self._threshold_mgr.print_thresholds()


# Global instance
_optimized_dispatcher: Optional[OptimizedDispatcher] = None


def get_optimized_dispatcher() -> OptimizedDispatcher:
    """Get the global optimized dispatcher."""
    global _optimized_dispatcher
    if _optimized_dispatcher is None:
        _optimized_dispatcher = OptimizedDispatcher()
    return _optimized_dispatcher


# ==================== Convenience API ====================

def optimized_call(op_name: str, *args, **kwargs) -> Any:
    """
    Call operator with all optimizations enabled.

    This is the main entry point for optimized dispatch.
    """
    return get_optimized_dispatcher().call(op_name, *args, **kwargs)


def enter_decode_phase(batch_size: int = 1):
    """Enter decode phase for optimizations."""
    get_optimized_dispatcher().enter_decode_phase(batch_size)


def exit_decode_phase():
    """Exit decode phase."""
    get_optimized_dispatcher().exit_decode_phase()


def decode_context(batch_size: int = 1):
    """Context manager for decode phase."""
    return get_optimized_dispatcher().decode_context(batch_size)


def prefill_context(batch_size: int = 1, seq_len: int = 1):
    """Context manager for prefill phase."""
    return get_optimized_dispatcher().prefill_context(batch_size, seq_len)


def print_optimization_report():
    """Print comprehensive optimization report."""
    get_optimized_dispatcher().print_report()

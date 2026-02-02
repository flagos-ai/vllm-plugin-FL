# Copyright (c) 2026 BAAI. All rights reserved.

"""
Context-Aware Dispatch for FlagGems Integration.

This module extends the base dispatch manager to support:
- Phase-aware dispatch (prefill vs decode)
- Shape-aware dispatch (tensor dimensions, batch size)
- Tiered operator policies
- Runtime performance tracking

The goal is to enable FlagGems only where it provides benefit,
while preserving vLLM's optimized kernels for critical paths.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .operator_policy import (
    InferencePhase,
    OperatorTier,
    get_policy_manager,
    should_replace_operator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Context Tracking
# =============================================================================

@dataclass
class DispatchContext:
    """Runtime context for dispatch decisions."""
    phase: InferencePhase = InferencePhase.MIXED
    batch_size: int = 1
    seq_len: int = 1
    is_first_token: bool = False  # First decode token (TTFT critical)
    request_id: Optional[str] = None


class ContextManager:
    """
    Thread-local context manager for inference phase tracking.

    This allows different inference requests (in continuous batching)
    to have independent context without locking.
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
        self._local = threading.local()
        self._global_context = DispatchContext()

    def _get_context(self) -> DispatchContext:
        """Get thread-local context, falling back to global."""
        if hasattr(self._local, 'context'):
            return self._local.context
        return self._global_context

    def set_phase(self, phase: InferencePhase):
        """Set current inference phase."""
        ctx = self._get_context()
        ctx.phase = phase
        # Also update policy manager
        get_policy_manager().set_context(phase=phase)

    def set_batch_info(self, batch_size: int, seq_len: int):
        """Set batch information."""
        ctx = self._get_context()
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        get_policy_manager().set_context(batch_size=batch_size, seq_len=seq_len)

    def set_first_token(self, is_first: bool):
        """Mark if this is the first decode token (TTFT)."""
        ctx = self._get_context()
        ctx.is_first_token = is_first

    @property
    def phase(self) -> InferencePhase:
        return self._get_context().phase

    @property
    def batch_size(self) -> int:
        return self._get_context().batch_size

    @property
    def seq_len(self) -> int:
        return self._get_context().seq_len

    @property
    def is_first_token(self) -> bool:
        return self._get_context().is_first_token

    @contextmanager
    def prefill_context(self, batch_size: int = 1, seq_len: int = 1):
        """Context manager for prefill phase."""
        old_phase = self.phase
        old_batch = self.batch_size
        old_seq = self.seq_len
        try:
            self.set_phase(InferencePhase.PREFILL)
            self.set_batch_info(batch_size, seq_len)
            yield
        finally:
            self.set_phase(old_phase)
            self.set_batch_info(old_batch, old_seq)

    @contextmanager
    def decode_context(self, batch_size: int = 1, is_first: bool = False):
        """Context manager for decode phase."""
        old_phase = self.phase
        old_batch = self.batch_size
        old_first = self.is_first_token
        try:
            self.set_phase(InferencePhase.DECODE)
            self.set_batch_info(batch_size, seq_len=1)
            self.set_first_token(is_first)
            yield
        finally:
            self.set_phase(old_phase)
            self.set_batch_info(old_batch, self.seq_len)
            self.set_first_token(old_first)


# Global context manager
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


# =============================================================================
# Performance Tracking
# =============================================================================

@dataclass
class OperatorStats:
    """Performance statistics for an operator."""
    call_count: int = 0
    total_time_ms: float = 0.0
    gems_calls: int = 0
    native_calls: int = 0
    gems_time_ms: float = 0.0
    native_time_ms: float = 0.0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0

    @property
    def gems_avg_ms(self) -> float:
        return self.gems_time_ms / self.gems_calls if self.gems_calls > 0 else 0

    @property
    def native_avg_ms(self) -> float:
        return self.native_time_ms / self.native_calls if self.native_calls > 0 else 0


class PerformanceTracker:
    """
    Tracks operator performance for adaptive dispatch decisions.

    This enables runtime learning of which operators benefit from GEMs.
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
        self._stats: Dict[str, OperatorStats] = {}
        self._enabled = os.environ.get("GEMS_PERF_TRACKING", "0") == "1"
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(
        self,
        op_name: str,
        duration_ms: float,
        used_gems: bool,
    ):
        """Record an operator call."""
        if not self._enabled:
            return

        with self._lock:
            if op_name not in self._stats:
                self._stats[op_name] = OperatorStats()

            stats = self._stats[op_name]
            stats.call_count += 1
            stats.total_time_ms += duration_ms

            if used_gems:
                stats.gems_calls += 1
                stats.gems_time_ms += duration_ms
            else:
                stats.native_calls += 1
                stats.native_time_ms += duration_ms

    def get_stats(self, op_name: str) -> Optional[OperatorStats]:
        """Get stats for an operator."""
        return self._stats.get(op_name)

    def should_prefer_gems(self, op_name: str, threshold: float = 0.9) -> Optional[bool]:
        """
        Determine if GEMs should be preferred based on observed performance.

        Returns:
            True if GEMs is faster
            False if native is faster
            None if not enough data
        """
        stats = self._stats.get(op_name)
        if stats is None:
            return None

        # Need at least 10 calls of each to compare
        if stats.gems_calls < 10 or stats.native_calls < 10:
            return None

        gems_avg = stats.gems_avg_ms
        native_avg = stats.native_avg_ms

        if gems_avg < native_avg * threshold:
            return True
        elif native_avg < gems_avg * threshold:
            return False
        return None  # Too close to call

    def print_report(self):
        """Print performance report."""
        if not self._stats:
            print("No performance data collected.")
            return

        print("\n" + "="*80)
        print("OPERATOR PERFORMANCE REPORT")
        print("="*80)
        print(f"\n{'Operator':<30} {'Calls':<10} {'GEMs Avg':<12} {'Native Avg':<12} {'Winner':<10}")
        print("-"*80)

        for op_name, stats in sorted(self._stats.items()):
            gems_avg = f"{stats.gems_avg_ms:.3f}ms" if stats.gems_calls > 0 else "N/A"
            native_avg = f"{stats.native_avg_ms:.3f}ms" if stats.native_calls > 0 else "N/A"

            winner = "-"
            if stats.gems_calls >= 10 and stats.native_calls >= 10:
                if stats.gems_avg_ms < stats.native_avg_ms * 0.9:
                    winner = "GEMs"
                elif stats.native_avg_ms < stats.gems_avg_ms * 0.9:
                    winner = "Native"
                else:
                    winner = "Tie"

            print(f"{op_name:<30} {stats.call_count:<10} {gems_avg:<12} {native_avg:<12} {winner:<10}")


# Global performance tracker
_perf_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker."""
    global _perf_tracker
    if _perf_tracker is None:
        _perf_tracker = PerformanceTracker()
    return _perf_tracker


# =============================================================================
# Context-Aware Dispatch Functions
# =============================================================================

def should_use_gems(
    op_name: str,
    tensor: Optional[torch.Tensor] = None,
    force_native: bool = False,
) -> bool:
    """
    Determine if an operator should use FlagGems implementation.

    This is the main entry point for context-aware dispatch decisions.

    Args:
        op_name: Name of the aten operator
        tensor: Primary input tensor (for shape-based decisions)
        force_native: Force native implementation

    Returns:
        True if GEMs should be used, False for native
    """
    if force_native:
        return False

    # Check if GEMs is enabled at all
    use_flaggems = os.environ.get("USE_FLAGGEMS", "True").lower() == "true"
    if not use_flaggems:
        return False

    # Get context
    ctx = get_context_manager()

    # Prepare context for policy check
    tensor_elements = tensor.numel() if tensor is not None else 0

    # Check tiered policy
    should_replace = should_replace_operator(
        op_name,
        tensor_elements=tensor_elements,
    )

    # Additional decode-phase protection
    if ctx.phase == InferencePhase.DECODE:
        # During decode, be extra conservative
        policy = get_policy_manager().get_policy(op_name)
        if policy is not None and policy.tier != OperatorTier.TIER_2_SAFE:
            return False

    # First token (TTFT) protection - use native for lowest latency
    if ctx.is_first_token:
        return False

    return should_replace


def wrap_with_context_dispatch(
    op_name: str,
    gems_impl: Callable,
    native_impl: Callable,
) -> Callable:
    """
    Create a wrapper that dispatches between GEMs and native based on context.

    Args:
        op_name: Operator name for policy lookup
        gems_impl: FlagGems implementation
        native_impl: Native (vLLM/CUDA) implementation

    Returns:
        Wrapped function that dispatches based on context
    """
    tracker = get_performance_tracker()

    def dispatch_wrapper(*args, **kwargs):
        # Get primary tensor for shape info
        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break

        use_gems = should_use_gems(op_name, tensor)

        if tracker.enabled:
            start = time.perf_counter()

        try:
            if use_gems:
                result = gems_impl(*args, **kwargs)
            else:
                result = native_impl(*args, **kwargs)
        finally:
            if tracker.enabled:
                duration_ms = (time.perf_counter() - start) * 1000
                tracker.record(op_name, duration_ms, use_gems)

        return result

    dispatch_wrapper.__name__ = f"context_dispatch_{op_name}"
    return dispatch_wrapper


# =============================================================================
# Convenience API
# =============================================================================

def enter_prefill_phase(batch_size: int = 1, seq_len: int = 1):
    """Enter prefill phase."""
    get_context_manager().set_phase(InferencePhase.PREFILL)
    get_context_manager().set_batch_info(batch_size, seq_len)


def enter_decode_phase(batch_size: int = 1, is_first_token: bool = False):
    """Enter decode phase."""
    get_context_manager().set_phase(InferencePhase.DECODE)
    get_context_manager().set_batch_info(batch_size, seq_len=1)
    get_context_manager().set_first_token(is_first_token)


def get_current_phase() -> InferencePhase:
    """Get current inference phase."""
    return get_context_manager().phase


def print_dispatch_summary():
    """Print summary of dispatch decisions and performance."""
    print("\n" + "="*80)
    print("CONTEXT-AWARE DISPATCH SUMMARY")
    print("="*80)

    ctx = get_context_manager()
    print(f"\nCurrent Phase: {ctx.phase.name}")
    print(f"Batch Size: {ctx.batch_size}")
    print(f"Seq Length: {ctx.seq_len}")

    # Policy summary
    get_policy_manager().print_summary()

    # Performance summary
    if get_performance_tracker().enabled:
        get_performance_tracker().print_report()

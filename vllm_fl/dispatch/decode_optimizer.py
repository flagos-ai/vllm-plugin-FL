# Copyright (c) 2026 BAAI. All rights reserved.

"""
Decode Path Optimizer for vLLM-FL.

This module provides specialized optimizations for the decode phase,
where latency is critical and batch sizes are typically small.

Key Optimizations:
1. Bypass FlagGems entirely for decode-critical operators
2. Minimize dispatch overhead
3. Pre-compiled fast paths for common decode shapes
4. CUDA Graph support for ultra-low latency
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import torch

logger = logging.getLogger(__name__)


class DecodeMode(Enum):
    """Decode optimization modes."""
    NORMAL = auto()        # Standard decode path
    ULTRA_LOW_LATENCY = auto()  # Minimize all overheads
    CUDA_GRAPH = auto()    # Use CUDA graphs


@dataclass
class DecodeConfig:
    """Configuration for decode optimizations."""
    # Operators that ALWAYS use CUDA during decode (never FlagGems)
    cuda_only_ops: Set[str]
    # Maximum batch size to apply decode optimizations
    max_decode_batch: int
    # Enable CUDA graph capture
    enable_cuda_graph: bool
    # Bypass dispatch system entirely for these ops
    fast_path_ops: Set[str]


# Default decode configuration based on benchmarks
DEFAULT_DECODE_CONFIG = DecodeConfig(
    cuda_only_ops={
        # These ops are 9-11x faster with CUDA during decode
        'silu_and_mul',
        'rmsnorm',
        'fused_add_rmsnorm',
        'layer_norm',
        # Attention-related (should always be FlashAttention)
        'attention',
        'paged_attention',
        'flash_attention',
        # Matrix ops (cuBLAS optimized)
        'mm',
        'addmm',
        'bmm',
        'linear',
    },
    max_decode_batch=32,  # Above this, optimizations may not help
    enable_cuda_graph=True,
    fast_path_ops={
        'silu_and_mul',
        'rmsnorm',
    },
)


class DecodePathOptimizer:
    """
    Optimizer for decode phase operations.

    This class provides:
    1. Fast paths that bypass dispatch overhead
    2. CUDA-only enforcement for critical operators
    3. CUDA graph support for repeated decode steps
    4. Batch-size aware optimization decisions
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

        # Configuration
        self._config = DEFAULT_DECODE_CONFIG
        self._mode = DecodeMode.NORMAL

        # Fast path implementations (bypass dispatch)
        self._fast_paths: Dict[str, Callable] = {}

        # CUDA graph related
        self._cuda_graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self._graph_inputs: Dict[str, List[torch.Tensor]] = {}
        self._graph_outputs: Dict[str, List[torch.Tensor]] = {}

        # Thread-local decode state
        self._local = threading.local()

        # Statistics
        self._stats = {
            'fast_path_calls': 0,
            'dispatch_bypassed': 0,
            'cuda_graph_runs': 0,
            'total_decode_ops': 0,
        }

        # Register fast paths
        self._register_fast_paths()

        logger.info("DecodePathOptimizer initialized")

    def _register_fast_paths(self):
        """Register fast path implementations for decode-critical ops."""

        # silu_and_mul fast path
        try:
            from vllm._custom_ops import silu_and_mul as vllm_silu_and_mul
            self._fast_paths['silu_and_mul'] = vllm_silu_and_mul
            logger.debug("Registered silu_and_mul fast path")
        except ImportError:
            # Fallback to PyTorch
            def silu_and_mul_fallback(x: torch.Tensor) -> torch.Tensor:
                d = x.shape[-1] // 2
                return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
            self._fast_paths['silu_and_mul'] = silu_and_mul_fallback
            logger.debug("Using silu_and_mul fallback")

        # rmsnorm fast path
        try:
            from vllm._custom_ops import rms_norm as vllm_rms_norm
            self._fast_paths['rmsnorm'] = vllm_rms_norm
            logger.debug("Registered rmsnorm fast path")
        except ImportError:
            def rmsnorm_fallback(
                x: torch.Tensor,
                weight: torch.Tensor,
                epsilon: float = 1e-6
            ) -> torch.Tensor:
                variance = x.pow(2).mean(-1, keepdim=True)
                x_norm = x * torch.rsqrt(variance + epsilon)
                return x_norm * weight
            self._fast_paths['rmsnorm'] = rmsnorm_fallback
            logger.debug("Using rmsnorm fallback")

    @property
    def is_decode_phase(self) -> bool:
        """Check if we're in decode phase."""
        return getattr(self._local, 'is_decode', False)

    @property
    def decode_batch_size(self) -> int:
        """Get current decode batch size."""
        return getattr(self._local, 'batch_size', 1)

    def enter_decode(self, batch_size: int = 1, mode: DecodeMode = DecodeMode.NORMAL):
        """Enter decode phase."""
        self._local.is_decode = True
        self._local.batch_size = batch_size
        self._local.mode = mode
        self._mode = mode

    def exit_decode(self):
        """Exit decode phase."""
        self._local.is_decode = False
        self._local.batch_size = 1
        self._local.mode = DecodeMode.NORMAL
        self._mode = DecodeMode.NORMAL

    @contextmanager
    def decode_context(self, batch_size: int = 1, mode: DecodeMode = DecodeMode.NORMAL):
        """Context manager for decode phase."""
        self.enter_decode(batch_size, mode)
        try:
            yield
        finally:
            self.exit_decode()

    def should_use_fast_path(self, op_name: str) -> bool:
        """Check if fast path should be used for an operator."""
        if not self.is_decode_phase:
            return False

        if self.decode_batch_size > self._config.max_decode_batch:
            return False

        return op_name in self._config.fast_path_ops

    def should_force_cuda(self, op_name: str) -> bool:
        """Check if CUDA should be forced for an operator."""
        if not self.is_decode_phase:
            return False

        if self.decode_batch_size > self._config.max_decode_batch:
            return False

        return op_name in self._config.cuda_only_ops

    def get_fast_path(self, op_name: str) -> Optional[Callable]:
        """Get fast path implementation if available."""
        if self.should_use_fast_path(op_name):
            self._stats['fast_path_calls'] += 1
            return self._fast_paths.get(op_name)
        return None

    def wrap_for_decode(self, op_name: str, fn: Callable) -> Callable:
        """
        Wrap a function with decode optimizations.

        Args:
            op_name: Operator name
            fn: Original function

        Returns:
            Wrapped function with decode optimizations
        """
        fast_path = self._fast_paths.get(op_name)

        @functools.wraps(fn)
        def decode_optimized(*args, **kwargs):
            self._stats['total_decode_ops'] += 1

            # Check if we should use fast path
            if self.is_decode_phase:
                # Try fast path first
                if fast_path is not None and self.should_use_fast_path(op_name):
                    self._stats['dispatch_bypassed'] += 1
                    return fast_path(*args, **kwargs)

                # Force CUDA for critical ops
                if self.should_force_cuda(op_name):
                    # Ensure we're using CUDA implementation
                    if hasattr(fn, '_cuda_impl'):
                        return fn._cuda_impl(*args, **kwargs)

            # Default path
            return fn(*args, **kwargs)

        decode_optimized._original = fn
        return decode_optimized

    # ==================== CUDA Graph Support ====================

    def capture_cuda_graph(
        self,
        graph_id: str,
        fn: Callable,
        sample_inputs: List[torch.Tensor],
        warmup_runs: int = 3,
    ) -> None:
        """
        Capture a CUDA graph for a function.

        Args:
            graph_id: Unique identifier for the graph
            fn: Function to capture
            sample_inputs: Sample inputs for graph capture
            warmup_runs: Number of warmup runs before capture
        """
        if not self._config.enable_cuda_graph:
            return

        # Warmup
        for _ in range(warmup_runs):
            fn(*sample_inputs)
        torch.cuda.synchronize()

        # Create static tensors for graph
        static_inputs = [t.clone() for t in sample_inputs]
        self._graph_inputs[graph_id] = static_inputs

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = fn(*static_inputs)

        self._cuda_graphs[graph_id] = graph
        self._graph_outputs[graph_id] = [outputs] if not isinstance(outputs, tuple) else list(outputs)

        logger.info(f"Captured CUDA graph: {graph_id}")

    def run_cuda_graph(
        self,
        graph_id: str,
        inputs: List[torch.Tensor],
    ) -> Optional[List[torch.Tensor]]:
        """
        Run a captured CUDA graph.

        Args:
            graph_id: Graph identifier
            inputs: Input tensors (will be copied to static inputs)

        Returns:
            Output tensors, or None if graph not found
        """
        if graph_id not in self._cuda_graphs:
            return None

        # Copy inputs to static buffers
        for static, new in zip(self._graph_inputs[graph_id], inputs):
            static.copy_(new)

        # Replay graph
        self._cuda_graphs[graph_id].replay()
        self._stats['cuda_graph_runs'] += 1

        return self._graph_outputs[graph_id]

    def has_cuda_graph(self, graph_id: str) -> bool:
        """Check if a CUDA graph exists."""
        return graph_id in self._cuda_graphs

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, int]:
        """Get optimizer statistics."""
        return self._stats.copy()

    def print_report(self):
        """Print optimization report."""
        print("\n" + "=" * 60)
        print("DECODE PATH OPTIMIZER REPORT")
        print("=" * 60)
        print(f"\nTotal decode ops: {self._stats['total_decode_ops']}")
        print(f"Fast path calls: {self._stats['fast_path_calls']}")
        print(f"Dispatch bypassed: {self._stats['dispatch_bypassed']}")
        print(f"CUDA graph runs: {self._stats['cuda_graph_runs']}")

        if self._stats['total_decode_ops'] > 0:
            bypass_rate = self._stats['dispatch_bypassed'] / self._stats['total_decode_ops'] * 100
            print(f"\nDispatch bypass rate: {bypass_rate:.1f}%")

        print(f"\nRegistered fast paths: {list(self._fast_paths.keys())}")
        print(f"Captured CUDA graphs: {list(self._cuda_graphs.keys())}")


# Global instance
_decode_optimizer: Optional[DecodePathOptimizer] = None


def get_decode_optimizer() -> DecodePathOptimizer:
    """Get the global decode optimizer."""
    global _decode_optimizer
    if _decode_optimizer is None:
        _decode_optimizer = DecodePathOptimizer()
    return _decode_optimizer


# Convenience functions
def enter_decode_phase(batch_size: int = 1, mode: DecodeMode = DecodeMode.NORMAL):
    """Enter decode phase globally."""
    get_decode_optimizer().enter_decode(batch_size, mode)


def exit_decode_phase():
    """Exit decode phase globally."""
    get_decode_optimizer().exit_decode()


def decode_phase(batch_size: int = 1, mode: DecodeMode = DecodeMode.NORMAL):
    """Context manager for decode phase."""
    return get_decode_optimizer().decode_context(batch_size, mode)


def is_decode_optimized(op_name: str) -> bool:
    """Check if an operator will be decode-optimized."""
    opt = get_decode_optimizer()
    return opt.should_use_fast_path(op_name) or opt.should_force_cuda(op_name)

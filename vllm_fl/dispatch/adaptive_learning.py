# Copyright (c) 2026 BAAI. All rights reserved.

"""
Adaptive Performance Learning for Operator Dispatch.

This module implements runtime performance learning to automatically
determine the optimal backend (CUDA vs FlagGems) for each operator
based on observed performance data.

Key Features:
- Shape-aware performance tracking
- Automatic crossover point detection
- Runtime policy adaptation
- Persistent learning across sessions
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class ShapeStats:
    """Performance statistics for a specific (operator, shape) combination."""
    gems_times: List[float] = field(default_factory=list)
    cuda_times: List[float] = field(default_factory=list)
    gems_total: float = 0.0
    cuda_total: float = 0.0
    last_decision: Optional[str] = None
    decision_confidence: float = 0.0

    @property
    def gems_count(self) -> int:
        return len(self.gems_times)

    @property
    def cuda_count(self) -> int:
        return len(self.cuda_times)

    @property
    def gems_avg(self) -> float:
        return self.gems_total / self.gems_count if self.gems_count > 0 else float('inf')

    @property
    def cuda_avg(self) -> float:
        return self.cuda_total / self.cuda_count if self.cuda_count > 0 else float('inf')

    def add_sample(self, time_ms: float, backend: str, max_samples: int = 100):
        """Add a performance sample."""
        if backend == 'gems':
            self.gems_times.append(time_ms)
            self.gems_total += time_ms
            # Keep only recent samples
            if len(self.gems_times) > max_samples:
                removed = self.gems_times.pop(0)
                self.gems_total -= removed
        else:
            self.cuda_times.append(time_ms)
            self.cuda_total += time_ms
            if len(self.cuda_times) > max_samples:
                removed = self.cuda_times.pop(0)
                self.cuda_total -= removed

    def get_preferred_backend(self, min_samples: int = 20, threshold: float = 0.9) -> Optional[str]:
        """
        Determine preferred backend based on collected data.

        Args:
            min_samples: Minimum samples needed for each backend
            threshold: Ratio threshold to declare a winner (0.9 = 10% faster)

        Returns:
            'gems', 'cuda', or None if insufficient data
        """
        if self.gems_count < min_samples or self.cuda_count < min_samples:
            return None

        gems_avg = self.gems_avg
        cuda_avg = self.cuda_avg

        if gems_avg < cuda_avg * threshold:
            self.last_decision = 'gems'
            self.decision_confidence = cuda_avg / gems_avg
            return 'gems'
        elif cuda_avg < gems_avg * threshold:
            self.last_decision = 'cuda'
            self.decision_confidence = gems_avg / cuda_avg
            return 'cuda'

        # Too close to call - prefer CUDA (lower risk)
        self.last_decision = 'cuda'
        self.decision_confidence = 1.0
        return 'cuda'


class AdaptiveLearner:
    """
    Adaptive performance learner for operator dispatch.

    This class:
    1. Collects performance data for each (operator, shape) combination
    2. Automatically detects crossover points
    3. Updates dispatch policy based on learned data
    4. Persists learning across sessions
    """

    _instance = None
    _lock = threading.Lock()

    # Shape quantization buckets for grouping similar shapes
    BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    SEQ_BUCKETS = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    HIDDEN_BUCKETS = [512, 1024, 2048, 4096, 8192, 16384]

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

        # Shape-specific stats: {op_name: {quantized_shape: ShapeStats}}
        self._stats: Dict[str, Dict[Tuple, ShapeStats]] = defaultdict(dict)

        # Learned policy cache: {op_name: {quantized_shape: 'gems'|'cuda'}}
        self._policy_cache: Dict[str, Dict[Tuple, str]] = defaultdict(dict)

        # Configuration
        self._enabled = os.environ.get("ADAPTIVE_LEARNING", "1") == "1"
        self._min_samples = int(os.environ.get("ADAPTIVE_MIN_SAMPLES", "20"))
        self._threshold = float(os.environ.get("ADAPTIVE_THRESHOLD", "0.9"))
        self._exploration_rate = float(os.environ.get("ADAPTIVE_EXPLORATION", "0.1"))

        # Persistence
        self._cache_dir = Path(os.environ.get(
            "ADAPTIVE_CACHE_DIR",
            os.path.expanduser("~/.cache/vllm_fl/adaptive")
        ))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Lock for thread safety
        self._data_lock = threading.Lock()

        # Load persisted data
        self._load_cache()

        logger.info(f"AdaptiveLearner initialized (enabled={self._enabled})")

    @staticmethod
    def _quantize_to_bucket(value: int, buckets: List[int]) -> int:
        """Quantize a value to the nearest bucket."""
        for bucket in buckets:
            if value <= bucket:
                return bucket
        return buckets[-1]

    def _quantize_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Quantize shape to reduce cardinality."""
        if len(shape) == 1:
            return (self._quantize_to_bucket(shape[0], self.HIDDEN_BUCKETS),)
        elif len(shape) == 2:
            return (
                self._quantize_to_bucket(shape[0], self.BATCH_BUCKETS),
                self._quantize_to_bucket(shape[1], self.HIDDEN_BUCKETS),
            )
        elif len(shape) >= 3:
            return (
                self._quantize_to_bucket(shape[0], self.BATCH_BUCKETS),
                self._quantize_to_bucket(shape[1], self.SEQ_BUCKETS),
                self._quantize_to_bucket(shape[-1], self.HIDDEN_BUCKETS),
            )
        return shape

    def record(
        self,
        op_name: str,
        shape: Tuple[int, ...],
        duration_ms: float,
        backend: str,
    ) -> None:
        """
        Record a performance sample.

        Args:
            op_name: Operator name
            shape: Input tensor shape
            duration_ms: Execution time in milliseconds
            backend: 'gems' or 'cuda'
        """
        if not self._enabled:
            return

        q_shape = self._quantize_shape(shape)

        with self._data_lock:
            if q_shape not in self._stats[op_name]:
                self._stats[op_name][q_shape] = ShapeStats()

            self._stats[op_name][q_shape].add_sample(duration_ms, backend)

            # Check if we should update policy
            stats = self._stats[op_name][q_shape]
            preferred = stats.get_preferred_backend(self._min_samples, self._threshold)
            if preferred:
                self._policy_cache[op_name][q_shape] = preferred

    def get_preferred_backend(
        self,
        op_name: str,
        shape: Tuple[int, ...],
    ) -> Optional[str]:
        """
        Get the preferred backend for an operator with given shape.

        Returns:
            'gems', 'cuda', or None if unknown
        """
        if not self._enabled:
            return None

        q_shape = self._quantize_shape(shape)

        # Check cache first
        if op_name in self._policy_cache:
            if q_shape in self._policy_cache[op_name]:
                return self._policy_cache[op_name][q_shape]

        return None

    def should_explore(self, op_name: str, shape: Tuple[int, ...]) -> bool:
        """
        Determine if we should explore alternative backend.

        Uses epsilon-greedy exploration to occasionally try
        the non-preferred backend to gather more data.
        """
        import random

        if not self._enabled:
            return False

        q_shape = self._quantize_shape(shape)

        # Always explore if we don't have enough data
        if op_name not in self._stats or q_shape not in self._stats[op_name]:
            return True

        stats = self._stats[op_name][q_shape]
        if stats.gems_count < self._min_samples or stats.cuda_count < self._min_samples:
            return True

        # Epsilon-greedy exploration
        return random.random() < self._exploration_rate

    def get_exploration_backend(self, op_name: str, shape: Tuple[int, ...]) -> str:
        """
        Get the backend to use for exploration.

        Returns the backend with fewer samples.
        """
        q_shape = self._quantize_shape(shape)

        if op_name not in self._stats or q_shape not in self._stats[op_name]:
            # No data - explore CUDA first (safer)
            return 'cuda'

        stats = self._stats[op_name][q_shape]

        # Explore the one with fewer samples
        if stats.gems_count < stats.cuda_count:
            return 'gems'
        return 'cuda'

    def _load_cache(self) -> None:
        """Load persisted learning data."""
        cache_file = self._cache_dir / "adaptive_policy.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                for op_name, shapes in data.items():
                    for shape_str, backend in shapes.items():
                        # Convert string back to tuple
                        shape = tuple(map(int, shape_str.strip('()').split(',')))
                        self._policy_cache[op_name][shape] = backend

                logger.info(f"Loaded {sum(len(s) for s in self._policy_cache.values())} cached policies")
            except Exception as e:
                logger.warning(f"Failed to load adaptive cache: {e}")

    def save_cache(self) -> None:
        """Persist learned policies to disk."""
        cache_file = self._cache_dir / "adaptive_policy.json"

        # Convert tuples to strings for JSON
        data = {}
        for op_name, shapes in self._policy_cache.items():
            data[op_name] = {str(shape): backend for shape, backend in shapes.items()}

        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {sum(len(s) for s in self._policy_cache.values())} policies")
        except Exception as e:
            logger.warning(f"Failed to save adaptive cache: {e}")

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of collected statistics."""
        summary = {
            'operators': {},
            'total_samples': 0,
            'learned_policies': 0,
        }

        for op_name, shapes in self._stats.items():
            op_summary = {
                'shapes': len(shapes),
                'total_gems_samples': sum(s.gems_count for s in shapes.values()),
                'total_cuda_samples': sum(s.cuda_count for s in shapes.values()),
                'learned': len(self._policy_cache.get(op_name, {})),
            }
            summary['operators'][op_name] = op_summary
            summary['total_samples'] += op_summary['total_gems_samples'] + op_summary['total_cuda_samples']

        summary['learned_policies'] = sum(len(s) for s in self._policy_cache.values())
        return summary

    def print_report(self) -> None:
        """Print detailed learning report."""
        print("\n" + "=" * 80)
        print("ADAPTIVE LEARNING REPORT")
        print("=" * 80)

        summary = self.get_stats_summary()
        print(f"\nTotal samples collected: {summary['total_samples']}")
        print(f"Learned policies: {summary['learned_policies']}")

        print(f"\n{'Operator':<25} {'Shapes':<10} {'GEMs':<10} {'CUDA':<10} {'Learned':<10}")
        print("-" * 65)

        for op_name, op_stats in summary['operators'].items():
            print(f"{op_name:<25} {op_stats['shapes']:<10} "
                  f"{op_stats['total_gems_samples']:<10} "
                  f"{op_stats['total_cuda_samples']:<10} "
                  f"{op_stats['learned']:<10}")

        # Print learned policies
        if self._policy_cache:
            print("\n" + "-" * 65)
            print("LEARNED POLICIES:")
            print("-" * 65)
            for op_name, shapes in self._policy_cache.items():
                print(f"\n{op_name}:")
                for shape, backend in list(shapes.items())[:5]:  # Show first 5
                    stats = self._stats.get(op_name, {}).get(shape)
                    if stats:
                        speedup = stats.decision_confidence
                        print(f"  {shape} -> {backend} (speedup: {speedup:.2f}x)")
                    else:
                        print(f"  {shape} -> {backend}")
                if len(shapes) > 5:
                    print(f"  ... and {len(shapes) - 5} more")


# Global instance
_adaptive_learner: Optional[AdaptiveLearner] = None


def get_adaptive_learner() -> AdaptiveLearner:
    """Get the global adaptive learner instance."""
    global _adaptive_learner
    if _adaptive_learner is None:
        _adaptive_learner = AdaptiveLearner()
    return _adaptive_learner


def record_performance(
    op_name: str,
    tensor: torch.Tensor,
    duration_ms: float,
    backend: str,
) -> None:
    """Convenience function to record performance."""
    get_adaptive_learner().record(op_name, tuple(tensor.shape), duration_ms, backend)


def get_optimal_backend(op_name: str, tensor: torch.Tensor) -> Optional[str]:
    """Convenience function to get optimal backend."""
    return get_adaptive_learner().get_preferred_backend(op_name, tuple(tensor.shape))

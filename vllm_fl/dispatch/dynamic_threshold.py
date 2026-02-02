# Copyright (c) 2026 BAAI. All rights reserved.

"""
Dynamic Threshold Adjustment for Operator Dispatch.

This module provides dynamic threshold management for deciding when to use
FlagGems vs CUDA kernels based on tensor shapes and runtime conditions.

Key Features:
1. Shape-aware crossover point detection
2. Dynamic threshold adjustment based on observed performance
3. Prefill vs decode phase-specific thresholds
4. Hardware-aware threshold initialization
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class CrossoverPoint:
    """
    Crossover point where FlagGems becomes faster than CUDA.

    Below this threshold, CUDA is faster.
    Above this threshold, FlagGems may be faster.
    """
    # Minimum total tokens (batch * seq_len) for FlagGems
    min_tokens: int = 2048

    # Minimum batch size
    min_batch: int = 4

    # Minimum sequence length
    min_seq_len: int = 128

    # Minimum tensor elements
    min_elements: int = 100_000

    # Confidence (0-1) based on observed data
    confidence: float = 0.5

    # Whether this is learned or default
    is_learned: bool = False


@dataclass
class OperatorThresholds:
    """Thresholds for a specific operator."""
    op_name: str

    # Prefill phase crossover
    prefill: CrossoverPoint = field(default_factory=CrossoverPoint)

    # Decode phase - typically never use FlagGems
    decode: CrossoverPoint = field(default_factory=lambda: CrossoverPoint(
        min_tokens=1_000_000,  # Effectively never
        min_batch=1000,
        min_seq_len=10000,
        min_elements=10_000_000,
        confidence=1.0,
        is_learned=False,
    ))


# Default thresholds based on benchmarks
DEFAULT_THRESHOLDS: Dict[str, OperatorThresholds] = {
    'silu_and_mul': OperatorThresholds(
        op_name='silu_and_mul',
        prefill=CrossoverPoint(
            min_tokens=8192,    # FlagGems never faster in our tests
            min_batch=32,
            min_seq_len=256,
            min_elements=500_000,
            confidence=0.9,
            is_learned=False,
        ),
    ),
    'rmsnorm': OperatorThresholds(
        op_name='rmsnorm',
        prefill=CrossoverPoint(
            min_tokens=8192,    # FlagGems matches at ~8K tokens
            min_batch=16,
            min_seq_len=512,
            min_elements=300_000,
            confidence=0.8,
            is_learned=False,
        ),
    ),
    'layer_norm': OperatorThresholds(
        op_name='layer_norm',
        prefill=CrossoverPoint(
            min_tokens=2048,    # FlagGems competitive earlier
            min_batch=8,
            min_seq_len=256,
            min_elements=200_000,
            confidence=0.7,
            is_learned=False,
        ),
    ),
    'mm': OperatorThresholds(
        op_name='mm',
        prefill=CrossoverPoint(
            min_tokens=4096,
            min_batch=8,
            min_seq_len=512,
            min_elements=500_000,
            confidence=0.6,
            is_learned=False,
        ),
    ),
    'gelu': OperatorThresholds(
        op_name='gelu',
        prefill=CrossoverPoint(
            min_tokens=1024,    # Simple elementwise - FlagGems OK
            min_batch=4,
            min_seq_len=128,
            min_elements=100_000,
            confidence=0.7,
            is_learned=False,
        ),
    ),
}


class DynamicThresholdManager:
    """
    Manages dynamic thresholds for operator dispatch.

    This class:
    1. Provides default thresholds based on benchmarks
    2. Adjusts thresholds based on runtime performance
    3. Handles phase-specific (prefill vs decode) thresholds
    4. Supports hardware-specific initialization
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

        # Current thresholds
        self._thresholds: Dict[str, OperatorThresholds] = DEFAULT_THRESHOLDS.copy()

        # Hardware info for threshold adjustment
        self._gpu_name = ""
        self._compute_capability = (0, 0)
        self._memory_gb = 0

        # Learning state
        self._learning_enabled = os.environ.get("DYNAMIC_THRESHOLD_LEARNING", "1") == "1"
        self._update_history: Dict[str, List[Tuple[CrossoverPoint, float]]] = {}

        # Initialize hardware info
        self._init_hardware_info()

        # Adjust thresholds based on hardware
        self._adjust_for_hardware()

        logger.info(f"DynamicThresholdManager initialized for {self._gpu_name}")

    def _init_hardware_info(self):
        """Initialize hardware information."""
        if torch.cuda.is_available():
            self._gpu_name = torch.cuda.get_device_name(0)
            self._compute_capability = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            self._memory_gb = props.total_memory / (1024 ** 3)

    def _adjust_for_hardware(self):
        """Adjust thresholds based on hardware characteristics."""
        # A100 (sm_80) - well optimized
        if self._compute_capability >= (8, 0):
            # A100 has excellent CUDA kernels, raise thresholds
            self._scale_thresholds(1.2)

        # H100 (sm_90) - even better CUDA
        if self._compute_capability >= (9, 0):
            self._scale_thresholds(1.5)

        # Older hardware - lower thresholds (FlagGems more competitive)
        if self._compute_capability < (7, 5):
            self._scale_thresholds(0.7)

    def _scale_thresholds(self, factor: float):
        """Scale all thresholds by a factor."""
        for op_thresholds in self._thresholds.values():
            op_thresholds.prefill.min_tokens = int(op_thresholds.prefill.min_tokens * factor)
            op_thresholds.prefill.min_elements = int(op_thresholds.prefill.min_elements * factor)

    def get_threshold(self, op_name: str) -> OperatorThresholds:
        """Get thresholds for an operator."""
        if op_name in self._thresholds:
            return self._thresholds[op_name]

        # Return conservative default for unknown ops
        return OperatorThresholds(
            op_name=op_name,
            prefill=CrossoverPoint(
                min_tokens=4096,
                min_batch=8,
                min_seq_len=256,
                min_elements=200_000,
                confidence=0.3,
                is_learned=False,
            ),
        )

    def should_use_flaggems(
        self,
        op_name: str,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        is_decode: bool = False,
    ) -> bool:
        """
        Determine if FlagGems should be used based on thresholds.

        Args:
            op_name: Operator name
            batch_size: Current batch size
            seq_len: Current sequence length
            hidden_size: Hidden dimension size
            is_decode: Whether in decode phase

        Returns:
            True if FlagGems should be used
        """
        thresholds = self.get_threshold(op_name)
        crossover = thresholds.decode if is_decode else thresholds.prefill

        # Check all conditions
        total_tokens = batch_size * seq_len
        total_elements = total_tokens * hidden_size

        if total_tokens < crossover.min_tokens:
            return False
        if batch_size < crossover.min_batch:
            return False
        if seq_len < crossover.min_seq_len:
            return False
        if total_elements < crossover.min_elements:
            return False

        return True

    def update_threshold(
        self,
        op_name: str,
        new_crossover: CrossoverPoint,
        phase: str = 'prefill',
        performance_gain: float = 0.0,
    ):
        """
        Update threshold based on observed performance.

        Args:
            op_name: Operator name
            new_crossover: New crossover point
            phase: 'prefill' or 'decode'
            performance_gain: Observed speedup (>1 means FlagGems faster)
        """
        if not self._learning_enabled:
            return

        if op_name not in self._thresholds:
            self._thresholds[op_name] = OperatorThresholds(op_name=op_name)

        thresholds = self._thresholds[op_name]
        current = thresholds.prefill if phase == 'prefill' else thresholds.decode

        # Track history
        if op_name not in self._update_history:
            self._update_history[op_name] = []
        self._update_history[op_name].append((new_crossover, performance_gain))

        # Only update if we have enough data and new point is better
        history = self._update_history[op_name]
        if len(history) >= 5:
            # Average the recent crossover points
            avg_tokens = int(sum(h[0].min_tokens for h in history[-5:]) / 5)
            avg_batch = int(sum(h[0].min_batch for h in history[-5:]) / 5)
            avg_seq = int(sum(h[0].min_seq_len for h in history[-5:]) / 5)
            avg_elements = int(sum(h[0].min_elements for h in history[-5:]) / 5)
            avg_gain = sum(h[1] for h in history[-5:]) / 5

            # Update if performance gain is significant
            if avg_gain > 1.1:  # FlagGems 10% faster
                new_point = CrossoverPoint(
                    min_tokens=avg_tokens,
                    min_batch=avg_batch,
                    min_seq_len=avg_seq,
                    min_elements=avg_elements,
                    confidence=min(0.9, current.confidence + 0.1),
                    is_learned=True,
                )

                if phase == 'prefill':
                    thresholds.prefill = new_point
                else:
                    thresholds.decode = new_point

                logger.info(f"Updated {op_name} {phase} threshold: {new_point}")

    def get_recommendation(
        self,
        op_name: str,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        is_decode: bool = False,
    ) -> Dict[str, Any]:
        """
        Get detailed recommendation with reasoning.

        Returns dict with:
        - use_flaggems: bool
        - reason: str
        - confidence: float
        - thresholds: current thresholds
        """
        thresholds = self.get_threshold(op_name)
        crossover = thresholds.decode if is_decode else thresholds.prefill

        total_tokens = batch_size * seq_len
        total_elements = total_tokens * hidden_size

        use_flaggems = self.should_use_flaggems(
            op_name, batch_size, seq_len, hidden_size, is_decode
        )

        # Build reason
        reasons = []
        if total_tokens < crossover.min_tokens:
            reasons.append(f"tokens ({total_tokens}) < threshold ({crossover.min_tokens})")
        if batch_size < crossover.min_batch:
            reasons.append(f"batch ({batch_size}) < threshold ({crossover.min_batch})")
        if seq_len < crossover.min_seq_len:
            reasons.append(f"seq_len ({seq_len}) < threshold ({crossover.min_seq_len})")
        if total_elements < crossover.min_elements:
            reasons.append(f"elements ({total_elements}) < threshold ({crossover.min_elements})")

        if use_flaggems:
            reason = "All thresholds exceeded - FlagGems may be beneficial"
        else:
            reason = "Below thresholds: " + ", ".join(reasons) if reasons else "Using CUDA"

        return {
            'use_flaggems': use_flaggems,
            'reason': reason,
            'confidence': crossover.confidence,
            'is_learned': crossover.is_learned,
            'phase': 'decode' if is_decode else 'prefill',
            'thresholds': {
                'min_tokens': crossover.min_tokens,
                'min_batch': crossover.min_batch,
                'min_seq_len': crossover.min_seq_len,
                'min_elements': crossover.min_elements,
            },
            'current_values': {
                'total_tokens': total_tokens,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'total_elements': total_elements,
            },
        }

    def print_thresholds(self):
        """Print current thresholds."""
        print("\n" + "=" * 80)
        print("DYNAMIC THRESHOLDS")
        print("=" * 80)
        print(f"\nHardware: {self._gpu_name} (SM {self._compute_capability[0]}.{self._compute_capability[1]})")
        print(f"Memory: {self._memory_gb:.1f} GB")
        print(f"Learning: {'enabled' if self._learning_enabled else 'disabled'}")

        print(f"\n{'Operator':<20} {'Phase':<10} {'Tokens':<10} {'Batch':<8} {'SeqLen':<8} {'Conf':<8} {'Learned':<8}")
        print("-" * 80)

        for op_name, thresholds in sorted(self._thresholds.items()):
            # Prefill
            p = thresholds.prefill
            learned = "Yes" if p.is_learned else "No"
            print(f"{op_name:<20} {'prefill':<10} {p.min_tokens:<10} {p.min_batch:<8} {p.min_seq_len:<8} {p.confidence:<8.2f} {learned:<8}")

            # Decode (if different from default "never")
            d = thresholds.decode
            if d.min_tokens < 1_000_000:
                learned = "Yes" if d.is_learned else "No"
                print(f"{'':<20} {'decode':<10} {d.min_tokens:<10} {d.min_batch:<8} {d.min_seq_len:<8} {d.confidence:<8.2f} {learned:<8}")


# Global instance
_threshold_manager: Optional[DynamicThresholdManager] = None


def get_threshold_manager() -> DynamicThresholdManager:
    """Get the global threshold manager."""
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = DynamicThresholdManager()
    return _threshold_manager


# Convenience functions
def should_use_flaggems(
    op_name: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    is_decode: bool = False,
) -> bool:
    """Check if FlagGems should be used for given parameters."""
    return get_threshold_manager().should_use_flaggems(
        op_name, batch_size, seq_len, hidden_size, is_decode
    )


def get_dispatch_recommendation(
    op_name: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    is_decode: bool = False,
) -> Dict[str, Any]:
    """Get detailed dispatch recommendation."""
    return get_threshold_manager().get_recommendation(
        op_name, batch_size, seq_len, hidden_size, is_decode
    )

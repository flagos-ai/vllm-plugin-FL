# Copyright (c) 2026 BAAI. All rights reserved.

"""
Tiered Operator Policy for FlagGems Integration.

This module defines which operators should be replaced by FlagGems and under what conditions.
The policy is based on empirical performance analysis and inference workload characteristics.

Tier Definitions:
-----------------
- Tier-0 (NEVER_REPLACE): Critical operators where FlagGems causes catastrophic regression
- Tier-1 (CONDITIONAL): Replace only if GEMs kernel is proven faster for specific conditions
- Tier-2 (SAFE): Generally safe to replace with moderate overhead
- Tier-3 (EXPERIMENTAL): Opt-in experimental replacements

Usage Context:
--------------
- PREFILL: Long sequences, compute-bound, more tolerant to overhead
- DECODE: Single token, latency-critical, extremely sensitive to overhead
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any

logger = logging.getLogger(__name__)


class OperatorTier(Enum):
    """Operator replacement tier classification."""
    TIER_0_NEVER = 0      # Never replace - catastrophic regression
    TIER_1_CONDITIONAL = 1  # Replace only under specific conditions
    TIER_2_SAFE = 2       # Safe to replace
    TIER_3_EXPERIMENTAL = 3  # Experimental - opt-in only


class InferencePhase(Enum):
    """Inference phase for context-aware dispatch."""
    PREFILL = auto()
    DECODE = auto()
    MIXED = auto()  # Continuous batching with both


@dataclass
class OperatorCondition:
    """Conditions under which an operator can be replaced."""
    min_batch_size: int = 1
    min_seq_len: int = 1
    max_seq_len: int = 1_000_000
    allowed_phases: Set[InferencePhase] = field(default_factory=lambda: {InferencePhase.PREFILL, InferencePhase.DECODE})
    min_tensor_elements: int = 0
    custom_predicate: Optional[Callable[[Any], bool]] = None

    def is_satisfied(
        self,
        batch_size: int = 1,
        seq_len: int = 1,
        phase: InferencePhase = InferencePhase.MIXED,
        tensor_elements: int = 0,
        **kwargs
    ) -> bool:
        """Check if conditions are satisfied for replacement."""
        if batch_size < self.min_batch_size:
            return False
        if seq_len < self.min_seq_len or seq_len > self.max_seq_len:
            return False
        if phase not in self.allowed_phases and phase != InferencePhase.MIXED:
            return False
        if tensor_elements < self.min_tensor_elements:
            return False
        if self.custom_predicate and not self.custom_predicate(kwargs):
            return False
        return True


@dataclass
class OperatorPolicy:
    """Policy for a specific operator."""
    op_name: str
    tier: OperatorTier
    reason: str
    condition: Optional[OperatorCondition] = None
    alternative: Optional[str] = None  # Preferred alternative implementation

    def should_replace(self, **context) -> bool:
        """Determine if this operator should be replaced given context."""
        if self.tier == OperatorTier.TIER_0_NEVER:
            return False
        if self.tier == OperatorTier.TIER_3_EXPERIMENTAL:
            # Only replace if explicitly enabled
            if not os.environ.get("GEMS_EXPERIMENTAL", "").lower() == "true":
                return False
        if self.condition:
            return self.condition.is_satisfied(**context)
        return self.tier in (OperatorTier.TIER_2_SAFE, OperatorTier.TIER_3_EXPERIMENTAL)


# =============================================================================
# TIER-0: NEVER REPLACE (Catastrophic Regression)
# =============================================================================

TIER_0_OPERATORS = {
    # Attention operators - FlashAttention is 10-50x faster
    "_flash_attention_forward": OperatorPolicy(
        op_name="_flash_attention_forward",
        tier=OperatorTier.TIER_0_NEVER,
        reason="FlashAttention fused kernel is 10-50x faster than unfused ops",
        alternative="vllm.attention.backends.flash_attn"
    ),
    "_flash_attention_backward": OperatorPolicy(
        op_name="_flash_attention_backward",
        tier=OperatorTier.TIER_0_NEVER,
        reason="FlashAttention backward is critical for training",
    ),

    # Scaled dot product attention variants
    "_scaled_dot_product_attention": OperatorPolicy(
        op_name="_scaled_dot_product_attention",
        tier=OperatorTier.TIER_0_NEVER,
        reason="SDPA dispatches to optimized backends (Flash, Memory-efficient)",
    ),
    "_scaled_dot_product_flash_attention": OperatorPolicy(
        op_name="_scaled_dot_product_flash_attention",
        tier=OperatorTier.TIER_0_NEVER,
        reason="Direct FlashAttention call",
    ),

    # Matrix multiplies in attention path - cuBLAS is heavily optimized
    "bmm": OperatorPolicy(
        op_name="bmm",
        tier=OperatorTier.TIER_0_NEVER,
        reason="BMM in attention path uses cuBLAS with Tensor Cores; 2-5x faster",
        alternative="torch.bmm (cuBLAS)"
    ),

    # Softmax in attention - often fused with attention
    "_softmax": OperatorPolicy(
        op_name="_softmax",
        tier=OperatorTier.TIER_0_NEVER,
        reason="Softmax is fused in FlashAttention; standalone replacement breaks fusion",
    ),
    "_log_softmax": OperatorPolicy(
        op_name="_log_softmax",
        tier=OperatorTier.TIER_0_NEVER,
        reason="Log-softmax often fused for numerical stability",
    ),
}


# =============================================================================
# TIER-1: CONDITIONAL REPLACEMENT
# =============================================================================

# Condition: Only replace in prefill phase with large tensors
PREFILL_ONLY_LARGE = OperatorCondition(
    min_batch_size=4,
    min_seq_len=128,
    allowed_phases={InferencePhase.PREFILL},
    min_tensor_elements=100_000,
)

# Condition: Only for large batch operations
LARGE_BATCH_ONLY = OperatorCondition(
    min_batch_size=8,
    min_tensor_elements=50_000,
)

TIER_1_OPERATORS = {
    # Matrix multiply - cuBLAS is optimized but FlagGems may win for specific shapes
    "mm": OperatorPolicy(
        op_name="mm",
        tier=OperatorTier.TIER_1_CONDITIONAL,
        reason="cuBLAS is highly optimized; FlagGems may win for non-standard shapes",
        condition=PREFILL_ONLY_LARGE,
    ),
    "addmm": OperatorPolicy(
        op_name="addmm",
        tier=OperatorTier.TIER_1_CONDITIONAL,
        reason="Fused add+mm; depends on shape",
        condition=PREFILL_ONLY_LARGE,
    ),

    # Layer normalization - vLLM has optimized version
    "native_layer_norm": OperatorPolicy(
        op_name="native_layer_norm",
        tier=OperatorTier.TIER_1_CONDITIONAL,
        reason="vLLM has fused layer norm; FlagGems may be slower",
        condition=LARGE_BATCH_ONLY,
    ),

    # Convolutions - cuDNN is heavily optimized
    "conv2d": OperatorPolicy(
        op_name="conv2d",
        tier=OperatorTier.TIER_1_CONDITIONAL,
        reason="cuDNN convolution is highly optimized",
        condition=LARGE_BATCH_ONLY,
    ),
}


# =============================================================================
# TIER-2: SAFE TO REPLACE
# =============================================================================

TIER_2_OPERATORS = {
    # Elementwise activations - relatively simple, FlagGems competitive
    "silu": OperatorPolicy(
        op_name="silu",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple elementwise; FlagGems is competitive",
    ),
    "silu_": OperatorPolicy(
        op_name="silu_",
        tier=OperatorTier.TIER_2_SAFE,
        reason="In-place SiLU; FlagGems is competitive",
    ),
    "gelu": OperatorPolicy(
        op_name="gelu",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple elementwise; FlagGems is competitive",
    ),
    "relu": OperatorPolicy(
        op_name="relu",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple elementwise; FlagGems is competitive",
    ),
    "sigmoid": OperatorPolicy(
        op_name="sigmoid",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple elementwise; FlagGems is competitive",
    ),
    "tanh": OperatorPolicy(
        op_name="tanh",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple elementwise; FlagGems is competitive",
    ),

    # Basic elementwise math
    "add.Tensor": OperatorPolicy(
        op_name="add.Tensor",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple elementwise; FlagGems is competitive",
    ),
    "mul.Tensor": OperatorPolicy(
        op_name="mul.Tensor",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple elementwise; FlagGems is competitive",
    ),
    "div.Tensor": OperatorPolicy(
        op_name="div.Tensor",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple elementwise; FlagGems is competitive",
    ),
    "sub.Tensor": OperatorPolicy(
        op_name="sub.Tensor",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple elementwise; FlagGems is competitive",
    ),

    # Unary math operations
    "exp": OperatorPolicy(
        op_name="exp",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple unary; FlagGems is competitive",
    ),
    "log": OperatorPolicy(
        op_name="log",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple unary; FlagGems is competitive",
    ),
    "sqrt": OperatorPolicy(
        op_name="sqrt",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple unary; FlagGems is competitive",
    ),
    "rsqrt": OperatorPolicy(
        op_name="rsqrt",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple unary; FlagGems is competitive",
    ),
    "neg": OperatorPolicy(
        op_name="neg",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple unary; FlagGems is competitive",
    ),
    "abs": OperatorPolicy(
        op_name="abs",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Simple unary; FlagGems is competitive",
    ),

    # Reductions (generally safe but watch for small tensors)
    "sum": OperatorPolicy(
        op_name="sum",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Reduction; FlagGems is competitive for large tensors",
    ),
    "mean": OperatorPolicy(
        op_name="mean",
        tier=OperatorTier.TIER_2_SAFE,
        reason="Reduction; FlagGems is competitive for large tensors",
    ),
}


# =============================================================================
# TIER-3: EXPERIMENTAL (Opt-in only)
# =============================================================================

TIER_3_OPERATORS = {
    # These require GEMS_EXPERIMENTAL=true to enable
    "baddbmm": OperatorPolicy(
        op_name="baddbmm",
        tier=OperatorTier.TIER_3_EXPERIMENTAL,
        reason="Complex batched operation; needs validation",
    ),
    "addr": OperatorPolicy(
        op_name="addr",
        tier=OperatorTier.TIER_3_EXPERIMENTAL,
        reason="Outer product; needs validation",
    ),
}


# =============================================================================
# Policy Manager
# =============================================================================

class OperatorPolicyManager:
    """
    Manages operator replacement policies for FlagGems integration.

    This class provides a centralized way to query whether an operator
    should be replaced by FlagGems given the current context.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Combine all policies
        self._policies: Dict[str, OperatorPolicy] = {}
        self._policies.update(TIER_0_OPERATORS)
        self._policies.update(TIER_1_OPERATORS)
        self._policies.update(TIER_2_OPERATORS)
        self._policies.update(TIER_3_OPERATORS)

        # Runtime context
        self._current_phase = InferencePhase.MIXED
        self._batch_size = 1
        self._seq_len = 1

        # Statistics
        self._replacement_stats: Dict[str, Dict[str, int]] = {
            "replaced": {},
            "skipped": {},
        }

        logger.info(f"OperatorPolicyManager initialized with {len(self._policies)} policies")

    def set_context(
        self,
        phase: InferencePhase = None,
        batch_size: int = None,
        seq_len: int = None,
    ):
        """Update runtime context for dispatch decisions."""
        if phase is not None:
            self._current_phase = phase
        if batch_size is not None:
            self._batch_size = batch_size
        if seq_len is not None:
            self._seq_len = seq_len

    def get_policy(self, op_name: str) -> Optional[OperatorPolicy]:
        """Get policy for a specific operator."""
        return self._policies.get(op_name)

    def should_replace(self, op_name: str, tensor_elements: int = 0, **kwargs) -> bool:
        """
        Determine if an operator should be replaced by FlagGems.

        Args:
            op_name: Name of the aten operator
            tensor_elements: Number of elements in primary input tensor
            **kwargs: Additional context for custom predicates

        Returns:
            True if operator should be replaced, False otherwise
        """
        policy = self._policies.get(op_name)

        if policy is None:
            # Unknown operator - default to NOT replacing (conservative)
            self._record_stat("skipped", op_name)
            return False

        context = {
            "batch_size": self._batch_size,
            "seq_len": self._seq_len,
            "phase": self._current_phase,
            "tensor_elements": tensor_elements,
            **kwargs
        }

        should = policy.should_replace(**context)

        if should:
            self._record_stat("replaced", op_name)
        else:
            self._record_stat("skipped", op_name)

        return should

    def _record_stat(self, category: str, op_name: str):
        """Record replacement statistics."""
        if op_name not in self._replacement_stats[category]:
            self._replacement_stats[category][op_name] = 0
        self._replacement_stats[category][op_name] += 1

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get replacement statistics."""
        return self._replacement_stats

    def print_summary(self):
        """Print policy summary."""
        tier_counts = {tier: 0 for tier in OperatorTier}
        for policy in self._policies.values():
            tier_counts[policy.tier] += 1

        print("\n" + "="*60)
        print("OPERATOR POLICY SUMMARY")
        print("="*60)
        print(f"\nTier-0 (NEVER REPLACE):     {tier_counts[OperatorTier.TIER_0_NEVER]}")
        print(f"Tier-1 (CONDITIONAL):       {tier_counts[OperatorTier.TIER_1_CONDITIONAL]}")
        print(f"Tier-2 (SAFE):              {tier_counts[OperatorTier.TIER_2_SAFE]}")
        print(f"Tier-3 (EXPERIMENTAL):      {tier_counts[OperatorTier.TIER_3_EXPERIMENTAL]}")
        print(f"\nTotal policies: {len(self._policies)}")

        if self._replacement_stats["replaced"]:
            print(f"\nReplaced operators: {len(self._replacement_stats['replaced'])}")
        if self._replacement_stats["skipped"]:
            print(f"Skipped operators: {len(self._replacement_stats['skipped'])}")


# Global instance
_policy_manager: Optional[OperatorPolicyManager] = None


def get_policy_manager() -> OperatorPolicyManager:
    """Get the global operator policy manager."""
    global _policy_manager
    if _policy_manager is None:
        _policy_manager = OperatorPolicyManager()
    return _policy_manager


def should_replace_operator(op_name: str, **context) -> bool:
    """Convenience function to check if an operator should be replaced."""
    return get_policy_manager().should_replace(op_name, **context)


def set_inference_phase(phase: InferencePhase):
    """Set the current inference phase for context-aware dispatch."""
    get_policy_manager().set_context(phase=phase)


def set_batch_context(batch_size: int, seq_len: int):
    """Set batch context for context-aware dispatch."""
    get_policy_manager().set_context(batch_size=batch_size, seq_len=seq_len)

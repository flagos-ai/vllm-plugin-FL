"""Compatibility import for the FL-local current Ascend GDN implementation."""

from vllm_fl.dispatch.backends.vendor.ascend.impl.triton.fla.chunk import (
    ChunkGatedDeltaRuleFunction,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd,
)

__all__ = [
    "ChunkGatedDeltaRuleFunction",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
]

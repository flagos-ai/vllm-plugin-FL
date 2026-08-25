# Copyright (c) 2026 BAAI. All rights reserved.
#
# Break-graph support for vllm-plugin-FL.
#
# vLLM 0.24.0 introduced ``breakable_cudagraph`` — a mode where a single
# forward-pass stream capture is split at attention / kv-cache boundaries
# into alternating graph-segments and eager-segments, avoiding the need for
# torch.compile FX-graph splitting.
#
# The vLLM 0.28.0 target retains this interface, so this module is a thin
# re-export of the upstream symbols. No FL-specific fallback is maintained.
#
# Key upstream design:
#   - ``unified_attention_with_output`` is already decorated with
#     ``@eager_break_during_capture`` in vLLM 0.28.0. OOT attention backends
#     (e.g. AttentionFLBackend) implement ``AttentionImpl.forward`` which is
#     called from within ``unified_attention_with_output``.  This means OOT
#     backends automatically participate in breakable capture without any
#     additional wrapping at the dispatch/registry level.
#   - The FL dispatch op ``attention_backend`` is an init-time selector
#     (returns a class path string) and is NOT invoked during CUDA graph
#     capture.  Wrapping it would have no effect on capture behavior.

from __future__ import annotations

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    BreakableCUDAGraphWrapper,
    eager_break_during_capture,
    is_breakable_cudagraph_enabled,
)

__all__ = [
    "is_breakable_cudagraph_enabled",
    "eager_break_during_capture",
    "BreakableCUDAGraphCapture",
    "BreakableCUDAGraphWrapper",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.

"""Keep the independent deep_gemm implementation, remove framework-only indirection."""

import functools
import os
import torch


@functools.lru_cache(maxsize=1)
def _backend():
    os.environ.setdefault(
        "DG_JIT_CACHE_DIR",
        os.path.join(
            os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm")),
            "deep_gemm",
        ),
    )
    import deep_gemm

    return deep_gemm.bf16_mqa_logits


def bf16_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    if q.shape[0] == 0 or kv.shape[0] == 0:
        return torch.empty(
            (q.shape[0], kv.shape[0]), device=q.device, dtype=torch.float32
        )
    return _backend()(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Pinned DeepGEMM entry points and launch-only MetaX eager compatibility."""

import functools
import os

import torch
import triton

from ..attention.launch import local_launch

_selected_backend = None


def _eager_logits(kernel, q, kv, weights, ks, ke):
    rows, heads, dim = q.shape
    columns = kv.shape[0]
    output = torch.empty((rows, columns), device=q.device, dtype=torch.float32)
    kernel[(triton.cdiv(rows, 64), triton.cdiv(columns, 64))](
        q,
        kv,
        weights,
        ks,
        ke,
        output,
        rows,
        columns,
        heads,
        dim,
        *q.stride(),
        *kv.stride(),
        *weights.stride(),
        *output.stride(),
        True,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=4,
    )
    return output


@functools.lru_cache(maxsize=2)
def _backend(enforce_eager=False):
    os.environ.setdefault(
        "DG_JIT_CACHE_DIR",
        os.path.join(
            os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm")),
            "deep_gemm",
        ),
    )
    import deep_gemm

    if not enforce_eager:
        return deep_gemm.bf16_mqa_logits
    from deep_gemm import bf16_attention

    # The pinned SDK's cpasync/storeCoalesce pipeline aborts FlagTree on the
    # eager short-prefill path. Reuse its kernel with the basic pipeline also
    # used by FL row sharding; never mutate the SDK's global kernel object.
    kernel = local_launch(
        bf16_attention.bf16_mqa_logits_kernel,
        {"pipeline": "basic", "scenario": ""},
    )
    return functools.partial(_eager_logits, kernel)


def configure(enforce_eager=False):
    """Bind the native service mode once, before model execution."""
    global _selected_backend
    if not isinstance(enforce_eager, bool):
        raise TypeError("enforce_eager must be a bool")
    backend = _backend(enforce_eager)
    if _selected_backend is not None and _selected_backend is not backend:
        raise RuntimeError("DeepGEMM execution mode cannot change after initialization")
    _selected_backend = backend


def bf16_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    if q.shape[0] == 0 or kv.shape[0] == 0:
        return torch.empty(
            (q.shape[0], kv.shape[0]), device=q.device, dtype=torch.float32
        )
    backend = _selected_backend if _selected_backend is not None else _backend()
    return backend(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from vllm_fl.minimax_m3_config import load

"""Length-independent local-query packed BF16 GQA; no cross-rank exchange.
Kernel handles query-specific top-k membership and arbitrary page tables. Caller
guard restricts H4/HKV1/D128, block128, K16, BF16. GPU lengths and membership remain dynamic; wholly unselected pages skip GEMMs. Top-k order is
replayed ascending, retaining exactly the selected set and causal mask; therefore
numerical rather than bitwise equivalence is required."""
import torch
import triton

from .long_prefill import packed_prefill_kernel

LAST_KERNELS = []
COLLECT_METADATA = False
WARPS = load().attention["base_prefill"]["warps"]
STAGES = load().attention["base_prefill"]["stages"]
QPACK = load().attention["base_prefill"]["qpack"]
KTILE = load().attention["base_prefill"]["ktile"]


@torch.no_grad()
def minimax_m3_sparse_attn(
    q, kv, idx, bt, cu, lens, prefix, max_query_len, num_kv_heads, scale, out
):
    assert q.shape[1:] == (4, 128) and num_kv_heads == 1 and idx.shape[-1] == 16
    assert kv.dtype == q.dtype == torch.bfloat16
    k = packed_prefill_kernel[(triton.cdiv(max_query_len, QPACK), cu.numel() - 1)](
        q,
        kv,
        idx,
        bt,
        cu,
        lens,
        prefix,
        out,
        *q.stride(),
        *kv.stride(),
        *idx.stride(),
        bt.stride(0),
        *out.stride(),
        scale,
        QPACK,
        KTILE,
        False,  # reuse the unchanged full-page traversal branch
        num_warps=WARPS,
        num_stages=STAGES,
    )
    if COLLECT_METADATA:
        LAST_KERNELS.append(k)

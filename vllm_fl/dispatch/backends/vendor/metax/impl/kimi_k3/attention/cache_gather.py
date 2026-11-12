# SPDX-License-Identifier: Apache-2.0
"""BF16-only port of vLLM gather_and_maybe_dequant_cache indexing semantics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(
    SRC,
    DST,
    TABLE,
    CU,
    STARTS,
    TOKENS: tl.constexpr,
    REQUESTS: tl.constexpr,
    WIDTH: tl.constexpr,
    BS: tl.constexpr,
    SRC_BLOCK,  # Runtime stride avoids the large-constant lowering fault on C550.
    SRC_ENTRY: tl.constexpr,
    DST_ENTRY: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COLUMNS: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    NR: tl.constexpr,
    BD: tl.constexpr,
):
    token = tl.program_id(0) * 4 + tl.arange(0, 4)
    requests = tl.arange(0, NR)
    begins = tl.load(CU + requests, requests < REQUESTS, other=2147483647)
    seq = (
        tl.sum(
            ((token[:, None] >= begins[None, :]) & (requests[None, :] < REQUESTS)).to(
                tl.int32
            ),
            axis=1,
        )
        - 1
    )
    begin = tl.load(CU + seq, token < TOKENS, other=0)
    start = tl.load(STARTS + seq, token < TOKENS, other=0) if HAS_STARTS else 0
    position = token - begin + start
    logical_block = position // BS
    valid = (token < TOKENS) & (logical_block < TABLE_COLUMNS)
    physical_block = tl.load(TABLE + seq * TABLE_ROW + logical_block, valid, other=0)
    d = tl.arange(0, BD)
    values = tl.load(
        SRC
        + physical_block[:, None].to(tl.int64) * SRC_BLOCK
        + (position % BS)[:, None] * SRC_ENTRY
        + d[None, :],
        valid[:, None] & (d[None, :] < WIDTH),
        other=0,
    )
    tl.store(
        DST + token[:, None].to(tl.int64) * DST_ENTRY + d[None, :],
        values,
        valid[:, None] & (d[None, :] < WIDTH),
    )


def gather(
    src_cache,
    dst,
    block_table,
    cu_seq_lens,
    token_to_seq,
    num_tokens,
    kv_cache_dtype,
    scale,
    seq_starts=None,
):
    # Like upstream's page kernel, derive request identity from cu_seq_lens.
    if (
        kv_cache_dtype not in ("auto", "bfloat16")
        or src_cache.dtype != torch.bfloat16
        or dst.dtype != torch.bfloat16
    ):
        raise NotImplementedError("The K3 gather adapter supports BF16 cache only")
    if (
        src_cache.shape[-1] != 576
        or dst.shape[-1] != 576
        or src_cache.stride(-1) != 1
        or dst.stride(-1) != 1
    ):
        raise ValueError("Expected contiguous latent512+extra64 entries")
    if num_tokens == 0:
        return
    requests = cu_seq_lens.numel() - 1
    _gather[(triton.cdiv(num_tokens, 4),)](
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        seq_starts,
        num_tokens,
        requests,
        576,
        src_cache.shape[1],
        src_cache.stride(0),
        src_cache.stride(1),
        dst.stride(0),
        block_table.stride(0),
        block_table.shape[1],
        seq_starts is not None,
        triton.next_power_of_2(requests),
        1024,
        num_warps=4,
    )

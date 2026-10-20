# Copyright (c) 2026 BAAI. All rights reserved.

"""Paged QSA scores without the failing Hopper/FlagTree dot lowering.

The upstream QSA caller still owns chunking, top-k and index expansion. This
kernel computes the same sum of per-head ReLU dot products using FP32 scalar
reductions, preserving paging and causal visibility without an MMA pipeline.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _paged_scores(
    q,
    cache,
    table,
    starts,
    visible,
    output,
    q_row_stride,
    q_head_stride,
    cache_page_stride,
    cache_row_stride,
    table_stride,
    output_stride,
    query_offset,
    num_queries,
    PAGE_SIZE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    DECODE_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = tl.program_id(0)
    if DECODE_LEN:
        row = request * DECODE_LEN + tl.program_id(1)
        valid_row = tl.program_id(1) < DECODE_LEN
    else:
        base = tl.load(starts)
        start = tl.maximum(tl.load(starts + request) - base, query_offset)
        end = tl.minimum(
            tl.load(starts + request + 1) - base, query_offset + num_queries
        )
        row = start + tl.program_id(1)
        valid_row = row < end
    if not valid_row:
        return
    n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    length = tl.load(visible + row)
    valid_n = (n < WIDTH) & (n < length)
    d = tl.arange(0, BLOCK_D)
    page = tl.load(
        table + request * table_stride + n // PAGE_SIZE,
        mask=valid_n & (n // PAGE_SIZE < TABLE_WIDTH),
        other=0,
    )
    keys = tl.load(
        cache
        + page[:, None].to(tl.int64) * cache_page_stride
        + (n % PAGE_SIZE)[:, None] * cache_row_stride
        + d[None, :],
        mask=valid_n[:, None] & (d[None, :] < DIM),
        other=0.0,
    ).to(tl.float32)
    score = tl.full((BLOCK_N,), 0, tl.float32)
    for head in tl.static_range(HEADS):
        query = tl.load(
            q + row * q_row_stride + head * q_head_stride + d, mask=d < DIM, other=0.0
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)
    tl.store(output + (row - query_offset) * output_stride + n, score, mask=valid_n)


def _validate(q, cache, table, visible):
    if q.ndim != 3 or cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA expects [rows, heads, dim] and [pages, tokens, 1, dim]")
    if q.shape[-1] != cache.shape[-1] or q.dtype != cache.dtype:
        raise ValueError("QSA query and cache dimensions/dtypes must match")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raise ValueError("Unsupported QSA query/cache dtype")
    if any(not t.is_cuda or t.device != q.device for t in (q, cache, table, visible)):
        raise ValueError("QSA inputs must share one CUDA device")
    if q.stride(-1) != 1 or cache.stride(-1) != 1:
        raise ValueError("QSA head dimensions must be contiguous")
    if table.ndim != 2 or table.stride(1) != 1 or table.dtype != torch.int32:
        raise ValueError("QSA page table must have contiguous int32 rows")
    if visible.shape != (q.shape[0],) or visible.stride(0) != 1:
        raise ValueError("QSA visibility must be one contiguous length per query")
    if visible.dtype != torch.int32:
        raise ValueError("QSA visibility must be int32")


def prefill_logits(
    q,
    k_cache,
    page_table,
    query_start_loc,
    visible_blocks,
    logits_workspace,
    max_query_len,
    logits_width,
    query_offset,
    num_queries,
):
    """Match upstream's chunked prefill logits ABI and workspace aliasing."""
    _validate(q, k_cache, page_table, visible_blocks)
    if (
        query_start_loc.shape != (page_table.shape[0] + 1,)
        or query_start_loc.dtype != torch.int32
        or not query_start_loc.is_contiguous()
        or query_start_loc.device != q.device
    ):
        raise ValueError(
            "QSA query offsets must be contiguous int32 on the query device"
        )
    if not 0 <= query_offset <= query_offset + num_queries <= q.shape[0]:
        raise ValueError("Invalid QSA query chunk")
    if not 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]:
        raise ValueError("Invalid QSA logits width")
    if (
        logits_workspace.dtype != torch.float32
        or logits_workspace.device != q.device
        or not logits_workspace.is_contiguous()
        or logits_workspace.numel() < num_queries * logits_width
    ):
        raise ValueError(
            "QSA logits workspace must contain sufficient contiguous FP32 storage"
        )
    logits = logits_workspace[: num_queries * logits_width].view(
        num_queries, logits_width
    )
    if num_queries == 0:
        return logits
    _paged_scores[
        (
            page_table.shape[0],
            min(num_queries, max_query_len),
            triton.cdiv(logits_width, 16),
        )
    ](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        query_offset,
        num_queries,
        PAGE_SIZE=k_cache.shape[1],
        TABLE_WIDTH=page_table.shape[1],
        HEADS=q.shape[1],
        DIM=q.shape[2],
        WIDTH=logits_width,
        DECODE_LEN=0,
        BLOCK_N=16,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        num_warps=4,
    )
    return logits


def decode_logits(q, k_cache, page_table, visible_blocks, decode_query_len):
    """Score a request-major decode batch using the same paged reduction."""
    _validate(q, k_cache, page_table, visible_blocks)
    if (
        decode_query_len <= 0
        or q.shape[0] % decode_query_len
        or page_table.shape[0] != q.shape[0] // decode_query_len
    ):
        raise ValueError("Invalid QSA request-major decode batch")
    width = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], width), dtype=torch.float32, device=q.device)
    _paged_scores[(page_table.shape[0], decode_query_len, triton.cdiv(width, 16))](
        q,
        k_cache,
        page_table,
        visible_blocks,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        0,
        q.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        TABLE_WIDTH=page_table.shape[1],
        HEADS=q.shape[1],
        DIM=q.shape[2],
        WIDTH=width,
        DECODE_LEN=decode_query_len,
        BLOCK_N=16,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        num_warps=4,
    )
    return logits


def select_paged_decode(
    q,
    k_cache,
    page_table,
    visible_blocks,
    token_topk,
    compress_ratio,
    decode_query_len,
    block_indices,
):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import _topk

    if token_topk % compress_ratio or block_indices.shape != (
        q.shape[0],
        token_topk // compress_ratio,
    ):
        raise ValueError("Invalid QSA top-k output")
    logits = decode_logits(q, k_cache, page_table, visible_blocks, decode_query_len)
    _topk(
        logits,
        visible_blocks,
        token_topk,
        compress_ratio,
        block_indices,
        torch.empty(1024 * 1024, dtype=torch.uint8, device=q.device),
    )


def install_nvidia_qsa(vllm_config) -> bool:
    """Scope the workaround to Qwen4Exp EMPTY on Hopper with FlagTree 3.6."""
    from importlib.metadata import PackageNotFoundError, version

    from vllm_fl.nvidia_compat import has_native_vllm_kernels

    model = vllm_config.model_config
    if model is None or not {
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpForCausalLM",
    }.intersection(model.architectures):
        return False
    if has_native_vllm_kernels() or torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        version("flagtree")
    except PackageNotFoundError:
        return False
    if not triton.__version__.startswith("3.6."):
        return False
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    qsa_indexer._prefill_logits = prefill_logits
    qsa_indexer.qsa_select_paged_decode = select_paged_decode
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "Using paged QSA scalar reductions for Qwen4Exp on EMPTY Hopper/FlagTree 3.6"
    )
    return True

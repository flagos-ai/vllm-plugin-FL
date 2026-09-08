# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon MQA-logits implementations for the GLM-5.2 V3.2 FP8 Indexer.

Prefill reuses the installed FlagGems ``fp8_mqa_logits`` implementation.

Decode cannot directly reuse FlagGems ``fp8_paged_mqa_logits`` because the
vLLM V3.2 Indexer cache uses a split layout inside every physical block:

    [all token FP8 K values][all token FP32 scales]

whereas FlagGems' paged op expects an interleaved per-token layout.

The decode kernel below keeps the existing vLLM/FlagGems cache writer
unchanged and only changes how paged MQA reads K values and scales.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ------------------------------------------------------------
# Functional bring-up configuration for GLM-5.2 / BW1000.
#
# GLM-5.2 V3.2 Indexer:
#
#   index_n_heads = 64
#   index_head_dim = 128
#
# Keep these tiles conservative until correctness is established.
# ------------------------------------------------------------

_BLOCK_KV = 16
_BLOCK_D = 128
_BLOCK_H = 16


def _cdiv(
    x: int,
    y: int,
) -> int:
    return (x + y - 1) // y


def glm_hygon_indexer_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[
        torch.Tensor,
        torch.Tensor,
    ],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Prefill FP8 MQA logits using the installed FlagGems kernel."""

    from flag_gems.ops.fp8_mqa_logits import fp8_mqa_logits

    return fp8_mqa_logits(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits,
    )


# ============================================================
# Decode
# ============================================================

@triton.jit
def _fill_neg_inf_kernel(
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = (
        pid * BLOCK
        + tl.arange(0, BLOCK)
    )

    mask = offsets < n_elements

    tl.store(
        out_ptr + offsets,
        float("-inf"),
        mask=mask,
    )


@triton.jit
def _glm_hygon_fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    weights_ptr,
    logits_ptr,
    block_tables_ptr,
    context_lens_ptr,
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kvblk,
    stride_wrow,
    stride_wh,
    stride_lrow,
    stride_lcol,
    stride_btb,
    stride_bts,
    next_n: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    block_size: tl.constexpr,
    max_model_len,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Paged FP8 MQA for vLLM V3.2 split-layout Indexer cache."""

    pid_row = tl.program_id(0)
    pid_kv_tile = tl.program_id(1)

    batch_idx = (
        pid_row // next_n
    )

    next_n_idx = (
        pid_row % next_n
    )

    # context_lens_ptr stores one final sequence length per request.
    #
    # For speculative decode:
    #
    # token 0 attends to L-next_n+1
    # token 1 attends to L-next_n+2
    # ...
    # token next_n-1 attends to L
    final_context_len = tl.load(
        context_lens_ptr + batch_idx
    )

    query_seq_pos = (
        final_context_len
        - next_n
        + next_n_idx
    )

    effective_context_len = (
        query_seq_pos + 1
    )

    kv_start = (
        pid_kv_tile * BLOCK_KV
    )

    offs_kv = tl.arange(
        0,
        BLOCK_KV,
    )

    kv_global_pos = (
        kv_start + offs_kv
    )

    valid_kv = (
        (kv_global_pos < effective_context_len)
        & (kv_global_pos < max_model_len)
    )

    # --------------------------------------------------------
    # Logical token position -> paged physical block.
    # --------------------------------------------------------

    logical_block_idx = (
        kv_global_pos // block_size
    )

    intra_block_pos = (
        kv_global_pos % block_size
    )

    phys_block_ids = tl.load(
        block_tables_ptr
        + batch_idx * stride_btb
        + logical_block_idx * stride_bts,
        mask=valid_kv,
        other=0,
    )

    # --------------------------------------------------------
    # V3.2 Indexer physical layout.
    #
    # kv_cache nominal tensor:
    #
    #   [num_blocks, block_size, head_dim + 4]
    #
    # But indexer_k_quant_and_cache flattens every block and stores:
    #
    #   values:
    #       block_size * head_dim bytes
    #
    #   scales:
    #       block_size * 4 bytes
    #
    # Therefore:
    #
    # block:
    #
    #   [K0][K1] ... [K(block_size-1)]
    #   [S0][S1] ... [S(block_size-1)]
    # --------------------------------------------------------

    block_base = (
        phys_block_ids.to(tl.int64)
        * stride_kvblk
    )

    value_base = (
        block_base
        + intra_block_pos.to(tl.int64)
        * dim
    )

    scale_base = (
        block_base
        + block_size * dim
        + intra_block_pos.to(tl.int64)
        * 4
    )

    # --------------------------------------------------------
    # Load one FP32 scale.
    #
    # kv_ptr is uint8. Reinterpret the four bytes at scale_base
    # as one uint32, then bitcast to float32.
    # --------------------------------------------------------

    scale_ptr = (
        kv_ptr + scale_base
    ).to(
        tl.pointer_type(
            tl.uint32,
            1,
        ),
        bitcast=True,
    )

    scale_u32 = tl.load(
        scale_ptr,
        mask=valid_kv,
        other=0,
    )

    k_scale = scale_u32.to(
        tl.float32,
        bitcast=True,
    )

    # --------------------------------------------------------
    # Load FP8 K.
    # --------------------------------------------------------

    offs_d = tl.arange(
        0,
        BLOCK_D,
    )

    d_mask = offs_d < dim

    kv_byte_ptrs = (
        kv_ptr
        + value_base[:, None]
        + offs_d[None, :]
    )

    kv_u8 = tl.load(
        kv_byte_ptrs,
        mask=(
            valid_kv[:, None]
            & d_mask[None, :]
        ),
        other=0,
    )

    # Current BW1000 PlatformFL reports:
    #
    #   torch.float8_e4m3fn
    #
    # Reinterpret raw cache bytes as E4M3 FP8.
    kv_fp8 = kv_u8.to(
        tl.float8e4nv,
        bitcast=True,
    )

    kv_f32 = kv_fp8.to(
        tl.float32
    )

    # --------------------------------------------------------
    # Query row.
    # --------------------------------------------------------

    q_base = (
        q_ptr
        + batch_idx * stride_qb
        + next_n_idx * stride_qn
    )

    logit_accum = tl.zeros(
        [BLOCK_KV],
        dtype=tl.float32,
    )

    # --------------------------------------------------------
    # Iterate Indexer heads.
    #
    # score[n] =
    #
    #   sum_h(
    #       relu(
    #           dot(q[h], k[n])
    #       )
    #       * weights[h]
    #   )
    # --------------------------------------------------------

    for h_start in tl.static_range(
        0,
        heads,
        BLOCK_H,
    ):
        offs_h = (
            h_start
            + tl.arange(
                0,
                BLOCK_H,
            )
        )

        h_mask = offs_h < heads

        q_ptrs = (
            q_base
            + offs_h[:, None]
            * stride_qh
            + offs_d[None, :]
            * stride_qd
        )

        q_vals = tl.load(
            q_ptrs,
            mask=(
                h_mask[:, None]
                & d_mask[None, :]
            ),
            other=0.0,
        ).to(
            tl.float32
        )

        row_weights = tl.load(
            weights_ptr
            + pid_row * stride_wrow
            + offs_h * stride_wh,
            mask=h_mask,
            other=0.0,
        ).to(
            tl.float32
        )

        # [BLOCK_KV, D]
        #     @
        # [D, BLOCK_H]
        #
        # ->
        #
        # [BLOCK_KV, BLOCK_H]
        partial_dot = tl.dot(
            kv_f32,
            tl.trans(q_vals),
            out_dtype=tl.float32,
        )

        # K quantization scale.
        partial_dot = (
            partial_dot
            * k_scale[:, None]
        )

        partial_dot = tl.maximum(
            partial_dot,
            0.0,
        )

        logit_accum += tl.sum(
            partial_dot
            * row_weights[None, :],
            axis=1,
        )

    # --------------------------------------------------------
    # Store valid context positions only.
    # Other positions were initialized to -inf.
    # --------------------------------------------------------

    out_ptrs = (
        logits_ptr
        + pid_row * stride_lrow
        + kv_global_pos
        * stride_lcol
    )

    tl.store(
        out_ptrs,
        logit_accum,
        mask=valid_kv,
    )


def glm_hygon_indexer_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    head_dim: int,
    quant_block_size: int,
) -> torch.Tensor:
    """Decode FP8 paged MQA from the V3.2 split-layout Indexer cache."""

    # --------------------------------------------------------
    # Input validation.
    # --------------------------------------------------------

    if q.ndim != 4:
        raise ValueError(
            "q must have shape "
            "[batch, next_n, heads, head_dim], "
            f"but got {tuple(q.shape)}."
        )

    if kv_cache.ndim != 3:
        raise ValueError(
            "kv_cache must have shape "
            "[num_blocks, block_size, head_dim + 4], "
            f"but got {tuple(kv_cache.shape)}."
        )

    if kv_cache.dtype != torch.uint8:
        raise TypeError(
            "GLM Hygon Indexer KV cache must "
            "be uint8, "
            f"but got {kv_cache.dtype}."
        )

    # Keep the initial Hygon implementation deliberately narrow.
    #
    # Your current probe reports:
    #
    #   current_platform.fp8_dtype()
    #   == torch.float8_e4m3fn
    #
    # Do not silently interpret another FP8 encoding as E4M3FN.
    if q.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "GLM Hygon paged MQA currently "
            "expects torch.float8_e4m3fn, "
            f"but got {q.dtype}."
        )

    if quant_block_size != head_dim:
        raise NotImplementedError(
            "GLM Hygon paged MQA currently "
            "supports one FP32 scale per K "
            "vector: quant_block_size == head_dim."
        )

    (
        batch_size,
        next_n,
        heads,
        dim,
    ) = q.shape

    (
        _,
        block_size,
        cache_width,
    ) = kv_cache.shape

    if dim != head_dim:
        raise ValueError(
            f"q head dim {dim} does not match "
            f"head_dim={head_dim}."
        )

    # First implementation intentionally specializes GLM-5.2's
    # index_head_dim=128.
    if head_dim != _BLOCK_D:
        raise NotImplementedError(
            "The initial GLM Hygon paged MQA "
            "kernel is specialized for "
            f"head_dim={_BLOCK_D}, "
            f"but got {head_dim}."
        )

    expected_cache_width = (
        head_dim + 4
    )

    if cache_width != expected_cache_width:
        raise ValueError(
            "Unexpected V3.2 Indexer cache width: "
            f"expected {expected_cache_width}, "
            f"got {cache_width}."
        )

    expected_weights_shape = (
        batch_size * next_n,
        heads,
    )

    if weights.shape != expected_weights_shape:
        raise ValueError(
            "weights must have shape "
            f"{expected_weights_shape}, "
            f"but got {tuple(weights.shape)}."
        )

    if block_tables.ndim != 2:
        raise ValueError(
            "block_tables must be 2D, "
            f"but got ndim={block_tables.ndim}."
        )

    if block_tables.shape[0] < batch_size:
        raise ValueError(
            "block_tables has fewer rows "
            "than decode batch size."
        )

    # --------------------------------------------------------
    # Normalize vLLM seq_lens.
    #
    # Normal decode:
    #
    #   [B]
    # or
    #   [B, 1]
    #
    # Native speculative decode:
    #
    #   [B, next_n]
    #
    # The final column stores the final context length.
    # The Triton kernel reconstructs the preceding per-token
    # lengths using next_n.
    # --------------------------------------------------------

    if context_lens.ndim == 2:
        if context_lens.shape[0] < batch_size:
            raise ValueError(
                "context_lens has fewer rows "
                "than decode batch size."
            )

        context_lens_1d = (
            context_lens[
                :batch_size,
                -1,
            ]
        )

    elif context_lens.ndim == 1:
        if context_lens.shape[0] < batch_size:
            raise ValueError(
                "context_lens has fewer elements "
                "than decode batch size."
            )

        context_lens_1d = (
            context_lens[
                :batch_size
            ]
        )

    else:
        raise ValueError(
            "context_lens must be 1D or 2D, "
            f"but got ndim={context_lens.ndim}."
        )

    if context_lens_1d.dtype != torch.int32:
        raise TypeError(
            "context_lens must be int32, "
            f"but got {context_lens_1d.dtype}."
        )

    if block_tables.dtype != torch.int32:
        raise TypeError(
            "block_tables must be int32, "
            f"but got {block_tables.dtype}."
        )

    if not kv_cache.is_contiguous():
        raise ValueError(
            "GLM Hygon Indexer KV cache "
            "must be contiguous."
        )

    q_contig = q.contiguous()

    weights_contig = (
        weights.contiguous()
    )

    context_lens_contig = (
        context_lens_1d.contiguous()
    )

    block_tables_contig = (
        block_tables[
            :batch_size
        ].contiguous()
    )

    total_rows = (
        batch_size * next_n
    )

    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    logits = torch.empty(
        (
            total_rows,
            max_model_len,
        ),
        device=q.device,
        dtype=torch.float32,
    )

    # Positions outside valid context stay -inf.
    n_elements = (
        total_rows * max_model_len
    )

    fill_block = 1024

    _fill_neg_inf_kernel[
        (
            _cdiv(
                n_elements,
                fill_block,
            ),
        )
    ](
        logits,
        n_elements,
        BLOCK=fill_block,
    )

    # Maximum logical KV positions addressable by block table.
    max_context = (
        block_tables_contig.shape[1]
        * block_size
    )

    grid = (
        total_rows,
        _cdiv(
            max_context,
            _BLOCK_KV,
        ),
    )

    _glm_hygon_fp8_paged_mqa_logits_kernel[
        grid
    ](
        q_contig,
        kv_cache,
        weights_contig,
        logits,
        block_tables_contig,
        context_lens_contig,
        q_contig.stride(0),
        q_contig.stride(1),
        q_contig.stride(2),
        q_contig.stride(3),
        kv_cache.stride(0),
        weights_contig.stride(0),
        weights_contig.stride(1),
        logits.stride(0),
        logits.stride(1),
        block_tables_contig.stride(0),
        block_tables_contig.stride(1),
        next_n,
        heads,
        dim,
        block_size,
        max_model_len,
        BLOCK_KV=_BLOCK_KV,
        BLOCK_D=_BLOCK_D,
        BLOCK_H=_BLOCK_H,
        num_warps=4,
        num_stages=1,
    )

    return logits
# Copyright (c) 2026 BAAI. All rights reserved.

"""Hopper FP8 indexer operations for DeepSeek-V4.1 on an EMPTY wheel.

The adapters preserve vLLM's output buffers and reject formats that the
selected FlagGems kernels do not implement. They do not change native
DeepGEMM/FlashMLA availability checks.
"""

import torch
import triton


def _same_cuda(*tensors):
    if not tensors or any(
        not tensor.is_cuda or tensor.device != tensors[0].device for tensor in tensors
    ):
        raise ValueError("DSV4.1 tensors must share a CUDA device")


def _validate_fp8_indexer_cache(kv_cache):
    """Require dense page contents, allowing upstream interleaved page strides."""
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.ndim not in (3, 4)
        or kv_cache.shape[-1] != 132
        or (kv_cache.ndim == 4 and kv_cache.shape[2] != 1)
        or kv_cache.stride(-1) != 1
        or kv_cache.stride(1) != 132
        or kv_cache.stride(0) < kv_cache.shape[1] * 132
        or kv_cache.stride(0) % 4
        or kv_cache.storage_offset() % 4
    ):
        raise ValueError(
            "Indexer cache requires packed uint8 FP8/FP32 pages with "
            "contiguous page contents and aligned, non-overlapping page strides"
        )


def fused_indexer_q_rope_quant(
    positions,
    index_q,
    index_q_cos_sin_cache,
    index_weights,
    index_weights_softmax_scale,
    index_weights_head_scale,
    use_fp4=False,
    weights_out_dtype=torch.float32,
):
    """Select vLLM's real Triton FP8 query kernel without the bundled FA package.

    The upstream CuTe dispatcher checks the installed DSL, but its kernel also
    imports utilities from vLLM's omitted FlashAttention package. Call the
    upstream Triton launcher directly; RoPE and scale folding stay upstream.
    """
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        _FUSED_INDEXER_Q_ROPE_QUANT_TRITON_KERNEL,
    )

    if use_fp4 or weights_out_dtype != torch.float32:
        raise NotImplementedError(
            "DSV4.1 EMPTY indexer Q requires FP8 and FP32 weights"
        )
    _same_cuda(positions, index_q, index_q_cos_sin_cache, index_weights)
    if (
        index_q.ndim != 3
        or index_q.shape[1] not in (32, 64)
        or index_q.shape[2] != 128
        or index_q.dtype != torch.bfloat16
        or index_q.stride(-1) != 1
    ):
        raise ValueError(
            "Indexer Q must be BF16 [tokens, 32/64 heads, 128] with contiguous head elements"
        )
    tokens, heads, _ = index_q.shape
    if (
        positions.dtype != torch.int64
        or positions.shape != (tokens,)
        or not positions.is_contiguous()
        or index_weights.shape != (tokens, heads)
        or index_weights.dtype not in (torch.bfloat16, torch.float32)
        or index_weights.stride(-1) != 1
        or index_q_cos_sin_cache.ndim != 2
        or index_q_cos_sin_cache.shape[1] != 64
        or index_q_cos_sin_cache.dtype != torch.float32
        or index_q_cos_sin_cache.stride(-1) != 1
    ):
        raise ValueError(
            "Indexer Q requires contiguous int64 positions, FP32 64-dim RoPE cache, and BF16/FP32 per-head weights"
        )
    query = torch.empty(index_q.shape, dtype=torch.float8_e4m3fn, device=index_q.device)
    weights = torch.empty(
        index_weights.shape, dtype=torch.float32, device=index_q.device
    )
    if tokens:
        _FUSED_INDEXER_Q_ROPE_QUANT_TRITON_KERNEL(
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            query,
            weights,
            fp8_max=448.0,
            use_fnuz=False,
        )
    return query, weights


def fp8_mqa_logits(q, kv, weights, row_starts, row_ends, clean_logits=False):
    from flag_gems.fused.fp8_fp4_mqa_logits import fp8_fp4_mqa_logits

    values, scales = q
    keys, key_scales = kv
    if scales is not None or values.dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "DSV4.1 indexer requires FP8 Q without separate scales"
        )
    if keys.dtype != torch.float8_e4m3fn or key_scales.dtype != torch.float32:
        raise NotImplementedError("DSV4.1 indexer requires FP8 K with FP32 scales")
    _same_cuda(values, keys, key_scales, weights, row_starts, row_ends)
    rows, heads, dim = values.shape
    if dim != 128 or heads not in (32, 64) or keys.shape[1:] != (dim,):
        raise ValueError("DSV4.1 indexer requires head dimension 128 and 32/64 heads")
    if weights.dtype != torch.float32 or weights.shape != (rows, heads):
        raise ValueError("Indexer weights must be FP32 [rows, heads]")
    if key_scales.shape != (keys.shape[0],):
        raise ValueError("Indexer key scales must contain one FP32 scale per token")
    if any(
        t.dtype != torch.int32 or t.shape != (rows,) for t in (row_starts, row_ends)
    ):
        raise ValueError("Indexer ranges must be int32 [rows]")
    return fp8_fp4_mqa_logits(q, kv, weights, row_starts, row_ends, clean_logits)


def fp8_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
    indices=None,
):
    if indices is not None:
        raise NotImplementedError(
            "DSV4.1 FP8 indexer does not implement varlen indices"
        )
    values, scales = q
    if scales is not None or values.dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "DSV4.1 indexer requires FP8 Q without separate scales"
        )
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim != 4 or values.shape[1] != 1:
        raise NotImplementedError("DSV4.1 FP8 indexer currently requires next_n=1")
    batch, _, heads, dim = values.shape
    if dim != 128 or heads not in (32, 64):
        raise ValueError("DSV4.1 indexer requires head dimension 128 and 32/64 heads")
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 4
        or kv_cache.shape[2:] != (1, 132)
    ):
        raise ValueError("FP8 indexer cache must be uint8 [pages, block, 1, 132]")
    block = kv_cache.shape[1]
    if block not in (32, 64, 128):
        raise NotImplementedError("FP8 indexer supports cache blocks 32, 64, or 128")
    _validate_fp8_indexer_cache(kv_cache)
    if not values.is_contiguous():
        raise ValueError("FP8 indexer queries must be contiguous")
    if (
        weights.dtype != torch.float32
        or weights.shape != (batch, heads)
        or not weights.is_contiguous()
    ):
        raise ValueError("Indexer weights must be contiguous FP32 [batch, heads]")
    if (
        context_lens.dtype != torch.int32
        or context_lens.numel() != batch
        or not context_lens.is_contiguous()
    ):
        raise ValueError("Indexer lengths must be contiguous int32 [batch, 1]")
    if (
        block_tables.dtype != torch.int32
        or block_tables.shape[0] != batch
        or block_tables.stride(-1) != 1
    ):
        raise ValueError("Indexer block table must be int32 with contiguous rows")
    if max_model_len <= 0 or block_tables.shape[1] < triton.cdiv(max_model_len, block):
        raise ValueError("Indexer block table cannot cover max_model_len")
    _same_cuda(values, kv_cache, weights, context_lens, block_tables)
    output = torch.full(
        (batch, max_model_len),
        float("-inf") if clean_logits else 0,
        dtype=torch.float32,
        device=values.device,
    )
    if batch:
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            _fp8_paged_mqa_logits_decode_kernel,
        )

        # This upstream Triton kernel also runs on Hopper without AITER. Its
        # separate physical page strides preserve padding and shared allocations;
        # its device-side lengths keep a fixed launch grid safe for CUDA graphs.
        # The public ROCm launcher fixes BLOCK_KV=64, whereas compressed DS pages
        # can have 32 tokens. Pass the actual page width to the shared kernel.
        page_bytes = kv_cache.view(kv_cache.shape[0], -1)
        page_values = page_bytes.view(torch.float8_e4m3fn)
        page_scales = page_bytes.view(torch.float32)
        splits = triton.cdiv(max_model_len, block)
        _fp8_paged_mqa_logits_decode_kernel[(batch, splits)](
            values,
            page_values,
            page_scales,
            weights,
            context_lens,
            block_tables,
            output,
            values.stride(0),
            values.stride(1),
            values.stride(2),
            weights.stride(0),
            page_values.stride(0),
            page_scales.stride(0),
            block * dim // 4,
            block_tables.stride(0),
            output.stride(0),
            block_tables.shape[1],
            max_model_len,
            NUM_HEADS=heads,
            HEAD_SIZE=dim,
            BLOCK_SIZE=block,
            BLOCK_KV=block,
            N_SPLITS=splits,
            NEXT_N=1,
            CTX_PER_ROW=True,
            num_warps=4,
            num_stages=2,
        )
    return output


def fp8_einsum(equation, a, b, out, recipe=(1, 128, 128)):
    from flag_gems.runtime.backend._nvidia.hopper.ops.fp8_einsum import (
        fp8_einsum as gems_einsum,
    )

    x, xs = a
    y, ys = b
    if equation != "bhr,hdr->bhd" or tuple(recipe) != (1, 128, 128):
        raise NotImplementedError(
            "Hopper DSV4.1 WO-A requires bhr,hdr->bhd with recipe (1,128,128)"
        )
    if x.dtype != torch.float8_e4m3fn or y.dtype != torch.float8_e4m3fn:
        raise NotImplementedError("Hopper DSV4.1 WO-A requires E4M3 FP8 operands")
    if xs.dtype != torch.float32 or ys.dtype != torch.float32:
        raise NotImplementedError("Hopper DSV4.1 WO-A requires FP32 scales")
    _same_cuda(x, xs, y, ys, out)
    batch, heads, width = x.shape
    if width % 128 or y.shape[0] != heads or y.shape[2] != width or y.shape[1] % 128:
        raise ValueError(
            "WO-A operands must have matching heads and dimensions divisible by 128"
        )
    if xs.shape != (batch, heads, width // 128) or ys.shape != (
        heads,
        y.shape[1] // 128,
        width // 128,
    ):
        raise ValueError("WO-A scale shapes do not match the block-128 operands")
    if out.dtype != torch.bfloat16 or out.shape != (batch, heads, y.shape[1]):
        raise ValueError("WO-A output must be BF16 [tokens, heads, output_dim]")
    # The pinned Hopper kernel uses split-K for the model's TP8 shape. Its
    # BF16 output accumulates separately rounded partial sums with atomic_add.
    # Keep the supported FP32 accumulator and round only once into vLLM's out.
    out.copy_(
        gems_einsum(
            equation, x, xs, y, ys, block_size=[128, 128], output_dtype=torch.float32
        )
    )

# Copyright (c) 2026 BAAI. All rights reserved.
"""DeepSeek-V4 indexer-Q RoPE+quant for PPU (no fp8e4nv triton support).

Ported from the T-Head vendor fork
(v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py, SM80/PPU branch):
the triton kernel emits scaled fp32 Q and folds the per-token-head scale
into weights_out; the wrapper then casts Q to int8 in PyTorch. The PPU
deep_gemm's fp8_fp4_mqa_logits/paged variant accepts int8 Q directly
(dispatching to its int8_mqa_logits kernels).

Key quantization notes carried over from the vendor kernel docstring:
  - scale divisor is 127.0 (INT8 max), not 448.0 (FP8 E4M3 max)
  - no UE8M0 rounding: the scale is folded into fp32 weights, so
    power-of-two rounding would only enlarge the quantization step.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM
    )
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fused_indexer_q_rope_fp32_kernel(
    pos_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    out_fp32_ptr,
    out_stride0,
    out_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
    index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 127.0)

    out_base_ptr = out_fp32_ptr + tok_idx * out_stride0 + head_idx * out_stride1
    if INDEX_Q_NOPE_DIM > 0:
        tl.store(out_base_ptr + nope_offset, tl.div_rn(x_nope, index_q_scale))
    out_rot_base = out_base_ptr + INDEX_Q_NOPE_DIM
    tl.store(out_rot_base + half_offset * 2, tl.div_rn(r_even, index_q_scale))
    tl.store(out_rot_base + half_offset * 2 + 1, tl.div_rn(r_odd, index_q_scale))

    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    )
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def _fl_indexer_q_rope_int8_impl(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]
    out_fp32 = torch.empty(
        (num_tokens, num_index_q_heads, index_q_head_dim),
        dtype=torch.float32,
        device=index_q.device,
    )
    weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    _fused_indexer_q_rope_fp32_kernel[(num_tokens, num_index_q_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        out_fp32,
        out_fp32.stride(0),
        out_fp32.stride(1),
        index_q_head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        weights_out,
        weights_out.stride(0),
        num_warps=1,
    )
    return out_fp32.to(torch.int8), weights_out


def _fl_indexer_q_rope_int8_fake(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(index_q, dtype=torch.int8),
        torch.empty_like(index_weights, dtype=torch.float32),
    )


direct_register_custom_op(
    op_name="fl_indexer_q_rope_int8",
    op_func=_fl_indexer_q_rope_int8_impl,
    fake_impl=_fl_indexer_q_rope_int8_fake,
)


def fused_indexer_q_rope_quant_ppu(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
):
    """Drop-in replacement for the upstream fused_indexer_q_rope_quant.

    Returns (q_int8, weights_out) matching the FP8-path contract (scale
    folded into weights); MXFP4 is not supported on PPU.
    """
    assert not use_fp4, "MXFP4 indexer Q is not supported on PPU"
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    return torch.ops.vllm.fl_indexer_q_rope_int8(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
    )

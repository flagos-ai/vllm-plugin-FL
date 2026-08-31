# Copyright (c) 2026 BAAI. All rights reserved.
"""DeepSeek-V4 o_proj for INT8 (W8A8 per-channel) wo_a on PPU.

Ported from the T-Head vendor fork (vllm 0.20.1+ppu):
  - triton kernel `_fused_inv_rope_float32_per_head` and the
    `fused_inv_rope_float32` wrapper come from
    v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py
  - the INT8 branch of DeepseekV4MLAAttentionWrapper.forward comes from
    model_executor/layers/deepseek_v4_attention.py

Rationale: the upstream NVIDIA `_o_proj` quantizes attention output to fp8
(triton fp8e4nv) and multiplies with an fp8 wo_a via deep_gemm's fp8_einsum.
Neither fp8e4nv triton codegen nor fp8 deep_gemm einsum is available on PPU,
and with this checkpoint wo_a is INT8 anyway. The vendor path applies inverse
RoPE in fp32 and runs a bf16/fp32 einsum against the dequantized wo_a.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _fused_inv_rope_float32_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    o_f_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    o_f_stride_token,
    o_f_stride_group,
    HEAD_DIM: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Fused inverse RoPE with direct float32 output (no quantisation).

    One program per (token, global_head). Applies inverse RoPE to the
    trailing rope_dim elements of each head, then stores float32 values into
    [num_tokens, n_groups, heads_per_group * head_dim] layout.
    """
    # int64: stride multiply overflows int32 past num_tokens=32768 (IMA).
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh

    if pid_token >= num_tokens:
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    # -- inverse RoPE (trailing rope_dim elements) ---------------------------
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= ROPE_START
    rope_local = offsets - ROPE_START

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    out_base = (
        o_f_ptr
        + pid_token * o_f_stride_token
        + g * o_f_stride_group
        + head_in_group * HEAD_DIM
    )
    tl.store(out_base + offsets, x)


def _fl_fused_inv_rope_float32_impl(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> torch.Tensor:
    o_f = torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.float32,
        device=o.device,
    )
    grid = (num_tokens, n_groups * heads_per_group)
    _fused_inv_rope_float32_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        o_f,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        o_f_stride_token=o_f.stride(0),
        o_f_stride_group=o_f.stride(1),
        HEAD_DIM=head_dim,
        ROPE_START=rope_start,
        HALF_ROPE=half_rope,
        num_warps=1,
        num_stages=1,
    )
    return o_f


def _fl_fused_inv_rope_float32_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> torch.Tensor:
    return torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.float32,
        device=o.device,
    )


direct_register_custom_op(
    op_name="fl_fused_inv_rope_float32",
    op_func=_fl_fused_inv_rope_float32_impl,
    fake_impl=_fl_fused_inv_rope_float32_fake,
)


def fused_inv_rope_float32(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Inverse RoPE, float32 out; see vendor docstring for layout contract."""
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    # Custom-op boundary keeps inductor from tracing into triton launch
    # (upstream gh-41106).
    return torch.ops.vllm.fl_fused_inv_rope_float32(
        o, positions, cos_sin_cache,
        heads_per_group, nope_dim, rope_dim // 2,
        num_tokens, n_groups, d, head_dim,
    )


def _dequant_channel(b: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """Per-channel INT8 weight -> float32, output layout [out, in].

    Handles both weight orientations: the vendor PPU kernel keeps [out, in],
    while upstream TritonInt8ScaledMMLinearKernel transposes to [in, out] in
    process_weights_after_loading. b_scale has one value per output channel.
    """
    if b_scale is None:
        return b.to(torch.float32)
    b_scale_f = b_scale.to(torch.float32).reshape(-1)
    n_out = b_scale_f.shape[0]
    if b.shape[0] == n_out:
        return b.to(torch.float32) * b_scale_f.unsqueeze(-1)
    assert b.shape[1] == n_out, (
        f"weight shape {tuple(b.shape)} does not match scale length {n_out}"
    )
    return (b.to(torch.float32) * b_scale_f.unsqueeze(0)).t().contiguous()


def _fl_unquantized_einsum_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    out.copy_(torch.einsum(equation, a, b).to(out.dtype))


def _fl_unquantized_einsum_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="fl_deepseek_v4_unquantized_einsum",
    op_func=_fl_unquantized_einsum_impl,
    mutates_args=["out"],
    fake_impl=_fl_unquantized_einsum_fake,
)


def _fl_int8_o_proj_core_impl(
    z: torch.Tensor,
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_f: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    """Chunked inv-RoPE + fp32 einsum, opaque to dynamo; writes into z.

    Fallback path when the deep_gemm int8 GEMM cannot be used (n_groups > 1
    or K % 128 != 0). z is caller-allocated inside the compiled graph so
    cudagraph pools it; chunking bounds the fp32 intermediate.
    """
    num_tokens = o.shape[0]
    CHUNK = 8192
    for start in range(0, num_tokens, CHUNK):
        end = min(start + CHUNK, num_tokens)
        o_f = fused_inv_rope_float32(
            o[start:end],
            positions[start:end],
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        z[start:end].copy_(
            torch.einsum("bhr,hdr->bhd", o_f, wo_a_f).to(torch.bfloat16)
        )


def _fl_int8_o_proj_gemm_impl(
    z: torch.Tensor,          # [T, 1, o_lora] bf16 out
    o: torch.Tensor,          # [T, H, head_dim] bf16 attention output
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    w_q: torch.Tensor,        # [o_lora, R] int8 (checkpoint layout, R=H*head_dim)
    w_s: torch.Tensor,        # [o_lora] fp32 per-channel scales
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    """INT8 o_proj for the n_groups == 1 case (TP8 deployment).

    inv-RoPE (fp32, triton) -> per-token int8 quant -> deep_gemm
    int8xint8->bf16 NT GEMM using the checkpoint's int8 wo_a directly (no
    dequantized fp32 weight copy, no fp32 einsum). Activation int8 quant is
    the only added rounding vs the fp32 path.
    """
    import deep_gemm

    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )

    num_tokens = o.shape[0]
    CHUNK = 8192
    for start in range(0, num_tokens, CHUNK):
        end = min(start + CHUNK, num_tokens)
        o_f = fused_inv_rope_float32(
            o[start:end],
            positions[start:end],
            cos_sin_cache,
            n_groups=1,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )  # [t, 1, R] fp32
        a2d = o_f.view(end - start, -1)
        a_q, a_s = per_token_quant_int8(a2d)
        deep_gemm.gemm_int8_int8_bf16_nt(
            (a_q, a_s.view(-1)),
            (w_q, w_s),
            z[start:end].view(end - start, -1),
        )


def _fl_int8_o_proj_gemm_fake(
    z, o, positions, cos_sin_cache, w_q, w_s,
    heads_per_group, nope_dim, rope_dim,
) -> None:
    return None


direct_register_custom_op(
    op_name="fl_int8_o_proj_gemm",
    op_func=_fl_int8_o_proj_gemm_impl,
    mutates_args=["z"],
    fake_impl=_fl_int8_o_proj_gemm_fake,
)


def _fl_int8_o_proj_core_fake(
    z, o, positions, cos_sin_cache, wo_a_f, n_groups, heads_per_group,
    nope_dim, rope_dim,
) -> None:
    return None


direct_register_custom_op(
    op_name="fl_int8_o_proj_core",
    op_func=_fl_int8_o_proj_core_impl,
    mutates_args=["z"],
    fake_impl=_fl_int8_o_proj_core_fake,
)


def int8_o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Replacement for DeepseekV4FlashMLAAttention._o_proj on PPU/INT8.

    Vendor INT8 branch: fp32 inverse RoPE -> dequant wo_a -> einsum -> wo_b.
    wo_a's fp32 dequant is cached on the layer (weights are static).
    """
    wo_a = self.wo_a.weight
    wo_a_scale = getattr(self.wo_a, "weight_scale", None)

    z = torch.empty(
        (o.shape[0], self.n_local_groups, self.o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )

    # Fast path: single local group (TP8) with K % 128 == 0 -> use the
    # checkpoint's int8 wo_a directly via deep_gemm (skips the fp32 weight
    # copy and fp32 einsum). Otherwise fall back to fp32 einsum.
    R = self.n_local_heads * (self.nope_head_dim + self.rope_head_dim)
    use_int8_gemm = (
        self.n_local_groups == 1
        and R % 128 == 0
        and wo_a.shape == (self.o_lora_rank, R)
        and wo_a_scale is not None
    )
    if use_int8_gemm:
        w_s = getattr(self, "_fl_wo_a_s_flat", None)
        if w_s is None:
            w_s = wo_a_scale.to(torch.float32).reshape(-1).contiguous()
            self._fl_wo_a_s_flat = w_s
        torch.ops.vllm.fl_int8_o_proj_gemm(
            z,
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            wo_a,
            w_s,
            self.n_local_heads,
            self.nope_head_dim,
            self.rope_head_dim,
        )
        return self.wo_b(z.flatten(1))

    wo_a_f = getattr(self, "_fl_wo_a_f_cache", None)
    if wo_a_f is None:
        wo_a_f = _dequant_channel(wo_a, wo_a_scale).reshape(
            self.n_local_groups, self.o_lora_rank, -1
        )
        self._fl_wo_a_f_cache = wo_a_f

    torch.ops.vllm.fl_int8_o_proj_core(
        z,
        o,
        positions,
        self.rotary_emb.cos_sin_cache,
        wo_a_f,
        self.n_local_groups,
        self.n_local_heads // self.n_local_groups,
        self.nope_head_dim,
        self.rope_head_dim,
    )
    return self.wo_b(z.flatten(1))

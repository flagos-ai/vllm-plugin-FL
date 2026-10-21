# SPDX-License-Identifier: Apache-2.0
"""Allocation-free DeepSeek-V4 QNorm/RoPE and FP8 KV-cache insert."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _qnorm_rope_kv_quant_insert_out_kernel(
    q_ptr,
    kv_ptr,
    q_out_ptr,
    k_cache_ptr,
    slot_mapping_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    num_tokens,
    num_tokens_insert,
    eps: tl.constexpr,
    num_heads: tl.constexpr,
    padded_heads: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_max: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    slot_idx = tl.program_id(1)
    if token_idx >= num_tokens:
        return

    if slot_idx < padded_heads:
        out_base = (
            q_out_ptr + (token_idx * padded_heads + slot_idx.to(tl.int64)) * head_dim
        )
        offsets = tl.arange(0, head_dim)
        if slot_idx < num_heads:
            q_base = q_ptr + (token_idx * num_heads + slot_idx.to(tl.int64)) * head_dim
            q_vals = tl.load(q_base + offsets).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(q_vals * q_vals, axis=0) / head_dim + eps)
            tl.store(
                out_base + offsets,
                (q_vals * inv_rms).to(q_out_ptr.type.element_ty),
                mask=offsets < nope_dim,
            )

            pair = tl.arange(0, rope_dim // 2)
            even = nope_dim + pair * 2
            odd = even + 1
            pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)
            cos = tl.load(cos_sin_cache_ptr + pos * rope_dim + pair).to(tl.float32)
            sin = tl.load(cos_sin_cache_ptr + pos * rope_dim + rope_dim // 2 + pair).to(
                tl.float32
            )
            q_even = tl.load(q_base + even).to(tl.float32) * inv_rms
            q_odd = tl.load(q_base + odd).to(tl.float32) * inv_rms
            tl.store(
                out_base + even,
                (q_even * cos - q_odd * sin).to(q_out_ptr.type.element_ty),
            )
            tl.store(
                out_base + odd,
                (q_even * sin + q_odd * cos).to(q_out_ptr.type.element_ty),
            )
        else:
            tl.store(out_base + offsets, 0.0)
        return

    # The extra head slot performs KV RoPE, UE8M0 quantization and insertion.
    if token_idx >= num_tokens_insert:
        return
    cache_slot = tl.load(slot_mapping_ptr + token_idx)
    if cache_slot < 0:
        return

    block_idx = cache_slot // cache_block_size
    pos_in_block = cache_slot % cache_block_size
    cache_block = k_cache_ptr + block_idx.to(tl.int64) * block_stride
    token_data = cache_block + pos_in_block * token_data_size
    scale_dim: tl.constexpr = nope_dim // quant_block + 1
    token_scales = (
        cache_block + cache_block_size * token_data_size + pos_in_block * scale_dim
    )
    kv_base = kv_ptr + token_idx * head_dim

    for block in tl.static_range(nope_dim // quant_block):
        offsets = block * quant_block + tl.arange(0, quant_block)
        values = tl.load(kv_base + offsets).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(values), axis=0), 1e-4)
        exponent = tl.ceil(tl.log2(amax / fp8_max))
        scale = tl.exp2(exponent)
        quant = tl.clamp(values / scale, -fp8_max, fp8_max).to(tl.float8e4nv)
        tl.store(token_data + offsets, quant.to(tl.uint8, bitcast=True))
        encoded = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0)
        tl.store(token_scales + block, encoded.to(tl.uint8))
    tl.store(token_scales + scale_dim - 1, 0)

    pair = tl.arange(0, rope_dim // 2)
    even = nope_dim + pair * 2
    odd = even + 1
    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)
    cos = tl.load(cos_sin_cache_ptr + pos * rope_dim + pair).to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * rope_dim + rope_dim // 2 + pair).to(
        tl.float32
    )
    kv_even = tl.load(kv_base + even).to(tl.float32)
    kv_odd = tl.load(kv_base + odd).to(tl.float32)
    rope_out = (token_data + nope_dim).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_out + pair * 2, kv_even * cos - kv_odd * sin)
    tl.store(rope_out + pair * 2 + 1, kv_even * sin + kv_odd * cos)


def qnorm_rope_kv_quant_insert_out(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    padded_heads: int,
    eps: float,
    cache_block_size: int,
    q_out: torch.Tensor,
) -> torch.Tensor:
    """Write padded Q and packed KV cache without allocating a padded Q."""
    assert q.shape[-1] == 512 and kv.shape[-1] == 512
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert q_out.shape == (q.shape[0], padded_heads, q.shape[-1])
    assert q_out.dtype == q.dtype and q_out.is_contiguous()

    grid = (q.shape[0], padded_heads + 1)
    _qnorm_rope_kv_quant_insert_out_kernel[grid](
        q,
        kv,
        q_out,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        q.shape[0],
        slot_mapping.shape[0],
        eps=eps,
        num_heads=q.shape[1],
        padded_heads=padded_heads,
        head_dim=512,
        nope_dim=448,
        rope_dim=64,
        quant_block=64,
        cache_block_size=cache_block_size,
        token_data_size=576,
        block_stride=k_cache.stride(0),
        fp8_max=torch.finfo(torch.float8_e4m3fn).max,
        num_warps=1,
    )
    return q_out

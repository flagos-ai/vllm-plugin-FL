# Copyright (c) 2026 BAAI. All rights reserved.
"""SM80/PPU dequantize-and-gather kernel for the DeepSeek-V4 paged K cache.

Ported verbatim from the T-Head vendor fork
(v1/attention/ops/deepseek_v4_ops/cache_utils.py): the upstream triton
kernel bitcasts uint8 -> tl.float8e4nv, which PPU triton cannot compile;
this variant decodes E4M3FN in software (_decode_e4m3fn).
"""

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _decode_e4m3fn(u):
    """Decode an E4M3FN byte (uint8) to fp32 using only uint/int/fp ops.

    Triton on SM80 cannot compile `tl.float8e4nv`, so we never load the
    FP8 dtype directly — we load uint8 and decode in software here. The
    expansion is ~6 ops per element, dwarfed by the surrounding matmul.

    E4M3FN: 1 sign + 4 exp (bias 7) + 3 mantissa.  No infinities.
    Subnormal (exp=0): value = (-1)^s * (mant/8) * 2^(1 - 7)
    Normal           : value = (-1)^s * (1 + mant/8) * 2^(exp - 7)
    NaN at 0x7F/0xFF is decoded numerically as ±480 — sparse-MLA inputs
    never hit this so the loss of NaN propagation is acceptable.
    """
    sign = u >> 7
    exp_bits = ((u >> 3) & 0x0F).to(tl.int32)
    mant = (u & 0x07).to(tl.int32)
    is_normal = exp_bits != 0
    sign_f = tl.where(sign != 0, -1.0, 1.0)
    mant_f = tl.where(
        is_normal,
        (8 + mant).to(tl.float32) * 0.125,
        mant.to(tl.float32) * 0.125,
    )
    # Subnormals: real exponent = 1 - bias.
    eff_exp = tl.where(is_normal, exp_bits, 1)
    factor = tl.exp2((eff_exp - 7).to(tl.float32))
    return sign_f * mant_f * factor



@triton.jit
def _dequantize_and_gather_k_kernel_sm80(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    output_dim: tl.constexpr,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
):
    """SM80 variant of `_dequantize_and_gather_k_kernel`. Replaces the
    `tl.float8e4nv` bitcast with the software `_decode_e4m3fn` decoder
    from PR 38476 (uint8 → fp32 in ~6 ops/elt). Same launch shape and
    layout as the SM90+ kernel."""
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block
            if qblock_start < fp8_dim:
                offsets = qblock_start + tl.arange(0, quant_block)
                mask = offsets < fp8_dim
                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)
                # Software fp8e4nv → fp32 (no tl.float8e4nv reference).
                x_float = _decode_e4m3fn(x_uint8)
                encoded_scale = tl.load(token_scale_ptr + qblock_idx)
                exponent = encoded_scale.to(tl.float32) - 127.0
                scale = tl.exp2(exponent)
                x_dequant = x_float * scale
                tl.store(
                    output_row_ptr + offsets,
                    x_dequant.to(tl.bfloat16),
                    mask=mask,
                )

        bf16_output_offset = fp8_dim
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def _dequantize_and_gather_k_cache_triton(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """SM80 Triton dispatch — uses _decode_e4m3fn instead of the
    fp8e4nv bitcast that Triton refuses to compile on SM80."""
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    FP8_MAX = 448.0
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel_sm80[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        output_dim=512,
        fp8_max=FP8_MAX,
        n_quant_blocks=7,
    )



def _dequant_gather_sm80_op(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    _dequantize_and_gather_k_cache_triton(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )


def _dequant_gather_sm80_op_fake(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="fl_deepseek_v4_dequant_gather_sm80",
    op_func=_dequant_gather_sm80_op,
    mutates_args=["out"],
    fake_impl=_dequant_gather_sm80_op_fake,
)


def dequantize_and_gather_k_cache_ppu(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
) -> None:
    """Drop-in replacement for upstream dequantize_and_gather_k_cache.

    use_fnuz is accepted for signature parity; the triton encoders on this
    stack are all OCP E4M3FN, so it is ignored.
    """
    torch.ops.vllm.fl_deepseek_v4_dequant_gather_sm80(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )

# Copyright (c) 2026 BAAI. All rights reserved.
"""Pure-PyTorch E4M3 (FP8) bit-unpack + bit-pack helpers.

torch-lpu has no ``float8`` compute/cast (casts to fp32/bf16 return garbage),
so every FP8 encode/decode on LPU goes through these manual bit helpers.
Inputs are raw ``uint8`` or a native ``float8_e4m3fn`` tensor viewed to uint8
first — the view is byte-preserving; the cast kernel is broken. Semantics
match ``torch.float8_e4m3fn.float()`` for all 256 byte patterns;
``quant_e4m3_blocks`` is the inverse: E8M0 block scales matching the evas
compressed-KV write kernel (see its docstring below).

"""
from __future__ import annotations

import torch


def dequant_e4m3(u8: torch.Tensor) -> torch.Tensor:
    """Decode raw E4M3 bytes to float32 (uint8 or native float8_e4m3fn).

    Args:
        u8: tensor holding E4M3 bit patterns — either dtype ``uint8``, or a
            native ``torch.float8_e4m3fn`` tensor (viewed to uint8 first;
            byte-identical — verified on LPU 2026-09-10).  Any shape.
    Returns:
        Same-shape float32 tensor. NaN bytes (e==15, m==7) yield NaN.
    """
    if u8.dtype == torch.float8_e4m3fn:
        u8 = u8.contiguous().view(torch.uint8)
    s = (u8 >> 7).float()
    e = (u8 >> 3) & 0x0F
    m = u8 & 0x07
    ef = e.float()
    mf = m.float()
    sf = 1 - 2 * s

    nan_mask = (e == 0b1111) & (m == 0b111)
    e15 = (e == 0b1111) & ~nan_mask  # e==15, m in [0,6] -> finite, max 448
    e0 = (e == 0) & ~(e == 0b1111)  # subnormal

    val = torch.zeros(u8.shape, dtype=torch.float32, device=u8.device)
    normal = ~(e == 0b1111) & ~e0
    val = torch.where(normal, sf * (2.0 ** (ef - 7)) * (1 + mf / 8), val)
    val = torch.where(e0, sf * (2.0 ** (-6)) * (mf / 8), val)
    val = torch.where(e15, sf * 256.0 * (1 + mf / 8), val)
    val = torch.where(nan_mask, torch.nan, val)
    return val


def _encode_e4m3(q: torch.Tensor) -> torch.Tensor:
    """Encode E4M3-grid fp32 values to uint8 bit patterns (round-to-nearest).

    ``q`` values are clamped to [-448, 448] (E4M3 max finite); NaN -> 0x7F.
    Uses the exact fp32 exponent bits (no ``log2`` float hazard) to place each
    value on the E4M3 grid ``2^(e-7) * (1 + m/8)`` (or the subnormal grid
    ``2^-6 * m/8`` for ``|q| < 2^-6``), then rounds the mantissa field.  A field
    that rounds up to 8 does not fit the 3-bit mantissa and carries into the
    exponent (``2^e * (1 + 8/8) == 2^(e+1)``); rounding it down to 7 instead
    would place the value one grid step low.

    Args:
        q: fp32/bf16 tensor of E4M3-representable magnitudes (any shape).
    Returns:
        Same-shape uint8 tensor of E4M3 bit patterns (matches
        ``q.to(torch.float8_e4m3fn).view(torch.uint8)`` for grid values).
    """
    q = q.float().contiguous()
    qbits = q.view(torch.int32)
    s_bit = (qbits >> 31) & 1  # sign bit (catches -0.0, which q < 0 misses)
    a = q.abs()
    is_nan = torch.isnan(q)
    a0 = a.masked_fill(is_nan, 0.0)

    # Exact fp32 biased exponent (E4M3 grid values are fp32-exact, >= 2^-9).
    xbits = a0.view(torch.int32)
    e = ((xbits >> 23) & 0xFF).to(torch.float32) - 127.0

    is_sub = a0 < (2.0 ** -6)  # E4M3 subnormal range: (0, 2^-6)
    is_zero = a0 == 0.0
    # normal: a = 2^e * (1 + m/8) -> m = (a/2^e - 1) * 8
    m_norm = (a0 / torch.exp2(e) - 1.0) * 8.0
    # subnormal: a = 2^-6 * (m/8) -> m = 512 * a
    m_sub = a0 * 512.0
    m = torch.where(is_sub, m_sub, m_norm)
    m = m.round()

    # Mantissa overflow: a rounded field of 8 is the next binade, so carry it
    # into the exponent rather than clamping to 7 (which would round toward
    # zero by a full grid step).  The subnormal branch carries the same way --
    # a field of 8 there is exactly 2^-6, the first normal -- so base exponent
    # 0 + 1 lands on the right code point.  The overflow is subtracted
    # arithmetically (m - 8*carry) instead of a where + zeros_like: the whole
    # encode runs per decode step, and on the LPU backend every
    # torch.where-with-a-python-scalar turns into a scalar_tensor materialize +
    # cast chain, which the stack attribution showed as the hot tail of this
    # function.
    carry = (m == 8.0).to(torch.float32)
    m = m - 8.0 * carry

    # (e+7) * keep: the kept lanes multiply by 1.0 (bit-exact), the zero and
    # subnormal lanes land on 0.0 exactly as the wheres they replace -- the
    # sign of a flushed zero is irrelevant because the int32 cast of -0.0 is 0.
    e_field = (e + 7.0) * (~is_zero & ~is_sub).to(torch.float32)
    e_field = (e_field + carry).clamp(0, 15)
    m = m * (~is_zero).to(torch.float32)

    bits = (e_field.to(torch.int32) << 3) | m.to(torch.int32)
    bits = bits | (s_bit << 7)
    # masked_fill with a Scalar, not where + torch.tensor: same 0x7F mapping,
    # no per-call scalar construction.  (The aten cast itself is NOT an
    # alternative: it exists in libtorch_lpu but emits garbage bytes on device
    # -- 448.0 -> 0xa0, NaN -> 0xa0 against 0x7e/0x7f -- so this bit-level
    # encoder is load-bearing, not a style choice.)
    bits = bits.masked_fill(is_nan, 0x7F)
    return bits.to(torch.uint8)


def quant_e4m3_blocks(
    x: torch.Tensor, block: int = 64, fp8_max: float = 448.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize to E4M3 uint8 bytes + E8M0 block scales (per-row blocks).

    Mirrors the evas compressed-KV write kernel convention
    (``_upstream/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py``):

      - input round-tripped fp32 -> bf16 -> fp32 (matches the reference kernel)
      - per ``block``-wide column block: ``amax = max(|x|, 1e-4)``
      - ``scale = 2**ceil(log2(amax / fp8_max))``, stored E8M0 byte ``exp + 127``
      - ``q = clamp(x / scale, -fp8_max, fp8_max)`` encoded as E4M3 bytes

    The last step is a DIRECT cast, matching the kernel ``tl.clamp(values / scale,
    +-FP8_MAX).to(tl.float8e4nv)``.  ``_encode_e4m3`` already rounds to the E4M3
    grid, so the ``torch.round`` this used to carry rounded to INTEGER values
    first, and the encoder then saw only whole numbers: the output collapsed to
    110 of the 254 finite E4M3 code points, against 232 for the direct cast,
    discarding every in-range mantissa (measured on CPU, 896x128 bf16 input,
    seed 21: 9.44 per cent of the 114688 bytes differ, and 187 of those are the
    ``|x / scale| <= 0.5`` values it also flushes to zero -- 185 strictly inside
    the half and 2 exactly on the tie, which round-half-to-even also sends to
    zero -- so the loss is precision, not just the subnormal tail).

    ``quant_act_per_token`` and ``quant_act_per_block`` below carried the same
    ``.round()``; so did the ``o_proj`` twin
    (``fused_inv_rope_fp8_quant.py``).  All were ported to the direct cast in
    one change, so no seat of this defect remains on the quant paths.  Their
    references differ from this one and each was checked separately before the
    ``.round()`` was dropped: ``quant_act_per_token`` mirrors vLLM
    ``scaled_fp8_quant`` (a C++ ``static_cast<fp8_t>``),
    ``quant_act_per_block`` mirrors vLLM ``_per_token_group_quant_fp8``
    (``tl.clamp(y / y_s, fp8_min, fp8_max).to(elem_ty)``), and the o_proj twin
    mirrors the vendored kernel (``_upstream/fused_inv_rope_fp8_quant.py``)
    which writes ``tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(float8e4nv)``.
    All are direct casts with no integer round.

    Args:
        x: [N, K] bf16/fp32 tensor to quantize.
        block: quantization block width (64 for DeepSeek-V4 nope=448 -> 7).
        fp8_max: E4M3 max finite magnitude (448).
    Returns:
        (q, s): q [N, K] uint8 E4M3 bytes; s [N, K // block] uint8 E8M0 scale
        bytes (``2**(s-127)``). ``K % block == 0`` required.
    """
    n, k = x.shape
    assert k % block == 0, f"K={k} not divisible by quant block {block}"
    nb = k // block
    xf = x.float()
    # Reference kernel casts fp32 -> bf16 -> fp32 before quant.
    xf = xf.to(torch.bfloat16).to(torch.float32)

    amax = xf.view(n, nb, block).abs().amax(dim=-1)  # [n, nb]
    amax = amax.clamp_min(1e-4)
    exp = torch.ceil(torch.log2(amax / fp8_max))
    exp = exp.clamp(-127, 128)
    scale = torch.exp2(exp)  # [n, nb] fp32
    q = (xf.view(n, nb, block) / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    q_bytes = _encode_e4m3(q.reshape(n, k))
    s_bytes = (exp + 127.0).to(torch.uint8)
    return q_bytes, s_bytes


def quant_act_per_token(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token FP8 E4M3 activation quant (uint8 bytes + fp32 scale).

    Mirrors ``scaled_fp8_quant(use_per_token_if_dynamic=True)`` with the
    weight-side E4M3 byte convention (LPU has no float8): one scale per token
    ``amax/448``, values clamped and packed to E4M3 bytes on the grid.  Used by
    the W8A8 MoE path (activation quantized + weight FP8); dequant on the
    compute side is ``dequant_e4m3(q) * scale``.

    The value is handed to :func:`_encode_e4m3` as ``x / scale``, with no
    ``torch.round`` in between -- the reference ``scaled_fp8_quant`` is a
    ``static_cast<fp8_t>`` (round to the E4M3 grid, nothing else) and the
    encoder already rounds to that grid.  A round to INTEGER values first would
    quantize twice, capping the output at 110 of the 254 finite E4M3 code
    points (measured on CPU, seed 1: 110 on both shapes the test uses, against
    176 for (16, 512) and 226 for (64, 1024) with the direct cast) and changing
    10.4 and 10.6 per cent of the bytes.

    Args:
        A: [M, K] bf16/fp32 activations (any 2-D row layout, incl. the
            flattened [M*top_k, K] expert slots of the down GEMM).
    Returns:
        (q, scale): q [M, K] uint8 E4M3 bytes; scale [M, 1] fp32 per token.
    """
    amax = A.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = (amax / 448.0).to(torch.float32)  # [M, 1]
    q = (A.float() / scale).clamp(-448, 448)
    return _encode_e4m3(q), scale


def quant_act_per_block(
    A: torch.Tensor, block: int = 128, fp8_max: float = 448.0, eps: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-(token, block) FP8 E4M3 activation quant (uint8 + fp32 scale).

    Mirrors ``per_token_group_quant_fp8`` -- the block-quant entry the vLLM MoE
    activation flow takes once a ``block_shape`` is declared -- under the LPU
    container contract.  Same E4M3 encode as :func:`quant_act_per_token`; only
    the reduction axis of ``amax`` differs: one scale per ``block``-wide column
    group instead of one per row.  The distinction is structural, not a
    refinement: a block-fp8 GEMM pairs the activation scale with the weight
    block, so a per-row scale cannot express its layout.

      - per group of ``block`` channels: ``amax = max(|x|, eps)``
      - ``scale = amax / fp8_max`` (fp32, no power-of-two rounding: vLLM's
        ``per_token_group_quant_fp8`` is only E8M0 when ``use_ue8m0`` is asked
        for, and the MoE activation path never asks)
      - ``q = clamp(x / scale, -fp8_max, fp8_max)`` as E4M3 bytes, a DIRECT
        cast: the reference ``_per_token_group_quant_fp8`` kernel writes
        ``tl.clamp(y / y_s, fp8_min, fp8_max).to(elem_ty)`` and the encoder
        already rounds to the E4M3 grid, so a ``torch.round`` here would
        quantize twice, capping the output at 110 of the 254 finite E4M3 code
        points (measured on CPU, seed 1, on the (16, 512) shape below: 110
        against 172 for the direct cast) and changing 9.55 per cent of the
        bytes

    Args:
        A: [..., K] bf16/fp32 activations, K the contraction axis (e.g. the
            [M, H] hidden states or the flattened [M*top_k, I] expert slots).
        block: quantization block width along K (128 = the weight block).
        fp8_max: E4M3 max finite magnitude (448).
        eps: floor on the group amax so an all-zero group cannot divide by
            zero; vLLM clamps its group amax with the same value.
    Returns:
        (q, scale): q [..., K] uint8 E4M3 bytes; scale [..., K // block] fp32.
        ``K % block == 0`` required.
    """
    k = A.shape[-1]
    assert k % block == 0, f"K={k} not divisible by quant block {block}"
    nb = k // block
    xf = A.float().reshape(-1, nb, block)
    amax = xf.abs().amax(dim=-1).clamp_min(eps)  # [M, nb]
    scale = (amax / fp8_max).to(torch.float32)
    q = (xf / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    return _encode_e4m3(q.reshape(A.shape)), scale.reshape(A.shape[:-1] + (nb,))

"""On-demand INT8 gate/up for large BF16 prefill batches.

The BF16 gate/up weight and its FP32 per-output scale remain resident. Each
call quantizes the activation rows and packs the weight into the layout required
by ``aten._int_mm``; all temporary tensors are owned by this call and use the
current stream.
"""

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch.library import Library

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

PACK_N = 64
PACK_K = 64
PACK_WARPS = 4
PACK_STAGES = 1
ROWMAJOR_BN = 64
ROWMAJOR_BK = 64
ROWMAJOR_WARPS = 4
ROWMAJOR_STAGES = 1
EPILOGUE_BLOCK = 1024
EPILOGUE_WARPS = 4
logger = init_logger("vllm.vllm_fl.prefill_gateup")
_selected_path_logged = False
_fallback_path_logged = False
_rowmajor_selected_path_logged = False
_rowmajor_fallback_path_logged = False
_runtime_dispatch_logging = False
_small_m_activation: Callable[[torch.Tensor], torch.Tensor] | None = None


def bind_small_m_activation(
    activation: Callable[[torch.Tensor], torch.Tensor] | None,
) -> None:
    """Bind vLLM's compiled native SiLU-and-multiply for the BF16 fallback."""
    if activation is not None and not callable(activation):
        raise TypeError("small-M activation must be callable")
    global _small_m_activation
    _small_m_activation = activation


def enable_runtime_dispatch_logging() -> None:
    """Enable the one-time selected-path log after model-load gating is done."""
    global _runtime_dispatch_logging
    _runtime_dispatch_logging = True


@triton.jit
def _quant_pad(
    X, Q, S, M: tl.constexpr, K: tl.constexpr, XM: tl.constexpr, BK: tl.constexpr
):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    if row < M:
        x = tl.load(X + row * XM + k, k < K, 0).to(tl.float32)
        scale = tl.math.div_rn(tl.maximum(tl.max(tl.abs(x), 0), 1e-8), 127.0)
        value = tl.math.div_rn(x, scale)
        q = tl.where(value >= 0, tl.floor(value + 0.5), tl.ceil(value - 0.5))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(Q + row * K + k, q.to(tl.int8), k < K)
        tl.store(S + row, scale)
    else:
        tl.store(Q + row * K + k, 0, k < K)
        tl.store(S + row, 0.0)


@triton.jit
def _pack_quant_transpose(
    W, SW, Q, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (n[:, None] < N) & (k[None, :] < K)
    w = tl.load(W + n[:, None] * K + k[None, :], mask, 0).to(tl.float32)
    scale = tl.load(SW + n, n < N, 1.0)
    value = tl.math.div_rn(w, scale[:, None])
    lo = tl.floor(value)
    frac = value - lo
    odd = (lo.to(tl.int32) % 2) != 0
    q = tl.where(frac > 0.5, lo + 1.0, lo)
    q = tl.where(frac == 0.5, tl.where(odd, lo + 1.0, lo), q)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(Q + k[None, :] * N + n[:, None], q.to(tl.int8), mask)


@triton.jit
def _quant_weight_rowmajor(
    W, SW, Q, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (n[:, None] < N) & (k[None, :] < K)
    w = tl.load(W + n[:, None] * K + k[None, :], mask, 0).to(tl.float32)
    scale = tl.load(SW + n, n < N, 1.0)
    value = tl.math.div_rn(w, scale[:, None])
    lower = tl.floor(value)
    fraction = value - lower
    odd = (lower.to(tl.int32) % 2) != 0
    q = tl.where(fraction > 0.5, lower + 1.0, lower)
    q = tl.where(fraction == 0.5, tl.where(odd, lower + 1.0, lower), q)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(Q + n[:, None] * K + k[None, :], q.to(tl.int8), mask)


@triton.jit
def _gateup_epilogue(
    C, SA, SW, Z, M: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // H, i % H
    mask = i < M * H
    base = row * (2 * H) + col
    gate_acc = tl.load(C + base, mask, 0).to(tl.float32)
    up_acc = tl.load(C + base + H, mask, 0).to(tl.float32)
    sa = tl.load(SA + row, mask, 0)
    gate_scale = tl.load(SW + col, mask, 0)
    up_scale = tl.load(SW + H + col, mask, 0)
    gate = (gate_acc * sa * gate_scale).to(tl.bfloat16).to(tl.float32)
    up = (up_acc * sa * up_scale).to(tl.bfloat16).to(tl.float32)
    z = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(Z + i, z.to(tl.bfloat16), mask)


def weight_scales(weight: torch.Tensor) -> torch.Tensor:
    """Return the FP32 symmetric per-output scale used by the on-demand pack."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("weight_scales requires a BF16 [N,K] weight")
    return weight.float().abs().amax(dim=1).clamp_min(1e-8) / 127.0


def _activation_rows_padded(m: int, min_rows: int = 16) -> int:
    if min_rows <= 0 or min_rows % 16:
        raise ValueError("min_rows must be a positive multiple of 16")
    return max(min_rows, triton.cdiv(m, 16) * 16)


def quantize_activation_padded(
    x: torch.Tensor, *, min_rows: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows and zero-fill to the requested multiple-of-16 minimum."""
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or x.stride(1) != 1
        or min(x.shape) <= 0
    ):
        raise ValueError("requires nonempty CUDA BF16 [M,K] with unit inner stride")
    m, k = x.shape
    mpad = _activation_rows_padded(m, min_rows)
    q = torch.empty((mpad, k), dtype=torch.int8, device=x.device)
    sx = torch.empty((mpad,), dtype=torch.float32, device=x.device)
    bk = triton.next_power_of_2(k)
    _quant_pad[(mpad,)](x, q, sx, m, k, x.stride(0), bk, num_warps=4, num_stages=1)
    return q, sx


def pack_weight_transpose(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize BF16 [N,K] by scale and write contiguous INT8 [K,N]."""
    if (
        weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or scale.dtype != torch.float32
        or scale.shape != (weight.shape[0],)
        or not scale.is_contiguous()
        or weight.device != scale.device
    ):
        raise ValueError("requires contiguous CUDA BF16 [N,K] and FP32 scale [N]")
    n, k = weight.shape
    if min(n, k) <= 0 or n % 16 or k % 16:
        raise ValueError("INT8 gate/up weight N and K must be positive multiples of 16")
    q = torch.empty((k, n), dtype=torch.int8, device=weight.device)
    _pack_quant_transpose[(triton.cdiv(k, PACK_K), triton.cdiv(n, PACK_N))](
        weight,
        scale,
        q,
        n,
        k,
        PACK_N,
        PACK_K,
        num_warps=PACK_WARPS,
        num_stages=PACK_STAGES,
    )
    return q


def quantize_weight_rowmajor(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize BF16 [N,K] per output row into contiguous INT8 [N,K]."""
    if (
        weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or scale.dtype != torch.float32
        or scale.shape != (weight.shape[0],)
        or not scale.is_contiguous()
        or weight.device != scale.device
    ):
        raise ValueError("requires contiguous CUDA BF16 [N,K] and FP32 scale [N]")
    n, k = weight.shape
    if min(n, k) <= 0 or n % 16 or k % 16:
        raise ValueError("INT8 gate/up weight N and K must be positive multiples of 16")
    q = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    _quant_weight_rowmajor[(triton.cdiv(n, ROWMAJOR_BN), triton.cdiv(k, ROWMAJOR_BK))](
        weight,
        scale,
        q,
        n,
        k,
        ROWMAJOR_BN,
        ROWMAJOR_BK,
        num_warps=ROWMAJOR_WARPS,
        num_stages=ROWMAJOR_STAGES,
    )
    return q


def _prefill_gateup_from_rowmajor_qweight(
    x: torch.Tensor,
    qx: torch.Tensor,
    sx: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Run INT8 matmul and the fused gate/up epilogue from row-major QW.

    ``qweight`` is the caller-owned contiguous INT8 [N,K] tensor. Its transpose
    is a metadata-only [K,N] view accepted by ``aten._int_mm``; this helper
    consumes caller-allocated padded activation and scales, then allocates an
    INT32 accumulator [Mpad,N] and BF16 result [M,N/2].
    """
    m, k = x.shape
    n = qweight.shape[0]
    qwt = qweight.t()
    if qwt.is_contiguous() or tuple(qwt.stride()) != (1, k):
        raise RuntimeError(
            f"unexpected row-major QW transpose view: stride={qwt.stride()}"
        )

    accum = torch.empty((qx.shape[0], n), dtype=torch.int32, device=x.device)
    torch.ops.aten._int_mm.out(qx, qwt, out=accum)
    h = n // 2
    z = torch.empty((m, h), dtype=torch.bfloat16, device=x.device)
    _gateup_epilogue[(triton.cdiv(m * h, EPILOGUE_BLOCK),)](
        accum,
        sx,
        scale,
        z,
        m,
        h,
        EPILOGUE_BLOCK,
        num_warps=EPILOGUE_WARPS,
        num_stages=1,
    )
    return z


def prefill_gateup_int8(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Compute INT8 gate/up and BF16 SiLU product for a large prefill batch."""
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or torch.version.hip is not None
        or x.stride(1) != 1
        or weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or x.device != weight.device
        or scale.device != x.device
        or scale.dtype != torch.float32
        or not scale.is_contiguous()
        or weight.shape[0] % 2
        or weight.shape[1] != x.shape[1]
        or scale.shape != (weight.shape[0],)
        or min(*x.shape, *weight.shape) <= 0
    ):
        raise ValueError(
            "requires compatible CUDA BF16 input/weight and FP32 gate/up scales"
        )
    m, k = x.shape
    n = weight.shape[0]
    if n % 16 or k % 16:
        raise ValueError("INT8 gate/up output N and K must be multiples of 16")

    qx, sx = quantize_activation_padded(x)
    qw = pack_weight_transpose(weight, scale)
    accum = torch.empty((qx.shape[0], n), dtype=torch.int32, device=x.device)
    torch.ops.aten._int_mm.out(qx, qw, out=accum)
    h = n // 2
    z = torch.empty((m, h), dtype=torch.bfloat16, device=x.device)
    _gateup_epilogue[(triton.cdiv(m * h, EPILOGUE_BLOCK),)](
        accum,
        sx,
        scale,
        z,
        m,
        h,
        EPILOGUE_BLOCK,
        num_warps=EPILOGUE_WARPS,
        num_stages=1,
    )
    return z


def prefill_gateup_int8_rowmajor(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Compute INT8 gate/up from row-major QW and consume its transpose view."""
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or torch.version.hip is not None
        or x.stride(1) != 1
        or weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or x.device != weight.device
        or scale.device != x.device
        or scale.dtype != torch.float32
        or not scale.is_contiguous()
        or weight.shape[0] % 2
        or weight.shape[1] != x.shape[1]
        or scale.shape != (weight.shape[0],)
        or min(*x.shape, *weight.shape) <= 0
    ):
        raise ValueError(
            "requires compatible CUDA BF16 input/weight and FP32 gate/up scales"
        )
    m, k = x.shape
    n = weight.shape[0]
    if n % 16 or k % 16:
        raise ValueError("INT8 gate/up output N and K must be multiples of 16")

    qx, sx = quantize_activation_padded(x)
    qw = quantize_weight_rowmajor(weight, scale)
    return _prefill_gateup_from_rowmajor_qweight(x, qx, sx, qw, scale)


def prefill_gateup_int8_prequantized_rowmajor(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Compute gate/up from persistent row-major INT8 QW without repacking.

    Accepts BF16 ``x`` [M,K], contiguous INT8 ``qweight`` [N,K] stored in
    output-row-major order, and contiguous FP32 ``scale`` [N]. Scales are the
    per-output-row dequantization scales used by the epilogue. The call
    quantizes only ``x``, consumes ``qweight.t()`` as a strided view in
    ``aten._int_mm``, then runs the existing fused BF16 gate/up epilogue.
    It does not quantize, transpose-copy, or reconstruct the weights.
    """
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or torch.version.hip is not None
        or x.stride(1) != 1
    ):
        raise ValueError("requires CUDA BF16 [M,K] with unit inner stride")
    if (
        qweight.ndim != 2
        or qweight.dtype != torch.int8
        or not qweight.is_cuda
        or not qweight.is_contiguous()
    ):
        raise ValueError("requires contiguous CUDA INT8 QW [N,K]")
    if scale.ndim != 1 or scale.dtype != torch.float32 or not scale.is_contiguous():
        raise ValueError("requires contiguous FP32 scale [N]")
    if x.device != qweight.device or scale.device != x.device:
        raise ValueError("x, QW, and scale must be on the same CUDA device")
    if (
        x.shape[0] <= 0
        or x.shape[1] <= 0
        or qweight.shape[0] <= 0
        or qweight.shape[1] <= 0
        or qweight.shape[0] % 2
        or qweight.shape[1] != x.shape[1]
    ):
        raise ValueError(
            "requires nonempty compatible x [M,K] and QW [N,K] with even N"
        )
    if scale.shape != (qweight.shape[0],):
        raise ValueError("scale must have shape [N] for QW [N,K]")
    _, k = x.shape
    n = qweight.shape[0]
    if n % 16 or k % 16:
        raise ValueError("INT8 gate/up output N and K must be multiples of 16")

    # aten._int_mm's CUDA implementation needs more than 16 left-hand rows.
    qx, sx = quantize_activation_padded(x, min_rows=32)
    return _prefill_gateup_from_rowmajor_qweight(x, qx, sx, qweight, scale)


def _prefill_gateup_int8_impl(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, min_tokens: int
) -> torch.Tensor:
    global _selected_path_logged, _fallback_path_logged
    m = x.shape[0]
    if m == 0:
        return x.new_empty((0, weight.shape[0] // 2))
    if m < min_tokens:
        if _runtime_dispatch_logging and not _fallback_path_logged:
            logger.info(
                "Prefill gate/up BF16 fallback first call: M=%d threshold=%d",
                m,
                min_tokens,
            )
            _fallback_path_logged = True
        projection = torch.mm(x, weight.t())
        if _small_m_activation is None:
            raise RuntimeError("BF16 fallback activation was not bound before dispatch")
        return _small_m_activation(projection)
    if _runtime_dispatch_logging and not _selected_path_logged:
        logger.info(
            "Prefill gate/up INT8 selected path first call: M=%d threshold=%d",
            m,
            min_tokens,
        )
        _selected_path_logged = True
    return prefill_gateup_int8(x, weight, scale)


def _prefill_gateup_int8_fake(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, min_tokens: int
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


def _prefill_gateup_int8_rowmajor_impl(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, min_tokens: int
) -> torch.Tensor:
    global _rowmajor_selected_path_logged, _rowmajor_fallback_path_logged
    m = x.shape[0]
    if m == 0:
        return x.new_empty((0, weight.shape[0] // 2))
    if m < min_tokens:
        if _runtime_dispatch_logging and not _rowmajor_fallback_path_logged:
            logger.info(
                "Prefill gate/up BF16 fallback first call: layout=rowmajor M=%d threshold=%d",
                m,
                min_tokens,
            )
            _rowmajor_fallback_path_logged = True
        projection = torch.mm(x, weight.t())
        if _small_m_activation is None:
            raise RuntimeError("BF16 fallback activation was not bound before dispatch")
        return _small_m_activation(projection)
    if _runtime_dispatch_logging and not _rowmajor_selected_path_logged:
        logger.info(
            "Prefill gate/up INT8 row-major selected path first call: M=%d threshold=%d",
            m,
            min_tokens,
        )
        _rowmajor_selected_path_logged = True
    return prefill_gateup_int8_rowmajor(x, weight, scale)


def _prefill_gateup_int8_rowmajor_fake(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, min_tokens: int
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


def _prefill_gateup_int8_prequantized_rowmajor_impl(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if x.ndim == 2 and x.shape[0] == 0:
        return x.new_empty((0, qweight.shape[0] // 2))
    return prefill_gateup_int8_prequantized_rowmajor(x, qweight, scale)


def _prefill_gateup_int8_prequantized_rowmajor_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], qweight.shape[0] // 2))


_vllm_fl_lib = Library("vllm_fl", "FRAGMENT")
direct_register_custom_op(
    "prefill_gateup_int8",
    _prefill_gateup_int8_impl,
    fake_impl=_prefill_gateup_int8_fake,
    target_lib=_vllm_fl_lib,
)
direct_register_custom_op(
    "prefill_gateup_int8_rowmajor",
    _prefill_gateup_int8_rowmajor_impl,
    fake_impl=_prefill_gateup_int8_rowmajor_fake,
    target_lib=_vllm_fl_lib,
)
direct_register_custom_op(
    "prefill_gateup_int8_prequantized_rowmajor",
    _prefill_gateup_int8_prequantized_rowmajor_impl,
    fake_impl=_prefill_gateup_int8_prequantized_rowmajor_fake,
    target_lib=_vllm_fl_lib,
)

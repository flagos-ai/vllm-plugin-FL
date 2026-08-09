# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU fix for _per_token_group_quant_fp8 / _per_token_group_quant_fp8_colmajor.

GCU hardware limits grid.x to 65535 and grid.y/grid.z to 255.  The upstream
launcher uses ``grid = (M,)`` where ``M = numel // group_size``, which exceeds
the limit for large tensors (e.g. 8192 × 4096 ÷ 128 = 262144).

This patch converts the 1-D grid into a 2-D grid that respects GCU limits,
following the same pattern as ``fused_recurrent_packed_decode.py``.
"""

from __future__ import annotations

import logging

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

GCU_MAX_GRID_X = 65535
GCU_MAX_GRID_YZ = 255
_patched = False


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _gcu_grid(total: int) -> tuple[int, int]:
    """Split *total* work-items into a 2-D grid respecting GCU hardware limits."""
    grid_x = min(total, GCU_MAX_GRID_X)
    grid_y = triton.cdiv(total, grid_x)
    if grid_y > GCU_MAX_GRID_YZ:
        grid_y = GCU_MAX_GRID_YZ
        grid_x = triton.cdiv(total, grid_y)
        grid_x = min(grid_x, GCU_MAX_GRID_X)
    return grid_x, grid_y


# ---------------------------------------------------------------------------
# GCU-compatible Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _per_token_group_quant_fp8_gcu(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.

    GCU variant: uses 2-D grid (program_id 0 + 1) so that grid.x never
    exceeds the hardware limit of 65535.
    """
    groups_per_row = y_num_columns // group_size

    # Map the 2-D program id to a flat group id.
    g_id = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    # Use multiply-by-reciprocal instead of division to match PyTorch's
    # tensor/scalar division precision (GPU fast-division for constexpr
    # divisors can introduce 1-ULP error that flips FP8 quantization at
    # representable-value boundaries).
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor_gcu(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor (column-major scales).

    GCU variant: uses 2-D grid (program_id 0 + 1) so that grid.x never
    exceeds the hardware limit of 65535.
    """
    groups_per_row = y_num_columns // group_size

    # Map the 2-D program id to a flat group id.
    g_id = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    # Ensure offset calculation uses int64 for y_s_ptr
    y_s_ptr_offset = (scale_col.to(tl.int64) * y_s_col_stride) + scale_row.to(tl.int64)
    y_s_ptr += y_s_ptr_offset

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


# ---------------------------------------------------------------------------
# GCU-compatible launcher (replaces per_token_group_quant_fp8)
# ---------------------------------------------------------------------------

def per_token_group_quant_fp8_gcu(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GCU-compatible version of ``per_token_group_quant_fp8``.

    Identical semantics to the upstream function, but launches Triton kernels
    with a 2-D grid that respects GCU hardware limits
    (grid.x ≤ 65535, grid.y ≤ 255).
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_fp8_min_max,
    )
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import (
        get_tma_aligned_size,
        is_deep_gemm_e8m0_used,
    )

    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    fp8_min, fp8_max = get_fp8_min_max()

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty(x.shape, device=x.device, dtype=dtype)

    # Allocate the scale tensor in either row- or column-major format.
    if column_major_scales:
        if tma_aligned_scales:
            m = x.shape[-2]
            sf_k = x.shape[-1] // group_size
            tma_aligned_m = get_tma_aligned_size(m, 4)
            shape = x.shape[:-2] + (m, sf_k)
            stride = (
                (1, tma_aligned_m)
                if x.dim() == 2
                else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            )
            x_s = torch.empty_strided(
                shape, stride, device=x.device, dtype=torch.float32
            )
        else:
            shape = x.shape[:-2] + (x.shape[-1] // group_size, x.shape[-2])
            x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(
                -1, -2
            )
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    # prefer CUDA kernel if available
    if current_platform.is_cuda() and x.is_contiguous():
        torch.ops._C.per_token_group_fp8_quant(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            use_ue8m0,
            column_major_scales,
            tma_aligned_scales,
        )
        return x_q, x_s

    # TRITON FALLBACK – use 2-D grid for GCU
    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    grid = _gcu_grid(M)

    if column_major_scales:
        _per_token_group_quant_fp8_colmajor_gcu[grid](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=use_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8_gcu[grid](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=use_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


# ---------------------------------------------------------------------------
# Torch-native (CPU) launcher
# ---------------------------------------------------------------------------

def per_token_group_quant_fp8_torch(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch (CPU-based) version of ``per_token_group_quant_fp8``.

    Semantics identical to the Triton-based ``per_token_group_quant_fp8_gcu``,
    but implemented with torch native ops executed on CPU.  Input is moved to
    CPU before computation and results are moved back to the original device.

    This is useful when:
    - The Triton kernel cannot be launched (e.g. grid-size limits on GCU).
    - You need a CPU-reference implementation for validation.
    """
    original_device = x.device

    # ------------------------------------------------------------------
    # Validation (identical to upstream / GCU version)
    # ------------------------------------------------------------------
    if use_ue8m0 is None:
        use_ue8m0 = False
    if dtype is None:
        dtype = torch.float8_e4m3fn

    # Resolve fp8 min/max from the target dtype.
    try:
        fp8_finfo = torch.finfo(dtype)
        fp8_min = fp8_finfo.min
        fp8_max = fp8_finfo.max
    except TypeError:
        fp8_min = -448.0
        fp8_max = 448.0

    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    assert out_q is None or out_q.shape == x.shape

    # ------------------------------------------------------------------
    # Move input to CPU and compute in float32
    # ------------------------------------------------------------------
    x_cpu = x.detach().cpu().float()
    *batch_dims, K = x.shape
    groups_per_row = K // group_size

    # Flatten all leading dims → (-1, group_size) for batched per-group ops.
    x_flat = x_cpu.reshape(-1, group_size)  # (M_total, group_size)

    # Per-group absmax (clamped to eps).
    absmax = x_flat.abs().amax(dim=-1)                     # (M_total,)
    absmax = torch.clamp(absmax, min=eps)

    # Scale: absmax / fp8_max, optionally rounded to power-of-two.
    scale_raw = absmax * (1.0 / fp8_max)                   # (M_total,)
    if use_ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale_raw)))
    else:
        scale = scale_raw

    # Quantize: clamp(x / scale, fp8_min, fp8_max).
    x_q_flat = x_flat / scale.unsqueeze(-1)                # (M_total, group_size)
    x_q_flat = torch.clamp(x_q_flat, fp8_min, fp8_max)

    # ------------------------------------------------------------------
    # Build outputs on the original device
    # ------------------------------------------------------------------
    if out_q is not None:
        # Write into caller-provided buffer.
        out_q.copy_(x_q_flat.reshape(x.shape).to(device=original_device, dtype=dtype))
        x_q = out_q
    else:
        x_q = x_q_flat.reshape(x.shape).to(device=original_device, dtype=dtype)

    # Scales: reshape back to (*batch_dims, M, groups_per_row).
    scale_cpu = scale.reshape(*batch_dims, groups_per_row)  # (*batch, M, sf_k)

    if column_major_scales:
        # For scales, the batch dimensions exclude the last two axes of x.
        scale_batch = x.shape[:-2]   # () for 2-D, (B,) for 3-D, etc.
        m = x.shape[-2]
        sf_k = groups_per_row

        if tma_aligned_scales:
            # TMA-aligned column-major layout.
            tma_aligned_m = ((m + 3) // 4) * 4  # get_tma_aligned_size(m, 4)

            if x.dim() == 2:
                # shape (M, sf_k), stride (1, tma_aligned_m)
                buf_size = tma_aligned_m * sf_k
                buf = torch.zeros(buf_size, device=original_device,
                                  dtype=torch.float32)
                scale_cpu_dev = scale_cpu.to(device=original_device)
                for j in range(sf_k):
                    buf[j * tma_aligned_m: j * tma_aligned_m + m].copy_(
                        scale_cpu_dev[:, j]
                    )
                x_s = torch.as_strided(buf, (m, sf_k), (1, tma_aligned_m))
            else:
                # 3-D: shape (B, M, sf_k), stride
                #      (tma_aligned_m * sf_k, 1, tma_aligned_m)
                B = scale_batch[0]
                buf_size = B * tma_aligned_m * sf_k
                buf = torch.zeros(buf_size, device=original_device,
                                  dtype=torch.float32)
                scale_cpu_dev = scale_cpu.reshape(B, m, sf_k).to(
                    device=original_device
                )
                for b in range(B):
                    base = b * tma_aligned_m * sf_k
                    for j in range(sf_k):
                        buf[base + j * tma_aligned_m:
                            base + j * tma_aligned_m + m].copy_(
                            scale_cpu_dev[b, :, j]
                        )
                x_s = torch.as_strided(
                    buf, (B, m, sf_k),
                    (tma_aligned_m * sf_k, 1, tma_aligned_m),
                )
        else:
            # Column-major (no TMA): logical shape (*scale_batch, M, sf_k)
            # backed by (*scale_batch, sf_k, M) and permuted.
            x_s_base = torch.zeros(
                *scale_batch, sf_k, m,
                device=original_device, dtype=torch.float32,
            )
            scale_cpu_dev = scale_cpu.to(device=original_device)
            for j in range(sf_k):
                x_s_base[..., j, :] = scale_cpu_dev[..., :, j]
            ndim_batch = len(scale_batch)
            x_s = x_s_base.permute(
                *range(ndim_batch), ndim_batch + 1, ndim_batch
            )
    else:
        # Row-major: simple reshape + move.
        x_s = scale_cpu.to(device=original_device)

    return x_q, x_s


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_per_token_group_quant_fp8_gcu_patch() -> None:
    """Patch ``per_token_group_quant_fp8`` in fp8_utils and all known importers
    for GCU grid limits.

    Patching only ``fp8_utils.per_token_group_quant_fp8`` is insufficient
    because many callers (such as ``fused_moe/utils.py``) use
    ``from ...fp8_utils import per_token_group_quant_fp8``, which creates
    a local binding that still references the original function.  We must
    also update the local reference in every importer.
    """
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    # All modules known to import per_token_group_quant_fp8 via
    # ``from ...fp8_utils import per_token_group_quant_fp8``.
    _IMPORTERS: list[str] = [
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        "vllm.model_executor.layers.fused_moe.utils",
        "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe",
        "vllm.model_executor.kernels.linear.scaled_mm.flashinfer",
        "vllm.model_executor.models.deepseek_v2",
    ]

    try:
        for module_name in _IMPORTERS:
            try:
                mod = __import__(module_name, fromlist=["per_token_group_quant_fp8"])
            except ImportError:
                continue
            if hasattr(mod, "per_token_group_quant_fp8"):
                # mod.per_token_group_quant_fp8 = per_token_group_quant_fp8_gcu
                mod.per_token_group_quant_fp8 = per_token_group_quant_fp8_torch

        _patched = True
        logger.info(
            "Patched per_token_group_quant_fp8 for GCU "
            "(grid.x <= %d, grid.y <= %d) in %d modules",
            GCU_MAX_GRID_X,
            GCU_MAX_GRID_YZ,
            len(_IMPORTERS),
        )
    except Exception as exc:
        logger.warning(
            "Failed to patch per_token_group_quant_fp8 for GCU: %s",
            exc,
        )

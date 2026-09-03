# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU fix for Qwen3-VL bilinear position-embedding Triton kernel.

GCU hardware limits grid.x to 65535.  The upstream kernel launches one CTA per
output token with ``grid=(total_out,)``, which fails when ``t * h * w > 65535``
(e.g. 2048 dummy video frames in warmup).
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

GCU_MAX_GRID_X = 65535
_patched = False


@triton.jit
def _bilinear_pos_embed_kernel_gcu(
    embed_ptr,
    output_ptr,
    H,
    W,
    h_scale,
    w_scale,
    NUM_GRID: tl.constexpr,
    M_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOTAL_OUT,
):
    """Fused bilinear pos-embed interpolation with spatial-merge reorder."""
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    if pid >= TOTAL_OUT:
        return

    total_spatial = H * W
    spatial_idx = pid % total_spatial

    num_blocks_w = W // M_SIZE
    block_idx = spatial_idx // (M_SIZE * M_SIZE)
    local_idx = spatial_idx % (M_SIZE * M_SIZE)
    br = block_idx // num_blocks_w
    bc = block_idx % num_blocks_w
    lr = local_idx // M_SIZE
    lc = local_idx % M_SIZE
    row = br * M_SIZE + lr
    col = bc * M_SIZE + lc

    h_frac = row.to(tl.float32) * h_scale
    w_frac = col.to(tl.float32) * w_scale

    hf = tl.math.floor(h_frac).to(tl.int32)
    wf = tl.math.floor(w_frac).to(tl.int32)
    hc = tl.minimum(hf + 1, NUM_GRID - 1)
    wc = tl.minimum(wf + 1, NUM_GRID - 1)

    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01

    off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
    off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
    off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
    off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
    out_off = pid * HIDDEN_DIM

    out_dtype = output_ptr.dtype.element_ty
    w00_c = w00.to(out_dtype)
    w01_c = w01.to(out_dtype)
    w10_c = w10.to(out_dtype)
    w11_c = w11.to(out_dtype)

    for d in tl.range(0, HIDDEN_DIM, BLOCK_D):
        cols = d + tl.arange(0, BLOCK_D)
        mask = cols < HIDDEN_DIM

        e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
        e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
        e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
        e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

        val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

        tl.store(output_ptr + out_off + cols, val, mask=mask)


def triton_pos_embed_interpolate_gcu(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """GCU-safe launcher: split grid across (x, y) when total_out exceeds 65535."""
    assert h % m_size == 0 and w % m_size == 0, (
        f"h={h} and w={w} must be divisible by m_size={m_size}"
    )
    hidden_dim = embed_weight.shape[1]
    total_out = t * h * w
    output = torch.empty(
        total_out,
        hidden_dim,
        device=embed_weight.device,
        dtype=dtype,
    )

    h_scale = float(num_grid_per_side - 1) / float(h - 1) if h > 1 else 0.0
    w_scale = float(num_grid_per_side - 1) / float(w - 1) if w > 1 else 0.0

    block_d = triton.next_power_of_2(hidden_dim)

    grid_x = min(total_out, GCU_MAX_GRID_X)
    grid_y = triton.cdiv(total_out, grid_x)

    _bilinear_pos_embed_kernel_gcu[(grid_x, grid_y)](
        embed_weight,
        output,
        h,
        w,
        h_scale,
        w_scale,
        num_grid_per_side,
        m_size,
        hidden_dim,
        block_d,
        total_out,
    )
    return output


def apply_bilinear_pos_embed_gcu_patch() -> None:
    """Replace upstream Triton launcher with the GCU grid-safe version."""
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    try:
        import vllm.model_executor.models.qwen3_vl as qwen3_vl

        if not getattr(qwen3_vl, "HAS_TRITON", False):
            return

        qwen3_vl.triton_pos_embed_interpolate = triton_pos_embed_interpolate_gcu
        qwen3_vl._bilinear_pos_embed_kernel = _bilinear_pos_embed_kernel_gcu
        _patched = True
        logger.info(
            "Patched Qwen3-VL bilinear pos embed for GCU (grid.x <= %d)",
            GCU_MAX_GRID_X,
        )
    except Exception as exc:
        logger.warning("Failed to patch bilinear pos embed for GCU: %s", exc)

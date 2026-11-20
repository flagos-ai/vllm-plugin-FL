# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon Triton implementation of dynamic symmetric INT8 quantization."""

import torch
from triton.language.extra import libdevice

from vllm.triton_utils import tl, triton


@triton.jit
def _per_token_quant_int8_one_kernel_opt(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    n_cols,
    BLOCK: tl.constexpr,
):
    """Quantize one contiguous row per Triton program.

    This is adapted from vllm_hcu's
    ``_per_token_quant_int8_one_kernel_opt``.  The expert-token filtering was
    removed because ``_C::dynamic_scaled_int8_quant`` operates on every row.
    """
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(
        x_ptr + row_id * stride_x + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.max(tl.abs(x))

    # Match vLLM's native dynamic symmetric kernel for an all-zero row:
    # scale=0 and inv_scale=0.  vllm_hcu clamps absmax to 1e-10, which is
    # harmless for MoE math but does not exactly satisfy the _C op contract.
    safe_absmax = tl.where(absmax == 0.0, 1.0, absmax)
    inv_scale = tl.where(absmax == 0.0, 0.0, 127.0 / safe_absmax)
    scale = absmax / 127.0
    x_q = libdevice.nearbyint(x * inv_scale).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale)


def dynamic_scaled_int8_quant_hygon(
    result: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    azp: torch.Tensor | None,
) -> None:
    """Implement vLLM's dynamic per-token symmetric INT8 ``_C`` op.

    The current W8A8 compressed-tensors path always supplies ``azp=None``.
    Dynamic asymmetric quantization is deliberately rejected until a referenced
    Hygon implementation is available.
    """
    if azp is not None:
        raise NotImplementedError(
            "Hygon Triton implementation supports only dynamic symmetric "
            "INT8 quantization (azp must be None)."
        )
    if input.ndim == 0 or input.shape[-1] == 0:
        raise ValueError(
            "dynamic_scaled_int8_quant requires a non-empty last dimension"
        )
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "dynamic_scaled_int8_quant input must be float16, bfloat16, or "
            f"float32, but got {input.dtype}"
        )
    if result.dtype is not torch.int8:
        raise TypeError(
            f"dynamic_scaled_int8_quant result must be int8, got {result.dtype}"
        )
    if scales.dtype is not torch.float32:
        raise TypeError(
            f"dynamic_scaled_int8_quant scales must be float32, got {scales.dtype}"
        )
    if result.shape != input.shape:
        raise ValueError(
            "dynamic_scaled_int8_quant result shape must match input shape: "
            f"{result.shape} != {input.shape}"
        )

    hidden_size = input.shape[-1]
    num_tokens = input.numel() // hidden_size
    if scales.numel() != num_tokens:
        raise ValueError(
            "dynamic_scaled_int8_quant requires one scale per input row: "
            f"expected {num_tokens}, got {scales.numel()}"
        )
    if not input.is_contiguous():
        raise ValueError("dynamic_scaled_int8_quant input must be contiguous")
    if not result.is_contiguous():
        raise ValueError("dynamic_scaled_int8_quant result must be contiguous")
    if not scales.is_contiguous():
        raise ValueError("dynamic_scaled_int8_quant scales must be contiguous")
    if result.device != input.device or scales.device != input.device:
        raise ValueError("dynamic_scaled_int8_quant tensors must be on the same device")

    if num_tokens == 0:
        return

    block = triton.next_power_of_2(hidden_size)
    num_warps = min(max(block // 256, 1), 8)
    _per_token_quant_int8_one_kernel_opt[(num_tokens,)](
        input,
        result,
        scales,
        stride_x=hidden_size,
        stride_xq=hidden_size,
        n_cols=hidden_size,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )

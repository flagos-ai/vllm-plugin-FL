# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe, cross-accelerator kernels for Qwen4 gated HyperConnection."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


def can_use_hc_triton(*tensors: torch.Tensor) -> bool:
    """Return whether the specialized contiguous inference path is usable."""

    return bool(
        HAS_TRITON
        and tensors
        and all(
            tensor.device.type not in ("cpu", "meta")
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and tensor.is_contiguous()
            for tensor in tensors
        )
    )


def _has_flat_row_layout(tensor: torch.Tensor) -> bool:
    """Return whether leading dims can be flattened using ``stride(-2)``.

    Packed HC projection output is split into ``[..., lowrank]`` and
    ``[..., hc_count]`` views.  The latter is contiguous within each row but
    retains the packed projection's wider row stride.  The injection kernel
    accepts that layout explicitly, while still rejecting transposes and
    layouts whose leading dimensions cannot be flattened into rows.
    """

    if tensor.ndim < 2 or tensor.stride(-1) != 1:
        return False
    if tensor.stride(-2) < tensor.shape[-1]:
        return False
    return all(
        tensor.stride(dim) == tensor.shape[dim + 1] * tensor.stride(dim + 1)
        for dim in range(tensor.ndim - 2)
    )


def can_use_hc_inject_triton(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
) -> bool:
    """Allow packed-projection logits with a wider row stride.

    ``_hc_inject_combine_kernel`` consumes ``stride_logits_row`` and only
    requires the branch dimension to be contiguous.  The other two inputs
    retain the stricter contiguous inference contract.
    """

    tensors = (injection_logits, block_output, residual)
    return bool(
        HAS_TRITON
        and all(
            tensor.device.type not in ("cpu", "meta")
            and tensor.dtype in (torch.bfloat16, torch.float16)
            for tensor in tensors
        )
        and len({tensor.device for tensor in tensors}) == 1
        and _has_flat_row_layout(injection_logits)
        and block_output.is_contiguous()
        and residual.is_contiguous()
    )


def can_use_hc_combine_norm_triton(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> bool:
    """Return whether delayed combine+grouped-RMSNorm can use Triton.

    The injection view may retain the wider row stride of the packed
    down+inject projection.  The residual and block output are required to
    remain contiguous because the fused kernel writes a fresh contiguous
    multi-stream state and normalizes each branch in place conceptually.
    """

    return bool(
        can_use_hc_inject_triton(injection_logits, block_output, residual)
        and norm_weight.device == residual.device
        and norm_weight.dtype in (torch.bfloat16, torch.float16)
        and norm_weight.is_contiguous()
    )


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    group_row = tl.program_id(0)
    offsets = tl.arange(0, block_h)
    mask = offsets < hidden_size
    input_base = group_row * hidden_size
    weight_base = (group_row % hc_count) * hidden_size
    values = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + weight_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
    tl.store(
        output_ptr + input_base + offsets,
        values * inv_rms * (1.0 + weight),
        mask=mask,
    )


@triton.jit
def _hc_gate_reduce_kernel(
    logits_ptr,
    normed_ptr,
    output_ptr,
    stride_logits_row,
    stride_normed_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.program_id(1) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    accumulator = tl.zeros((block_h,), dtype=tl.float32)
    for branch in tl.static_range(0, hc_count):
        branch_offsets = branch * hidden_size + offsets
        logits = tl.load(
            logits_ptr + row * stride_logits_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normed = tl.load(
            normed_ptr + row * stride_normed_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sigmoid(logits) * normed
    tl.store(
        output_ptr + row * stride_output_row + offsets,
        accumulator / hc_count,
        mask=mask,
    )


@triton.jit
def _hc_inject_combine_kernel(
    injection_logits_ptr,
    block_output_ptr,
    residual_ptr,
    output_ptr,
    stride_logits_row,
    stride_block_row,
    stride_residual_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    branch = tl.program_id(1)
    offsets = tl.program_id(2) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    branch_offsets = branch * hidden_size + offsets
    logits = tl.load(injection_logits_ptr + row * stride_logits_row + branch).to(
        tl.float32
    )
    injection_weight = 2.0 * tl.sigmoid(logits / hc_count)
    block_output = tl.load(
        block_output_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * stride_residual_row + branch_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + branch_offsets,
        residual + block_output * injection_weight,
        mask=mask,
    )


def qwen4_grouped_gemma_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    if hidden_states.ndim < 2 or hidden_states.shape[-1] != weight.numel():
        raise ValueError("Qwen4 HC RMSNorm received incompatible input and weight")
    if hc_count <= 0 or weight.numel() % hc_count:
        raise ValueError("Qwen4 HC RMSNorm requires a valid HC count")
    if not can_use_hc_triton(hidden_states, weight):
        raise RuntimeError("Qwen4 HC RMSNorm requires contiguous accelerator tensors")
    output = torch.empty_like(hidden_states)
    if not hidden_states.numel():
        return output
    hidden_size = weight.numel() // hc_count
    block_h = triton.next_power_of_2(hidden_size)
    rows = hidden_states.numel() // weight.numel()
    _grouped_gemma_rmsnorm_kernel[(rows * hc_count,)](
        hidden_states,
        weight,
        output,
        hidden_size=hidden_size,
        hc_count=hc_count,
        eps=eps,
        block_h=block_h,
        num_warps=8 if block_h > 2048 else 4,
    )
    return output


def qwen4_grouped_gemma_rmsnorm_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    del weight, hc_count, eps
    return torch.empty_like(hidden_states)


def qwen4_hc_gate_reduce(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if logits.shape != normed.shape or logits.ndim < 2:
        raise ValueError("Qwen4 HC gate logits and normalized input must match")
    if hc_count <= 0 or logits.shape[-1] % hc_count:
        raise ValueError("Qwen4 HC gate reduction requires a valid HC count")
    if not can_use_hc_triton(logits, normed):
        raise RuntimeError("Qwen4 HC gate reduction requires contiguous tensors")
    hidden_size = logits.shape[-1] // hc_count
    output = torch.empty(
        (*logits.shape[:-1], hidden_size), dtype=normed.dtype, device=normed.device
    )
    if not logits.numel():
        return output
    rows = logits.numel() // logits.shape[-1]
    block_h = 256
    _hc_gate_reduce_kernel[(rows, triton.cdiv(hidden_size, block_h))](
        logits,
        normed,
        output,
        logits.stride(-2),
        normed.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=4,
    )
    return output


def qwen4_hc_gate_reduce_fake(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del normed
    return torch.empty(
        (*logits.shape[:-1], logits.shape[-1] // hc_count),
        dtype=logits.dtype,
        device=logits.device,
    )


def qwen4_hc_inject_combine(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if injection_logits.shape != (*block_output.shape[:-1], hc_count):
        raise ValueError("Qwen4 HC injection logits have an invalid shape")
    if residual.shape != (
        *block_output.shape[:-1],
        hc_count * block_output.shape[-1],
    ):
        raise ValueError("Qwen4 HC residual and block output shapes are incompatible")
    if not can_use_hc_inject_triton(injection_logits, block_output, residual):
        raise RuntimeError("Qwen4 HC injection received an unsupported accelerator layout")
    output = torch.empty_like(residual)
    if not residual.numel():
        return output
    hidden_size = block_output.shape[-1]
    rows = block_output.numel() // hidden_size
    block_h = 256
    _hc_inject_combine_kernel[(rows, hc_count, triton.cdiv(hidden_size, block_h))](
        injection_logits,
        block_output,
        residual,
        output,
        injection_logits.stride(-2),
        block_output.stride(-2),
        residual.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=4,
    )
    return output


def qwen4_hc_inject_combine_fake(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del injection_logits, block_output, hc_count
    return torch.empty_like(residual)


@triton.jit
def _hc_combine_norm_kernel(
    block_ptr,
    residual_ptr,
    injection_ptr,
    weight_ptr,
    output_ptr,
    normed_ptr,
    stride_block_row,
    stride_residual_row,
    stride_injection_row,
    stride_output_row,
    stride_normed_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    weight_shared: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    """Fuse pending HC injection, materialization, and grouped RMSNorm.

    One program owns one token/branch pair.  All hidden-size tiles are kept
    in the program so the RMS reduction sees the same rounded-to-residual-dtype
    values that the unfused combine -> norm sequence would consume.
    """

    num_tiles: tl.constexpr = triton.cdiv(hidden_size, block_h)
    num_tiles_padded: tl.constexpr = triton.next_power_of_2(num_tiles)

    row = tl.program_id(0)
    branch = tl.program_id(1)
    tile_ids = tl.arange(0, num_tiles_padded)
    offsets = tile_ids[:, None] * block_h + tl.arange(0, block_h)[None, :]
    mask = offsets < hidden_size
    branch_offsets = branch * hidden_size + offsets
    weight_offsets = offsets if weight_shared else branch_offsets

    residual = tl.load(
        residual_ptr + row * stride_residual_row + branch_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    block = tl.load(
        block_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    injection = tl.load(
        injection_ptr + row * stride_injection_row + branch,
    ).to(tl.float32)
    injection = 2.0 * tl.sigmoid(injection / hc_count)

    # Match the official fused kernel's explicit residual-dtype boundary.
    combined = (residual + block * injection).to(output_ptr.dtype.element_ty)
    tl.store(
        output_ptr + row * stride_output_row + branch_offsets,
        combined,
        mask=mask,
    )

    combined_f32 = combined.to(tl.float32)
    sum_sq = tl.sum(tl.sum(combined_f32 * combined_f32, axis=1), axis=0)
    inv_rms = tl.rsqrt(sum_sq / hidden_size + eps)
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    normalized = combined_f32 * inv_rms
    normalized += normalized * weight.to(tl.float32)
    tl.store(
        normed_ptr + row * stride_normed_row + branch_offsets,
        normalized,
        mask=mask,
    )


def qwen4_hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused delayed HC combine + grouped Gemma RMSNorm.

    This is the vLLM-0.24-compatible counterpart of the official
    ``qwen3_8_flash_next_hc_combine_norm`` op.  It intentionally has a
    separate qwen4 name so it cannot collide with a vendored official op.
    """

    if residual.ndim != 2 or block_output.ndim != 2 or injection_logits.ndim != 2:
        raise ValueError("Qwen4 HC combine_norm currently requires 2-D tensors")
    if hc_count <= 0 or residual.shape[-1] % hc_count:
        raise ValueError("Qwen4 HC combine_norm requires a valid HC count")
    hidden_size = residual.shape[-1] // hc_count
    if block_output.shape != (residual.shape[0], hidden_size):
        raise ValueError("Qwen4 HC combine_norm block output shape is invalid")
    if injection_logits.shape != (residual.shape[0], hc_count):
        raise ValueError("Qwen4 HC combine_norm injection shape is invalid")
    if norm_weight.numel() not in (hidden_size, residual.shape[-1]):
        raise ValueError("Qwen4 HC combine_norm weight shape is invalid")
    if not can_use_hc_combine_norm_triton(
        injection_logits, block_output, residual, norm_weight
    ):
        raise RuntimeError(
            "Qwen4 HC combine_norm requires contiguous accelerator tensors"
        )

    output = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    if not residual.numel():
        return output, normalized

    block_h = 512
    rows = residual.shape[0]
    _hc_combine_norm_kernel[(rows, hc_count)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        output,
        normalized,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        output.stride(0),
        normalized.stride(0),
        hidden_size=hidden_size,
        hc_count=hc_count,
        weight_shared=norm_weight.numel() == hidden_size,
        eps=eps,
        block_h=block_h,
        num_warps=8 if hidden_size > 2048 else 4,
    )
    return output, normalized


def qwen4_hc_combine_norm_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    return torch.empty_like(residual), torch.empty_like(residual)


direct_register_custom_op(
    op_name="qwen4_grouped_gemma_rmsnorm",
    op_func=qwen4_grouped_gemma_rmsnorm,
    mutates_args=[],
    fake_impl=qwen4_grouped_gemma_rmsnorm_fake,
)
direct_register_custom_op(
    op_name="qwen4_hc_gate_reduce",
    op_func=qwen4_hc_gate_reduce,
    mutates_args=[],
    fake_impl=qwen4_hc_gate_reduce_fake,
)
direct_register_custom_op(
    op_name="qwen4_hc_inject_combine",
    op_func=qwen4_hc_inject_combine,
    mutates_args=[],
    fake_impl=qwen4_hc_inject_combine_fake,
)
direct_register_custom_op(
    op_name="qwen4_hc_combine_norm",
    op_func=qwen4_hc_combine_norm,
    mutates_args=[],
    fake_impl=qwen4_hc_combine_norm_fake,
)


__all__ = [
    "can_use_hc_combine_norm_triton",
    "can_use_hc_inject_triton",
    "can_use_hc_triton",
    "qwen4_grouped_gemma_rmsnorm",
    "qwen4_hc_combine_norm",
    "qwen4_hc_gate_reduce",
    "qwen4_hc_inject_combine",
]

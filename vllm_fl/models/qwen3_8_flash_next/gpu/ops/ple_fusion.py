# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe cross-vendor Triton fusion for the Qwen3.8 PLE prefill path.

The kernels in this file deliberately stop at communication and GEMM
boundaries.  They fuse only the pointwise/reduction chain around the PLE gate
and the token packing/layout/depthwise-convolution chain used by prefill.
Decode and speculative-decode keep using the existing implementation.
"""

from __future__ import annotations

import math
import os

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_nonnegative_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(0, int(value))
    except ValueError:
        return default


PLE_FUSION_ENABLED = _env_flag("VLLM_FL_PLE_FUSION", True)
PLE_FUSION_MIN_TOKENS = _env_nonnegative_int("VLLM_FL_PLE_FUSION_MIN_TOKENS", 1024)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _default_prefill_conv_config(device: torch.device) -> tuple[int, int]:
    """Return a measured Hopper config with a portable fallback."""

    if device.type == "cuda" and torch.version.hip is None:
        try:
            if torch.cuda.get_device_capability(device)[0] >= 9:
                return 8, 256
        except (AssertionError, RuntimeError):
            pass
    return 16, 128


@triton.jit
def _ple_gate_norm_kernel(
    key_ptr,
    query_ptr,
    value_ptr,
    key_weight_ptr,
    query_weight_ptr,
    conv_weight_ptr,
    gated_ptr,
    normalized_ptr,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    eps: tl.constexpr,
    inv_sqrt_hidden: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    """Fuse three grouped RMS norms and the complete PLE gate chain.

    A program owns one ``(token, HC branch)`` row.  Reductions accumulate in
    FP32, while explicit casts preserve the reference path's low-precision
    materialization points (normalized key/query, gate, and gated value).
    """

    group_row = tl.program_id(0)
    token = group_row // hc_count
    branch = group_row % hc_count
    offsets = tl.arange(0, block_h)
    mask = offsets < hidden_size
    row_offsets = group_row * hidden_size + offsets
    value_offsets = token * hidden_size + offsets
    weight_offsets = branch * hidden_size + offsets

    key = tl.load(key_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    query = tl.load(query_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    key_weight = tl.load(key_weight_ptr + weight_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    query_weight = tl.load(query_weight_ptr + weight_offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    key_inv_rms = tl.rsqrt(tl.sum(key * key, axis=0) / hidden_size + eps)
    query_inv_rms = tl.rsqrt(tl.sum(query * query, axis=0) / hidden_size + eps)

    # The eager reference stores both norm results in the source dtype before
    # multiplying them.  Preserve those rounding points instead of feeding an
    # all-FP32 dot product into the nonlinear gate.
    source_dtype: tl.constexpr = key_ptr.dtype.element_ty
    norm_key = (key * key_inv_rms * (1.0 + key_weight)).to(source_dtype)
    norm_query = (query * query_inv_rms * (1.0 + query_weight)).to(source_dtype)
    products = (norm_key * norm_query).to(source_dtype)
    gate_linear = (tl.sum(products.to(tl.float32), axis=0) * inv_sqrt_hidden).to(
        source_dtype
    )
    gate_linear_fp32 = gate_linear.to(tl.float32)
    gate_sign = tl.where(
        gate_linear_fp32 > 0.0,
        1.0,
        tl.where(gate_linear_fp32 < 0.0, -1.0, 0.0),
    )
    signed_root = (
        gate_sign * tl.sqrt(tl.maximum(tl.abs(gate_linear_fp32), 1.0e-6))
    ).to(source_dtype)
    gate = tl.sigmoid(signed_root.to(tl.float32)).to(source_dtype)

    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
    gated = (gate * value).to(source_dtype)
    gated_fp32 = gated.to(tl.float32)
    conv_weight = tl.load(conv_weight_ptr + weight_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    conv_inv_rms = tl.rsqrt(tl.sum(gated_fp32 * gated_fp32, axis=0) / hidden_size + eps)
    normalized = gated_fp32 * conv_inv_rms * (1.0 + conv_weight)

    tl.store(gated_ptr + row_offsets, gated, mask=mask)
    tl.store(normalized_ptr + row_offsets, normalized, mask=mask)


@triton.jit
def _ple_prefill_dilated_conv_kernel(
    x_ptr,
    weight_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_ptr,
    conv_state_ptr,
    output_ptr,
    num_state_rows: tl.constexpr,
    channels: tl.constexpr,
    state_len: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_channel: tl.constexpr,
    stride_weight_channel: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_query_start: tl.constexpr,
    stride_state_index: tl.constexpr,
    stride_has_initial: tl.constexpr,
    stride_state_row: tl.constexpr,
    stride_state_channel: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_output_token: tl.constexpr,
    stride_output_channel: tl.constexpr,
    null_block_id: tl.constexpr,
    kernel_width: tl.constexpr,
    dilation: tl.constexpr,
    has_state: tl.constexpr,
    block_t: tl.constexpr,
    block_c: tl.constexpr,
) -> None:
    """Compute dilated causal depthwise convolution without pack/transposes."""

    request = tl.program_id(0)
    time_block = tl.program_id(1)
    channel_block = tl.program_id(2)
    token_offsets = time_block * block_t + tl.arange(0, block_t)[:, None]
    channel_offsets = channel_block * block_c + tl.arange(0, block_c)[None, :]

    query_start = tl.load(query_start_loc_ptr + request * stride_query_start).to(
        tl.int64
    )
    query_end = tl.load(query_start_loc_ptr + (request + 1) * stride_query_start).to(
        tl.int64
    )
    query_len = query_end - query_start
    state_index = tl.load(state_indices_ptr + request * stride_state_index).to(tl.int64)
    valid_state = state_index != null_block_id
    if has_state:
        valid_state = valid_state & (state_index >= 0) & (state_index < num_state_rows)
        safe_state_index = tl.where(valid_state, state_index, 0)
        has_initial = (
            tl.load(has_initial_ptr + request * stride_has_initial).to(tl.int1)
            & valid_state
        )
    else:
        safe_state_index = 0
        has_initial = False

    valid_tokens = token_offsets < query_len
    valid_channels = channel_offsets < channels
    accumulator = tl.zeros((block_t, block_c), dtype=tl.float32)

    for tap in tl.static_range(0, kernel_width):
        source_position = token_offsets - (kernel_width - 1 - tap) * dilation
        source_is_token = source_position >= 0
        token_ptrs = (
            x_ptr
            + (query_start + source_position) * stride_x_token
            + channel_offsets * stride_x_channel
        )
        token_values = tl.load(
            token_ptrs,
            mask=(
                valid_tokens
                & valid_channels
                & source_is_token
                & (source_position < query_len)
            ),
            other=0.0,
        )
        if has_state:
            state_position = state_len + source_position
            state_ptrs = (
                conv_state_ptr
                + safe_state_index * stride_state_row
                + channel_offsets * stride_state_channel
                + state_position * stride_state_token
            )
            state_values = tl.load(
                state_ptrs,
                mask=(
                    valid_tokens
                    & valid_channels
                    & (~source_is_token)
                    & (state_position >= 0)
                    & (state_position < state_len)
                    & has_initial
                ),
                other=0.0,
            )
        else:
            state_values = tl.zeros((block_t, block_c), dtype=x_ptr.dtype.element_ty)
        # The masks are disjoint, avoiding a load-dependent tl.where and its
        # known ordering hazard on affected Triton versions.
        source_values = token_values + state_values
        conv_weight = tl.load(
            weight_ptr
            + channel_offsets * stride_weight_channel
            + tap * stride_weight_width,
            mask=valid_channels,
            other=0.0,
        )
        accumulator += source_values.to(tl.float32) * conv_weight.to(tl.float32)

    activated = accumulator * tl.sigmoid(accumulator)
    activated *= valid_state.to(tl.float32)
    output_ptrs = (
        output_ptr
        + (query_start + token_offsets) * stride_output_token
        + channel_offsets * stride_output_channel
    )
    tl.store(
        output_ptrs,
        activated,
        mask=valid_tokens & valid_channels,
    )


@triton.jit
def _ple_prefill_state_update_kernel(
    x_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_ptr,
    conv_state_ptr,
    num_state_rows: tl.constexpr,
    channels: tl.constexpr,
    state_len: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_channel: tl.constexpr,
    stride_query_start: tl.constexpr,
    stride_state_index: tl.constexpr,
    stride_has_initial: tl.constexpr,
    stride_state_row: tl.constexpr,
    stride_state_channel: tl.constexpr,
    stride_state_token: tl.constexpr,
    null_block_id: tl.constexpr,
    block_s: tl.constexpr,
    block_c: tl.constexpr,
) -> None:
    """Update the state after the output kernel has consumed the old state."""

    request = tl.program_id(0)
    channel_block = tl.program_id(1)
    state_offsets = tl.arange(0, block_s)[:, None]
    channel_offsets = channel_block * block_c + tl.arange(0, block_c)[None, :]
    valid_channels = channel_offsets < channels

    query_start = tl.load(query_start_loc_ptr + request * stride_query_start).to(
        tl.int64
    )
    query_end = tl.load(query_start_loc_ptr + (request + 1) * stride_query_start).to(
        tl.int64
    )
    query_len = query_end - query_start
    state_index = tl.load(state_indices_ptr + request * stride_state_index).to(tl.int64)
    valid_state = (
        (state_index != null_block_id)
        & (state_index >= 0)
        & (state_index < num_state_rows)
        & (query_len > 0)
    )
    safe_state_index = tl.where(valid_state, state_index, 0)
    has_initial = (
        tl.load(has_initial_ptr + request * stride_has_initial).to(tl.int1)
        & valid_state
    )

    source_position = query_len + state_offsets - state_len
    source_is_token = source_position >= 0
    token_ptrs = (
        x_ptr
        + (query_start + source_position) * stride_x_token
        + channel_offsets * stride_x_channel
    )
    token_values = tl.load(
        token_ptrs,
        mask=(
            valid_state
            & valid_channels
            & (state_offsets < state_len)
            & source_is_token
            & (source_position < query_len)
        ),
        other=0.0,
    )

    old_state_position = state_len + source_position
    old_state_ptrs = (
        conv_state_ptr
        + safe_state_index * stride_state_row
        + channel_offsets * stride_state_channel
        + old_state_position * stride_state_token
    )
    old_state_values = tl.load(
        old_state_ptrs,
        mask=(
            valid_state
            & valid_channels
            & (state_offsets < state_len)
            & (~source_is_token)
            & (old_state_position >= 0)
            & (old_state_position < state_len)
            & has_initial
        ),
        other=0.0,
    )
    next_state = token_values + old_state_values
    state_output_ptrs = (
        conv_state_ptr
        + safe_state_index * stride_state_row
        + channel_offsets * stride_state_channel
        + state_offsets * stride_state_token
    )
    tl.store(
        state_output_ptrs,
        next_state,
        mask=(valid_state & valid_channels & (state_offsets < state_len)),
    )


def can_use_ple_gate_norm_triton(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    hc_count: int,
    *,
    min_tokens: int = 0,
) -> bool:
    """Return whether the contiguous inference gate/norm path is supported."""

    tensors = (key, query, value, key_weight, query_weight, conv_weight)
    if not PLE_FUSION_ENABLED or not HAS_TRITON or hc_count <= 0:
        return False
    if key.ndim != 3 or query.shape != key.shape:
        return False
    tokens, branches, hidden_size = key.shape
    if branches != hc_count or tokens < min_tokens or hidden_size <= 0:
        return False
    if value.shape != (tokens, hidden_size):
        return False
    if any(weight.shape != (hc_count * hidden_size,) for weight in tensors[3:]):
        return False
    device = key.device
    return bool(
        key.dtype in _SUPPORTED_DTYPES
        and query.dtype == key.dtype
        and value.dtype == key.dtype
        and device.type not in ("cpu", "meta")
        and all(tensor.device == device for tensor in tensors)
        and all(tensor.dtype in _SUPPORTED_DTYPES for tensor in tensors[3:])
        and all(tensor.is_contiguous() for tensor in tensors)
    )


def qwen3_8_flash_next_ple_gate_norm_(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    gated: torch.Tensor,
    normalized: torch.Tensor,
    hc_count: int,
    eps: float,
) -> None:
    """Mutating custom-op implementation used by compiled model forward."""

    if not can_use_ple_gate_norm_triton(
        key,
        query,
        value,
        key_weight,
        query_weight,
        conv_weight,
        hc_count,
    ):
        raise RuntimeError("PLE gate/norm Triton kernel received unsupported tensors")
    if gated.shape != key.shape or normalized.shape != key.shape:
        raise ValueError("PLE gate/norm output buffers must match key shape")
    if gated.dtype != key.dtype or normalized.dtype != key.dtype:
        raise TypeError("PLE gate/norm output buffers must match key dtype")
    if not gated.is_contiguous() or not normalized.is_contiguous():
        raise ValueError("PLE gate/norm output buffers must be contiguous")
    if not key.numel():
        return

    hidden_size = key.shape[-1]
    block_h = triton.next_power_of_2(hidden_size)
    rows = key.shape[0] * hc_count
    _ple_gate_norm_kernel[(rows,)](
        key,
        query,
        value,
        key_weight,
        query_weight,
        conv_weight,
        gated,
        normalized,
        hidden_size=hidden_size,
        hc_count=hc_count,
        eps=eps,
        inv_sqrt_hidden=1.0 / math.sqrt(hidden_size),
        block_h=block_h,
        num_warps=8 if block_h > 2048 else 4,
    )


def qwen3_8_flash_next_ple_gate_norm_fake(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    gated: torch.Tensor,
    normalized: torch.Tensor,
    hc_count: int,
    eps: float,
) -> None:
    del (
        key,
        query,
        value,
        key_weight,
        query_weight,
        conv_weight,
        gated,
        normalized,
        hc_count,
        eps,
    )


def ple_gate_norm(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the two required results and launch the fused custom op."""

    gated = torch.empty_like(key)
    normalized = torch.empty_like(key)
    torch.ops.vllm.qwen3_8_flash_next_ple_gate_norm_(
        key,
        query,
        value,
        key_weight,
        query_weight,
        conv_weight,
        gated,
        normalized,
        hc_count,
        eps,
    )
    return gated, normalized


def can_use_ple_prefill_conv_triton(
    x: torch.Tensor,
    output: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_prefills: int,
    max_len: int,
    state_len: int,
    kernel_width: int,
    dilation: int,
    min_tokens: int = 0,
) -> bool:
    """Return whether PLE prefill can bypass pack/layout/native conv ops."""

    if not PLE_FUSION_ENABLED or not HAS_TRITON:
        return False
    if (
        x.ndim != 2
        or output.shape != x.shape
        or conv_weight.ndim != 2
        or conv_weight.shape != (x.shape[1], kernel_width)
        or x.shape[0] < min_tokens
        or num_prefills <= 0
        or max_len <= 0
        or kernel_width <= 0
        or kernel_width > 8
        or dilation <= 0
        or state_len != (kernel_width - 1) * dilation
    ):
        return False
    if query_start_loc.ndim != 1 or query_start_loc.numel() < num_prefills + 1:
        return False
    if state_indices.ndim != 1 or state_indices.numel() < num_prefills:
        return False
    if has_initial_state.ndim != 1 or has_initial_state.numel() < num_prefills:
        return False
    if conv_state.ndim != 3:
        return False
    if conv_state.shape[0] and (
        conv_state.shape[1] != x.shape[1] or conv_state.shape[2] < state_len
    ):
        return False
    device = x.device
    tensors = (
        output,
        conv_state,
        conv_weight,
        query_start_loc,
        state_indices,
        has_initial_state,
    )
    return bool(
        x.dtype in _SUPPORTED_DTYPES
        and output.dtype == x.dtype
        and conv_weight.dtype == x.dtype
        and conv_state.dtype in _SUPPORTED_DTYPES
        and device.type not in ("cpu", "meta")
        and all(tensor.device == device for tensor in tensors)
        and x.is_contiguous()
        and output.is_contiguous()
        and conv_weight.is_contiguous()
    )


def ple_prefill_short_conv_(
    x: torch.Tensor,
    output: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_prefills: int,
    max_len: int,
    state_len: int,
    kernel_width: int,
    dilation: int,
    null_block_id: int,
    min_tokens: int = 0,
    block_t: int | None = None,
    block_c: int | None = None,
) -> bool:
    """Write PLE prefill conv output/state directly; return whether fused."""

    if not can_use_ple_prefill_conv_triton(
        x,
        output,
        conv_state,
        conv_weight,
        query_start_loc,
        state_indices,
        has_initial_state,
        num_prefills=num_prefills,
        max_len=max_len,
        state_len=state_len,
        kernel_width=kernel_width,
        dilation=dilation,
        min_tokens=min_tokens,
    ):
        return False
    if block_t is None or block_c is None:
        if block_t is not None or block_c is not None:
            raise ValueError("PLE prefill Triton block sizes must be set together")
        block_t, block_c = _default_prefill_conv_config(x.device)
    if block_t <= 0 or block_c <= 0:
        raise ValueError("PLE prefill Triton block sizes must be positive")
    if block_t & (block_t - 1) or block_c & (block_c - 1):
        raise ValueError("PLE prefill Triton block sizes must be powers of two")

    has_state = conv_state.shape[0] > 0 and state_len > 0
    _ple_prefill_dilated_conv_kernel[
        (
            num_prefills,
            triton.cdiv(max_len, block_t),
            triton.cdiv(x.shape[1], block_c),
        )
    ](
        x,
        conv_weight,
        query_start_loc,
        state_indices,
        has_initial_state,
        conv_state,
        output,
        num_state_rows=conv_state.shape[0],
        channels=x.shape[1],
        state_len=state_len,
        stride_x_token=x.stride(0),
        stride_x_channel=x.stride(1),
        stride_weight_channel=conv_weight.stride(0),
        stride_weight_width=conv_weight.stride(1),
        stride_query_start=query_start_loc.stride(0),
        stride_state_index=state_indices.stride(0),
        stride_has_initial=has_initial_state.stride(0),
        stride_state_row=conv_state.stride(0),
        stride_state_channel=conv_state.stride(1),
        stride_state_token=conv_state.stride(2),
        stride_output_token=output.stride(0),
        stride_output_channel=output.stride(1),
        null_block_id=null_block_id,
        kernel_width=kernel_width,
        dilation=dilation,
        has_state=has_state,
        block_t=block_t,
        block_c=block_c,
        num_warps=8,
    )
    # State writes must be a second launch: updating in a peer CTA of the
    # output launch would race output CTAs that still consume the old state.
    if has_state:
        block_s = triton.next_power_of_2(state_len)
        _ple_prefill_state_update_kernel[
            (num_prefills, triton.cdiv(x.shape[1], block_c))
        ](
            x,
            query_start_loc,
            state_indices,
            has_initial_state,
            conv_state,
            num_state_rows=conv_state.shape[0],
            channels=x.shape[1],
            state_len=state_len,
            stride_x_token=x.stride(0),
            stride_x_channel=x.stride(1),
            stride_query_start=query_start_loc.stride(0),
            stride_state_index=state_indices.stride(0),
            stride_has_initial=has_initial_state.stride(0),
            stride_state_row=conv_state.stride(0),
            stride_state_channel=conv_state.stride(1),
            stride_state_token=conv_state.stride(2),
            null_block_id=null_block_id,
            block_s=block_s,
            block_c=block_c,
            num_warps=8,
        )
    return True


direct_register_custom_op(
    op_name="qwen3_8_flash_next_ple_gate_norm_",
    op_func=qwen3_8_flash_next_ple_gate_norm_,
    mutates_args=["gated", "normalized"],
    fake_impl=qwen3_8_flash_next_ple_gate_norm_fake,
)


__all__ = [
    "PLE_FUSION_ENABLED",
    "PLE_FUSION_MIN_TOKENS",
    "can_use_ple_gate_norm_triton",
    "can_use_ple_prefill_conv_triton",
    "ple_gate_norm",
    "ple_prefill_short_conv_",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-vendor row I/O for the Qwen3.8-Flash-Next PLE short-conv state.

The vLLM short-conv cache is exposed as a non-contiguous logical
``[rows, hidden, width]`` view of physical ``[rows, width, hidden]`` storage.
FlagGems 5.3/5.4 materializes that whole multi-GiB view for ``index_select``.
The Qwen3.8-Flash-Next worker therefore excludes that operator from FlagGems.
Eager gather uses native ATen, while CUDA/HIP graph capture uses a compact
stride-aware Triton gather: H100 A/B shows that the preferred implementation
reverses between those two modes. Scatter retains a vendor-neutral
stride-aware Triton kernel because a per-row ATen fallback creates too many
tiny graph nodes at production batch sizes. FlagTree provides the Triton
backend for supported accelerators.

The scatter write mask supplied by PLE prevents NULL/padded rows from writing
the reserved row zero; out-of-range rows are ignored.  When no mask is given,
the helper creates one that retains only the final occurrence of each row.
The non-Triton fallback has a fixed shape-derived trip count and performs no
device-to-host synchronization, so it remains CUDA-graph capturable.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


_BLOCK_SIZE = 1024
_NUM_WARPS = 8


def _use_ple_triton_kernels() -> bool:
    """Allow the PLE row-I/O kernels to be disabled for safe fallback.

    The default remains enabled.  Setting ``QWEN4_PLE_TRITON=0`` selects the
    graph-safe ATen implementations for both gather and scatter.
    """

    return os.getenv("QWEN4_PLE_TRITON", "1").lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _is_triton_device(tensor: torch.Tensor) -> bool:
    return HAS_TRITON and tensor.device.type not in ("cpu", "meta")


def _is_cuda_graph_capture(tensor: torch.Tensor) -> bool:
    """CUDA and ROCm PyTorch expose graph capture through ``torch.cuda``."""

    if tensor.device.type != "cuda":
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except (AttributeError, RuntimeError):
        return False


if HAS_TRITON:

    @triton.jit
    def _ple_state_gather_kernel_3d(
        state_ptr,
        indices_ptr,
        output_ptr,
        indices_stride,
        state_stride0,
        state_stride1,
        state_stride2,
        output_stride0,
        output_stride1,
        output_stride2,
        num_cache_rows,
        hidden_size,
        state_width,
        HIDDEN_FASTEST: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < hidden_size * state_width
        index = tl.load(indices_ptr + row * indices_stride).to(tl.int64)
        valid_index = (index >= 0) & (index < num_cache_rows)
        safe_index = tl.minimum(tl.maximum(index, 0), num_cache_rows - 1)
        if HIDDEN_FASTEST:
            hidden = offsets % hidden_size
            width = offsets // hidden_size
        else:
            hidden = offsets // state_width
            width = offsets % state_width
        source = (
            state_ptr
            + safe_index * state_stride0
            + hidden * state_stride1
            + width * state_stride2
        )
        destination = (
            output_ptr
            + row * output_stride0
            + hidden * output_stride1
            + width * output_stride2
        )
        values = tl.load(source, mask=valid & valid_index, other=0)
        tl.store(destination, values, mask=valid)

    @triton.jit
    def _ple_state_scatter_kernel_3d(
        state_ptr,
        indices_ptr,
        rows_ptr,
        indices_stride,
        write_mask_ptr,
        write_mask_stride,
        state_stride0,
        state_stride1,
        state_stride2,
        rows_stride0,
        rows_stride1,
        rows_stride2,
        num_cache_rows,
        hidden_size,
        state_width,
        HIDDEN_FASTEST: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_elements = hidden_size * state_width
        valid = offsets < row_elements
        valid &= tl.load(write_mask_ptr + row * write_mask_stride)
        index = tl.load(indices_ptr + row * indices_stride).to(tl.int64)
        valid &= (index >= 0) & (index < num_cache_rows)
        safe_index = tl.minimum(tl.maximum(index, 0), num_cache_rows - 1)
        if HIDDEN_FASTEST:
            hidden = offsets % hidden_size
            width = offsets // hidden_size
        else:
            hidden = offsets // state_width
            width = offsets % state_width
        source = (
            rows_ptr
            + row * rows_stride0
            + hidden * rows_stride1
            + width * rows_stride2
        )
        destination = (
            state_ptr
            + safe_index * state_stride0
            + hidden * state_stride1
            + width * state_stride2
        )
        values = tl.load(source, mask=valid, other=0)
        tl.store(destination, values, mask=valid)

else:  # pragma: no cover - exercised by import-only/CPU environments
    _ple_state_gather_kernel_3d = None
    _ple_state_scatter_kernel_3d = None


def _validate_state_tensor(state: torch.Tensor, name: str) -> None:
    if state.ndim != 3:
        raise ValueError(f"PLE {name} must be a rank-3 [rows, hidden, width] tensor")
    if state.numel() and not state.is_floating_point():
        raise TypeError(f"PLE {name} must have a floating-point dtype")


def _safe_state_indices(
    state: torch.Tensor,
    indices: torch.Tensor,
    *,
    indices_are_safe: bool,
) -> torch.Tensor:
    if indices.ndim != 1:
        raise ValueError("PLE state indices must be a one-dimensional tensor")
    # Keep this operation on-device.  In particular, do not inspect a CUDA
    # value on the host: all three callers are CUDA-graph captured paths.
    indices = indices.to(device=state.device, dtype=torch.int64)
    if indices_are_safe:
        return indices
    return torch.where(
        indices == NULL_BLOCK_ID,
        torch.zeros_like(indices),
        indices,
    )


def _empty_state_rows_like(state: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Allocate compact rows with the same dense inner layout as ``state``."""

    if state.stride(1) <= state.stride(2):
        return torch.empty(
            (num_rows, state.shape[2], state.shape[1]),
            dtype=state.dtype,
            device=state.device,
        ).transpose(1, 2)
    return torch.empty(
        (num_rows, state.shape[1], state.shape[2]),
        dtype=state.dtype,
        device=state.device,
    )


def _last_write_mask(indices: torch.Tensor) -> torch.Tensor:
    """Return a device-side mask retaining the last duplicate row."""

    if indices.numel() <= 1:
        return torch.ones_like(indices, dtype=torch.bool)
    duplicate_later = (
        indices.unsqueeze(-1) == indices.unsqueeze(0)
    ).triu(diagonal=1).any(dim=1)
    return ~duplicate_later


def ple_state_gather(
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor | None = None,
    *,
    indices_are_safe: bool = False,
) -> torch.Tensor:
    """Gather requested cache rows without materializing the full cache."""

    _validate_state_tensor(state, "state")
    safe_indices = _safe_state_indices(
        state, indices, indices_are_safe=indices_are_safe
    )
    expected_shape = (safe_indices.numel(),) + tuple(state.shape[1:])
    if output is not None:
        _validate_state_tensor(output, "state output")
        if output.shape != expected_shape:
            raise ValueError(
                "PLE state gather output has an invalid shape: "
                f"got {tuple(output.shape)}, expected {expected_shape}"
            )
        if output.device != state.device or output.dtype != state.dtype:
            raise ValueError(
                "PLE state gather output must match state device and dtype"
            )

    if not safe_indices.numel() or state.shape[1] == 0 or state.shape[2] == 0:
        if output is None:
            output = _empty_state_rows_like(state, safe_indices.numel())
        return output
    if state.shape[0] == 0:
        if output is None:
            output = _empty_state_rows_like(state, safe_indices.numel())
        output.zero_()
        return output

    # Native ATen wins in eager execution, but the single stride-aware Triton
    # node is faster once the serving path is captured into a CUDA/HIP graph.
    # The branch is evaluated while vLLM captures the graph; replay therefore
    # contains only the selected kernel and performs no host-side query.
    if (
        _use_ple_triton_kernels()
        and _is_cuda_graph_capture(state)
        and _is_triton_device(state)
        and _ple_state_gather_kernel_3d is not None
        and state.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        if output is None:
            output = _empty_state_rows_like(state, safe_indices.numel())
        row_elements = state.shape[1] * state.shape[2]
        _ple_state_gather_kernel_3d[
            (safe_indices.numel(), triton.cdiv(row_elements, _BLOCK_SIZE))
        ](
            state,
            safe_indices,
            output,
            safe_indices.stride(0),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            state.shape[0],
            state.shape[1],
            state.shape[2],
            HIDDEN_FASTEST=state.stride(1) <= state.stride(2),
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
        return output

    if indices_are_safe:
        # Production PLE metadata has already mapped NULL to zero and bounded
        # every physical cache row.  Return native ATen's compact result
        # directly: this is the measured fast path and avoids an extra copy.
        gathered = torch.ops.aten.index_select.default(state, 0, safe_indices)
    else:
        valid_indices = (safe_indices >= 0) & (safe_indices < state.shape[0])
        bounded_indices = safe_indices.clamp(0, state.shape[0] - 1)
        gathered = torch.ops.aten.index_select.default(state, 0, bounded_indices)
        gathered = torch.where(
            valid_indices.view(-1, 1, 1), gathered, torch.zeros_like(gathered)
        )
    if output is None:
        return gathered
    output.copy_(gathered)
    return output


def ple_state_scatter_(
    state: torch.Tensor,
    indices: torch.Tensor,
    rows: torch.Tensor,
    *,
    write_mask: torch.Tensor | None = None,
    indices_are_safe: bool = False,
) -> torch.Tensor:
    """Write cache rows through a stride-aware mask.

    Callers that provide ``write_mask`` are responsible for masking earlier
    duplicate destinations.  PLE metadata satisfies this invariant.  Without
    an explicit mask, this helper derives deterministic last-writer semantics.
    """

    _validate_state_tensor(state, "state")
    _validate_state_tensor(rows, "state rows")
    if rows.shape[0] != indices.numel() or tuple(rows.shape[1:]) != tuple(
        state.shape[1:]
    ):
        raise ValueError(
            "PLE state scatter rows and indices have incompatible shapes: "
            f"rows={tuple(rows.shape)}, indices={indices.numel()}, "
            f"state={tuple(state.shape)}"
        )
    if rows.device != state.device or rows.dtype != state.dtype:
        raise ValueError("PLE state scatter rows must match state device and dtype")

    safe_indices = _safe_state_indices(
        state, indices, indices_are_safe=indices_are_safe
    )
    if write_mask is None:
        write_mask = _last_write_mask(safe_indices)
    else:
        if write_mask.ndim != 1 or write_mask.numel() != safe_indices.numel():
            raise ValueError("PLE state scatter write_mask must match indices")
        write_mask = write_mask.to(device=state.device, dtype=torch.bool)

    if (
        not safe_indices.numel()
        or not state.shape[0]
        or state.shape[1] == 0
        or state.shape[2] == 0
    ):
        return state

    if (
        _use_ple_triton_kernels()
        and _is_triton_device(state)
        and _ple_state_scatter_kernel_3d is not None
        and state.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        row_elements = state.shape[1] * state.shape[2]
        _ple_state_scatter_kernel_3d[
            (safe_indices.numel(), triton.cdiv(row_elements, _BLOCK_SIZE))
        ](
            state,
            safe_indices,
            rows,
            safe_indices.stride(0),
            write_mask,
            write_mask.stride(0),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            rows.stride(0),
            rows.stride(1),
            rows.stride(2),
            state.shape[0],
            state.shape[1],
            state.shape[2],
            HIDDEN_FASTEST=state.stride(1) <= state.stride(2),
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
        return state

    valid_indices = (safe_indices >= 0) & (safe_indices < state.shape[0])
    bounded_indices = safe_indices.clamp(0, state.shape[0] - 1)
    effective_write_mask = write_mask & valid_indices
    # A single bulk index_copy_ has undefined behavior for duplicate indices.
    # Ordered one-row writes preserve last-writer-wins and let masked NULL rows
    # retain row zero without compacting metadata into a dynamic output shape.
    for row in range(safe_indices.numel()):
        index = bounded_indices[row : row + 1]
        existing = torch.ops.aten.index_select.default(state, 0, index)
        selected = torch.where(
            effective_write_mask[row], rows[row : row + 1], existing
        )
        torch.ops.aten.index_copy_.default(state, 0, index, selected)
    return state


__all__ = ["ple_state_gather", "ple_state_scatter_"]

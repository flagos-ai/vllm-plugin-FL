# SPDX-License-Identifier: Apache-2.0
"""HYV4 hyper-connection pointwise/reduction fusion.

The HYV4 hyper-connection uses an FP32 projection followed by a small gate
post-processing chain.  This module deliberately leaves the projection and
the RMS-style reduction in PyTorch: their FP32 accumulation order is part of
the model's reference contract.  The optional Triton kernels only consume the
projection output and fuse the cheap work that follows it:

``split -> scale/base -> sigmoid -> read collapse`` and the residual
``writeback``.

The kernels use self-locating two-dimensional grids and pass all strides as
runtime values.  They therefore do not need host metadata and can be replayed
from a fixed-address CUDA graph.  The validated HY4 path is enabled by default;
set ``VLLM_HY4_HC_POINTWISE_FUSION=0`` for a literal Torch rollback.
Unsupported devices, dtypes, and capture-before-warmup still use the fallback.
"""

from __future__ import annotations

import os
from typing import Final

import torch

try:  # Triton is optional for CPU/unit-test imports.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installations.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


HYV4_HC_FUSION_ENV: Final[str] = "VLLM_HY4_HC_POINTWISE_FUSION"
# Short spelling retained for launch scripts that use one switch per model.
HYV4_HC_FUSION_LEGACY_ENV: Final[str] = "VLLM_HY4_HC_FUSION"
# Keep the plugin-prefixed spelling as a compatibility alias for experiments
# that use the general vllm-fl namespace.
HYV4_HC_FUSION_ALIAS_ENV: Final[str] = "VLLM_FL_HY4_HC_FUSION"

_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_READ_WARMED: set[tuple[int, str, int, int]] = set()
_WRITE_WARMED: set[tuple[int, str, str, int, int]] = set()


def _env_enabled() -> bool:
    """Return whether the default-on HYV4 HC fusion remains enabled."""
    for name in (
        HYV4_HC_FUSION_ENV,
        HYV4_HC_FUSION_LEGACY_ENV,
        HYV4_HC_FUSION_ALIAS_ENV,
    ):
        value = os.getenv(name)
        if value is not None:
            return value.strip().lower() in _TRUE_VALUES
    return True


def _is_capturing(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return bool(torch.cuda.is_current_stream_capturing())


def _device_key(device: torch.device) -> int:
    # ``device.index`` is None for the current CUDA device.  Looking it up is
    # a host-side query, but it happens only at dispatch, never in the kernel
    # or from a graph-captured body.  The active device is stable during a
    # graph replay.
    return (
        torch.device(device).index
        if device.index is not None
        else torch.cuda.current_device()
    )


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _read_block_h(hidden_size: int) -> int:
    # Capping avoids an enormous register tile for HY4's 6144-wide hidden
    # state while the grid's second axis handles the remaining feature tiles.
    return min(1024, _next_power_of_two(hidden_size))


def _write_block_h(hidden_size: int) -> int:
    # Writeback has a larger stream-by-feature live tile than the read
    # reduction.  Keep its tile at 256 for wide HY4 hidden states to reduce
    # register pressure and launch tail cost.
    return min(256, _next_power_of_two(hidden_size))


def _num_warps(block_h: int) -> int:
    if block_h >= 128:
        return 4
    if block_h >= 64:
        return 2
    return 1


def _read_key(
    streams: torch.Tensor, num_streams: int, block_h: int
) -> tuple[int, str, int, int]:
    return (_device_key(streams.device), str(streams.dtype), num_streams, block_h)


def _write_key(
    streams: torch.Tensor, delta: torch.Tensor, num_streams: int, block_h: int
) -> tuple[int, str, str, int, int]:
    return (
        _device_key(streams.device),
        str(streams.dtype),
        str(delta.dtype),
        num_streams,
        block_h,
    )


def _read_inputs_are_supported(
    streams: torch.Tensor,
    gates: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    num_streams: int,
) -> bool:
    if triton is None or streams.device.type != "cuda":
        return False
    if streams.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if (
        gates.dtype != torch.float32
        or scale.dtype != torch.float32
        or base.dtype != torch.float32
    ):
        return False
    if streams.ndim != 3 or gates.ndim != 2:
        return False
    rows, stream_count, hidden = streams.shape
    return (
        rows > 0
        and stream_count == num_streams
        and hidden > 0
        and gates.shape == (rows, 2 * num_streams)
        and scale.numel() == 2
        and base.numel() == 2 * num_streams
        and streams.device == gates.device == scale.device == base.device
    )


def _write_inputs_are_supported(
    streams: torch.Tensor,
    delta: torch.Tensor,
    write: torch.Tensor,
    num_streams: int,
) -> bool:
    if triton is None or streams.device.type != "cuda":
        return False
    if streams.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if delta.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if (
        write.dtype != torch.float32
        or streams.ndim != 3
        or delta.ndim != 2
        or write.ndim != 2
    ):
        return False
    rows, stream_count, hidden = streams.shape
    return (
        rows > 0
        and stream_count == num_streams
        and hidden > 0
        and delta.shape == (rows, hidden)
        and write.shape == (rows, num_streams)
        and streams.device == delta.device == write.device
    )


def _torch_read_post_linear(
    streams: torch.Tensor,
    gates: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    eps: float,
    num_streams: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal Torch implementation of the fused post-linear read chain."""
    read, write = gates.split(num_streams, dim=-1)
    read = torch.sigmoid(read * scale[0] + base[:num_streams])
    write = magnitude * torch.sigmoid(write * scale[1] + base[num_streams:])
    read = read + eps
    write = write + eps
    collapsed = (read.unsqueeze(-1) * streams.float()).sum(dim=1)
    return collapsed.to(streams.dtype), write


def _torch_writeback(
    streams: torch.Tensor,
    delta: torch.Tensor,
    write: torch.Tensor,
) -> torch.Tensor:
    """Literal Torch implementation of the fused residual writeback."""
    return (
        streams.float() + write.float().unsqueeze(-1) * delta.float().unsqueeze(1)
    ).to(delta.dtype)


if triton is not None:

    @triton.jit
    def _hyv4_hc_read_kernel(
        streams_ptr,
        gates_ptr,
        scale_ptr,
        base_ptr,
        collapsed_ptr,
        write_ptr,
        rows,
        num_streams,
        hidden_size,
        streams_stride_0,
        streams_stride_1,
        streams_stride_2,
        gates_stride_0,
        gates_stride_1,
        collapsed_stride_0,
        collapsed_stride_1,
        write_stride_0,
        write_stride_1,
        magnitude,
        eps,
        BLOCK_H: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row = tl.program_id(0)
        block_h = tl.program_id(1)
        h = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
        s = tl.arange(0, BLOCK_S)
        h_mask = h < hidden_size
        s_mask = s < num_streams

        gate_read_ptrs = gates_ptr + row * gates_stride_0 + s * gates_stride_1
        gate_write_ptrs = (
            gates_ptr + row * gates_stride_0 + (num_streams + s) * gates_stride_1
        )
        gate_read_linear = tl.load(gate_read_ptrs, mask=s_mask, other=0.0)
        gate_write_linear = tl.load(gate_write_ptrs, mask=s_mask, other=0.0)
        scale_read = tl.load(scale_ptr)
        scale_write = tl.load(scale_ptr + 1)
        base_read = tl.load(base_ptr + s, mask=s_mask, other=0.0)
        base_write = tl.load(base_ptr + num_streams + s, mask=s_mask, other=0.0)
        read = tl.sigmoid(gate_read_linear * scale_read + base_read) + eps
        write = (
            magnitude * tl.sigmoid(gate_write_linear * scale_write + base_write) + eps
        )

        stream_ptrs = (
            streams_ptr
            + row * streams_stride_0
            + s[:, None] * streams_stride_1
            + h[None, :] * streams_stride_2
        )
        stream_mask = s_mask[:, None] & h_mask[None, :]
        stream_values = tl.load(stream_ptrs, mask=stream_mask, other=0.0).to(tl.float32)
        collapsed = tl.sum(read[:, None] * stream_values, axis=0)
        tl.store(
            collapsed_ptr + row * collapsed_stride_0 + h * collapsed_stride_1,
            collapsed,
            mask=h_mask,
        )

        # Only the first feature tile owns the write-gate stores.  This keeps
        # the grid fully self-locating without concurrent duplicate stores.
        if block_h == 0:
            tl.store(
                write_ptr + row * write_stride_0 + s * write_stride_1,
                write,
                mask=s_mask,
            )

    @triton.jit
    def _hyv4_hc_write_kernel(
        streams_ptr,
        delta_ptr,
        write_ptr,
        output_ptr,
        rows,
        num_streams,
        hidden_size,
        streams_stride_0,
        streams_stride_1,
        streams_stride_2,
        delta_stride_0,
        delta_stride_1,
        write_stride_0,
        write_stride_1,
        output_stride_0,
        output_stride_1,
        output_stride_2,
        BLOCK_H: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row = tl.program_id(0)
        block_h = tl.program_id(1)
        h = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
        s = tl.arange(0, BLOCK_S)
        h_mask = h < hidden_size
        s_mask = s < num_streams
        mask = s_mask[:, None] & h_mask[None, :]

        stream_ptrs = (
            streams_ptr
            + row * streams_stride_0
            + s[:, None] * streams_stride_1
            + h[None, :] * streams_stride_2
        )
        delta_ptrs = delta_ptr + row * delta_stride_0 + h * delta_stride_1
        write_ptrs = write_ptr + row * write_stride_0 + s * write_stride_1
        output_ptrs = (
            output_ptr
            + row * output_stride_0
            + s[:, None] * output_stride_1
            + h[None, :] * output_stride_2
        )
        stream_values = tl.load(stream_ptrs, mask=mask, other=0.0).to(tl.float32)
        delta_values = tl.load(delta_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        write_values = tl.load(write_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        result = stream_values + write_values[:, None] * delta_values[None, :]
        tl.store(output_ptrs, result, mask=mask)


def _read_post_linear_impl(
    streams: torch.Tensor,
    gates: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    eps: float,
    num_streams: int,
    *,
    use_fusion: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse HYV4 read gates and stream collapse when the guarded path applies.

    ``gates`` must be the FP32 result of the existing ``F.linear`` and norm
    chain.  No projection is reimplemented here, which preserves the exact
    reference GEMM/reduction semantics.  All tensor strides are forwarded to
    Triton, including a non-contiguous/expanded stream input.
    """
    if use_fusion is None:
        use_fusion = _env_enabled()
    if not use_fusion or not _read_inputs_are_supported(
        streams, gates, scale, base, num_streams
    ):
        return _torch_read_post_linear(
            streams, gates, scale, base, magnitude, eps, num_streams
        )

    rows, _, hidden_size = streams.shape
    block_h = _read_block_h(hidden_size)
    block_s = _next_power_of_two(num_streams)
    key = _read_key(streams, num_streams, block_h)
    if _is_capturing(streams.device) and key not in _READ_WARMED:
        return _torch_read_post_linear(
            streams, gates, scale, base, magnitude, eps, num_streams
        )

    collapsed = torch.empty(
        (rows, hidden_size), dtype=streams.dtype, device=streams.device
    )
    write = torch.empty((rows, num_streams), dtype=torch.float32, device=streams.device)
    grid = (rows, triton.cdiv(hidden_size, block_h))
    _hyv4_hc_read_kernel[grid](
        streams,
        gates,
        scale,
        base,
        collapsed,
        write,
        rows,
        num_streams,
        hidden_size,
        streams.stride(0),
        streams.stride(1),
        streams.stride(2),
        gates.stride(0),
        gates.stride(1),
        collapsed.stride(0),
        collapsed.stride(1),
        write.stride(0),
        write.stride(1),
        float(magnitude),
        float(eps),
        BLOCK_H=block_h,
        BLOCK_S=block_s,
        num_warps=_num_warps(block_h),
        num_stages=1,
    )
    if not _is_capturing(streams.device):
        _READ_WARMED.add(key)
    return collapsed, write


def _writeback_impl(
    streams: torch.Tensor,
    delta: torch.Tensor,
    write: torch.Tensor,
    num_streams: int,
    *,
    use_fusion: bool | None = None,
) -> torch.Tensor:
    """Fuse HYV4 residual writeback, retaining the Torch fallback."""
    if use_fusion is None:
        use_fusion = _env_enabled()
    if not use_fusion or not _write_inputs_are_supported(
        streams, delta, write, num_streams
    ):
        return _torch_writeback(streams, delta, write)

    rows, _, hidden_size = streams.shape
    block_h = _write_block_h(hidden_size)
    block_s = _next_power_of_two(num_streams)
    key = _write_key(streams, delta, num_streams, block_h)
    if _is_capturing(streams.device) and key not in _WRITE_WARMED:
        return _torch_writeback(streams, delta, write)

    # The original implementation casts the residual result to ``delta``'s
    # dtype (normally BF16, but retaining this matters for the mixed-dtype
    # reference path and tests).
    output = torch.empty(streams.shape, dtype=delta.dtype, device=streams.device)
    grid = (rows, triton.cdiv(hidden_size, block_h))
    _hyv4_hc_write_kernel[grid](
        streams,
        delta,
        write,
        output,
        rows,
        num_streams,
        hidden_size,
        streams.stride(0),
        streams.stride(1),
        streams.stride(2),
        delta.stride(0),
        delta.stride(1),
        write.stride(0),
        write.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_H=block_h,
        BLOCK_S=block_s,
        num_warps=_num_warps(block_h),
        num_stages=1,
    )
    if not _is_capturing(streams.device):
        _WRITE_WARMED.add(key)
    return output


def _read_post_linear_fake(
    streams: torch.Tensor,
    gates: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    eps: float,
    num_streams: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del gates, scale, base, magnitude, eps
    rows, _, hidden_size = streams.shape
    return (
        torch.empty((rows, hidden_size), dtype=streams.dtype, device=streams.device),
        torch.empty((rows, num_streams), dtype=torch.float32, device=streams.device),
    )


def _writeback_fake(
    streams: torch.Tensor,
    delta: torch.Tensor,
    write: torch.Tensor,
    num_streams: int,
) -> torch.Tensor:
    del write, num_streams
    return torch.empty(streams.shape, dtype=delta.dtype, device=streams.device)


def _read_post_linear_op(
    streams: torch.Tensor,
    gates: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    eps: float,
    num_streams: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _read_post_linear_impl(
        streams,
        gates,
        scale,
        base,
        magnitude,
        eps,
        num_streams,
        use_fusion=None,
    )


def _writeback_op(
    streams: torch.Tensor,
    delta: torch.Tensor,
    write: torch.Tensor,
    num_streams: int,
) -> torch.Tensor:
    return _writeback_impl(streams, delta, write, num_streams, use_fusion=None)


def read_post_linear(
    streams: torch.Tensor,
    gates: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    eps: float,
    num_streams: int,
    *,
    use_fusion: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch through the vLLM custom op when available.

    The custom-op boundary is important for Dynamo: its fake implementation
    carries only shape/dtype information through tracing, while the eager
    implementation below performs the env-gated Triton/Torch dispatch.  A
    direct call with ``use_fusion`` is retained for focused kernel tests and
    for installations that do not provide vLLM's registration helper.
    """
    if use_fusion is None and _CUSTOM_OP_REGISTERED and streams.device.type == "cuda":
        return torch.ops.vllm.hy_v4_hc_read_post_linear(
            streams,
            gates,
            scale,
            base,
            float(magnitude),
            float(eps),
            int(num_streams),
        )
    return _read_post_linear_impl(
        streams,
        gates,
        scale,
        base,
        magnitude,
        eps,
        num_streams,
        use_fusion=use_fusion,
    )


def writeback(
    streams: torch.Tensor,
    delta: torch.Tensor,
    write: torch.Tensor,
    num_streams: int,
    *,
    use_fusion: bool | None = None,
) -> torch.Tensor:
    """Dispatch residual writeback through the graph-safe custom op."""
    if use_fusion is None and _CUSTOM_OP_REGISTERED and streams.device.type == "cuda":
        return torch.ops.vllm.hy_v4_hc_writeback(
            streams, delta, write, int(num_streams)
        )
    return _writeback_impl(streams, delta, write, num_streams, use_fusion=use_fusion)


_CUSTOM_OP_REGISTERED = False
try:
    # vLLM's direct registration gives Dynamo/Inductor a fake implementation
    # and keeps this model-specific operator opaque during torch.compile.
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="hy_v4_hc_read_post_linear",
        op_func=_read_post_linear_op,
        mutates_args=[],
        fake_impl=_read_post_linear_fake,
    )
    direct_register_custom_op(
        op_name="hy_v4_hc_writeback",
        op_func=_writeback_op,
        mutates_args=[],
        fake_impl=_writeback_fake,
    )
    _CUSTOM_OP_REGISTERED = True
except (ImportError, AttributeError, RuntimeError, TypeError):
    # Unit tests and CPU-only plugin imports may not have vLLM's registration
    # helper; the wrapper remains fully functional through the direct path.
    pass


__all__ = [
    "HYV4_HC_FUSION_ALIAS_ENV",
    "HYV4_HC_FUSION_ENV",
    "HYV4_HC_FUSION_LEGACY_ENV",
    "read_post_linear",
    "writeback",
]

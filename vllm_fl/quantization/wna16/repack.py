# Copyright (c) 2026 BAAI. All rights reserved.

"""Reusable compressed-tensors W4A16 weight layout conversion."""

from __future__ import annotations

import torch


def repack_uint4_kpacked_to_npacked(
    weight_packed: torch.Tensor,
) -> torch.Tensor:
    """
    Convert compressed-tensors uint4 layout:

        [..., N, K // 8]

    to the layout required by vLLM Triton W4A16:

        [..., K, N // 8]

    The leading dimensions are preserved, allowing the same implementation
    to be extended to batched or expert weights.
    """
    if weight_packed.dtype != torch.int32:
        raise TypeError(
            "weight_packed must have dtype torch.int32, "
            f"got {weight_packed.dtype}"
        )

    if weight_packed.ndim < 2:
        raise ValueError(
            "weight_packed must have at least two dimensions"
        )

    output_size = weight_packed.shape[-2]
    packed_input_size = weight_packed.shape[-1]
    input_size = packed_input_size * 8

    if output_size % 8 != 0:
        raise ValueError(
            "W4A16 repack requires output size divisible by 8, "
            f"got {output_size}"
        )

    batch_shape = weight_packed.shape[:-2]

    shifts = torch.arange(
        8,
        dtype=torch.int32,
        device=weight_packed.device,
    ) * 4

    # [..., N, K // 8] -> [..., N, K]
    unpacked = (
        (
            weight_packed.contiguous().unsqueeze(-1)
            >> shifts
        )
        & 0xF
    ).reshape(
        *batch_shape,
        output_size,
        input_size,
    )

    # [..., N, K] -> [..., K, N]
    weight_kn = unpacked.transpose(-2, -1).contiguous()

    # [..., K, N] -> [..., K, N // 8, 8]
    nibble_view = weight_kn.reshape(
        *batch_shape,
        input_size,
        output_size // 8,
        8,
    )

    # The eight nibbles occupy disjoint bit ranges. Bitwise OR is therefore
    # exactly equivalent to sum(..., dim=-1), without invoking a reduction.
    repacked = torch.bitwise_and(
        nibble_view[..., 0],
        0xF,
    ).to(torch.int32)

    for index in range(1, 8):
        nibble = torch.bitwise_and(
            nibble_view[..., index],
            0xF,
        )
        nibble = torch.bitwise_left_shift(
            nibble,
            index * 4,
        )
        repacked = torch.bitwise_or(
            repacked,
            nibble,
        )

    return repacked.contiguous()


__all__ = [
    "repack_uint4_kpacked_to_npacked",
]
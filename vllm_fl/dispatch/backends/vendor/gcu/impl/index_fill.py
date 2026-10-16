# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep FlagGems index_fill launches within the Enflame grid limits."""

from __future__ import annotations

import importlib
import inspect
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from math import prod

# The private launcher/kernel contract is supported for this release only.
# CI verifies exact release bytes separately in docker/enflame/verify_index_fill.py;
# comments, formatting and source paths are not runtime API requirements.
_SUPPORTED_VERSION = "5.4.0"
_MAX_GRID_X = 65535
_MAX_GRID_Y = 255
_PATCH_STATE = "_fl_gcu_index_fill_patch"
_LAUNCH_PARAMETERS = ("out", "dim", "index", "value", "value_is_tensor", "fill_numel")
_KERNEL_PARAMETERS = (
    "out",
    "index",
    "value",
    "N",
    "index_len",
    "dim_size",
    *(f"shape_{i}" for i in range(8)),
    *(f"stride_{i}" for i in range(8)),
    "DIM",
    "RANK",
    "VALUE_IS_TENSOR",
    "BLOCK_SIZE",
)
_KERNEL_CONSTEXPRS = (22, 23, 24, 25)


def _tiles(outer: int, inner: int, index_len: int, capacity: int):
    """Partition the fill space without changing indices or increasing blocks."""
    if min(outer, inner, index_len, capacity) <= 0:
        raise ValueError("Index-fill tile dimensions must be positive")
    inner_step = min(inner, capacity)
    for inner_start in range(0, inner, inner_step):
        inner_count = min(inner_step, inner - inner_start)
        outer_step = min(outer, capacity // inner_count)
        for outer_start in range(0, outer, outer_step):
            outer_count = min(outer_step, outer - outer_start)
            index_step = min(index_len, capacity // (inner_count * outer_count))
            for index_start in range(0, index_len, index_step):
                index_count = min(index_step, index_len - index_start)
                yield (
                    outer_start,
                    outer_count,
                    inner_start,
                    inner_count,
                    index_start,
                    index_count,
                )


def _full_block_tiles(outer, inner, index_len, max_block_size, max_grid):
    """Tile int64 fills without inactive lanes beyond the fill-space tail."""
    if min(outer, inner, index_len, max_block_size, max_grid) <= 0:
        raise ValueError("Index-fill tile dimensions must be positive")
    count = outer * inner * index_len
    block_size = min(max_block_size, count & -count)
    # Distribute the power-of-two block size among divisors of the three
    # axes. A tile in these units contains a whole number of launch blocks.
    inner_factor = min(block_size, inner & -inner)
    outer_factor = min(block_size // inner_factor, outer & -outer)
    index_factor = block_size // (inner_factor * outer_factor)
    for os, oc, ins, inc, ns, nc in _tiles(
        outer // outer_factor,
        inner // inner_factor,
        index_len // index_factor,
        max_grid,
    ):
        yield (
            block_size,
            (
                os * outer_factor,
                oc * outer_factor,
                ins * inner_factor,
                inc * inner_factor,
                ns * index_factor,
                nc * index_factor,
            ),
        )


def _make_contiguous_launch(module, original):
    kernel = module.index_fill_kernel
    block_size = module._BLOCK_SIZE
    max_rank = module._MAX_RANK
    int64_dtype = module.torch.int64

    @wraps(original)
    def launch(out, dim, index, value, value_is_tensor, fill_numel):
        if out.device.type != "gcu":
            return original(out, dim, index, value, value_is_tensor, fill_numel)
        inner = prod(out.shape[dim + 1 :])
        block_n = min(64, 1 << (inner - 1).bit_length())
        block_m = block_size if inner <= 4 else max(1, block_size // block_n)
        outer_index_len = fill_numel // inner
        grid_x = (outer_index_len + block_m - 1) // block_m
        grid_y = (inner + block_n - 1) // block_n
        if grid_x <= _MAX_GRID_X and grid_y <= _MAX_GRID_Y:
            return original(out, dim, index, value, value_is_tensor, fill_numel)

        dim_size = out.size(dim)
        outer = out.numel() // (dim_size * inner)
        # This helper receives contiguous output, including the contiguous
        # temporary used by FlagGems for strided inputs with rank > 8. Collapse
        # surrounding dimensions as a view so the existing generic kernel can
        # address any original rank using three shape/stride slots.
        view = out.view(outer, dim_size, inner)
        if out.dtype == int64_dtype:
            # Keep partial fill-space blocks out of the GCU int64 store path.
            # Values and upstream invalid-index masking remain unchanged.
            tiles = _full_block_tiles(
                outer, inner, index.numel(), block_size, _MAX_GRID_X
            )
        else:
            tiles = (
                (block_size, tile)
                for tile in _tiles(
                    outer, inner, index.numel(), _MAX_GRID_X * block_size
                )
            )
        for launch_block_size, (
            outer_start,
            outer_count,
            inner_start,
            inner_count,
            index_start,
            index_count,
        ) in tiles:
            part = view.narrow(0, outer_start, outer_count).narrow(
                2, inner_start, inner_count
            )
            index_part = index.narrow(0, index_start, index_count)
            count = outer_count * inner_count * index_count
            shapes = list(part.shape) + [1] * (max_rank - 3)
            strides = list(part.stride()) + [0] * (max_rank - 3)
            kernel[((count + launch_block_size - 1) // launch_block_size,)](
                part,
                index_part,
                value,
                count,
                index_count,
                dim_size,
                *shapes,
                *strides,
                DIM=1,
                RANK=3,
                VALUE_IS_TENSOR=value_is_tensor,
                BLOCK_SIZE=launch_block_size,
            )
        return None

    return launch


def _install_index_fill_gcu_patch(module) -> bool:
    original = getattr(module, "_index_fill_contiguous_launch", None)
    previous = getattr(module, _PATCH_STATE, None)
    if previous is not None:
        if original is not previous[0] or module.index_fill_kernel is not previous[1]:
            raise RuntimeError(
                "Enflame index_fill implementation changed after patching"
            )
        return False

    try:
        installed_version = version("flag-gems")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "FlagGems version is unavailable for Enflame adapter"
        ) from exc
    if installed_version != _SUPPORTED_VERSION:
        raise RuntimeError(
            f"Unsupported FlagGems version {installed_version} for Enflame adapter; "
            f"expected {_SUPPORTED_VERSION}"
        )

    kernel = getattr(module, "index_fill_kernel", None)
    jit = getattr(kernel, "jit_function", None)
    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Unsupported FlagGems index_fill launcher API for Enflame"
        ) from exc
    if (
        tuple(parameters) != _LAUNCH_PARAMETERS
        or any(
            parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters.values()
        )
        or getattr(module, "_MAX_RANK", None) != 8
        or getattr(module, "_BLOCK_SIZE", None) != 512
        or not callable(getattr(kernel, "__getitem__", None))
        or tuple(getattr(kernel, "arg_names", ())) != _KERNEL_PARAMETERS
        or tuple(getattr(jit, "arg_names", ())) != _KERNEL_PARAMETERS
        or tuple(getattr(jit, "constexprs", ())) != _KERNEL_CONSTEXPRS
    ):
        raise RuntimeError("Unsupported FlagGems index_fill kernel API for Enflame")

    wrapped = _make_contiguous_launch(module, original)
    module._index_fill_contiguous_launch = wrapped
    setattr(module, _PATCH_STATE, (wrapped, module.index_fill_kernel, original))
    return True


def apply_index_fill_gcu_patch() -> bool:
    """Adapt the official launcher in memory, preserving FlagGems registration."""
    module = importlib.import_module("flag_gems.ops.index_fill")
    return _install_index_fill_gcu_patch(module)

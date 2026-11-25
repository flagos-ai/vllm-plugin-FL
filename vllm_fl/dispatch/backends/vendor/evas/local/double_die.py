# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Double-die layout helpers, plugin-side copy.

Wrap the ``torch.ops.lpu`` double-die primitives (``clone_with_split_dim``,
``unzip_double_die_tensor``, ``zip_single_die_tensor``) and the descriptor
repair that keeps a view's shards consistent with its logical shape. Backend
ABI rather than ops, so the plugin must not reach into
``flag_gems.runtime.backend._evas.utils``; copied so a FlagGems-internal path
cannot be renamed out from under the plugin, carrying only the symbols the
plugin consumes, byte-identical to the original.

"""

from typing import Tuple

import torch

__all__ = [
    "is_double_die_enabled",
    "to_split_dim",
    "to_split_dim_logical",
    "unzip",
]


def is_double_die_enabled() -> bool:
    try:
        import torch_lpu

        return bool(torch_lpu.lpu.is_double_die_enabled())
    except Exception:
        return False


def to_split_dim(tensor: torch.Tensor, split_dim: int) -> torch.Tensor:
    if tensor.split_dim() != split_dim:
        tensor = torch.ops.lpu.clone_with_split_dim(tensor, split_dim=split_dim)
    return tensor


def unzip(
    tensor: torch.Tensor,
    split_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    split_size = tensor.shape[split_dim]
    if split_size == 1:
        raise RuntimeError(
            f"unzip does not support split_size=1 for split_dim={split_dim}"
        )
    return torch.ops.lpu.unzip_double_die_tensor(tensor)


def to_split_dim_logical(tensor: torch.Tensor, split_dim: int) -> torch.Tensor:
    """``to_split_dim`` that also repairs a stale double-die descriptor.

    A view that keeps its parent buffer's storage -- a row slice, a reshape of
    one -- still reports the parent's ``split_dim()``, so plain ``to_split_dim``
    returns it unchanged while ``unzip`` sizes its parts from the *descriptor*
    and hands back the parent buffer's two die halves instead of the view's.
    A caller that writes into the shards must be repaired here rather than in
    an ``unzip_logical``-style helper, because the clone that helper builds
    lives inside its own frame, so those writes land in a temporary that is
    dropped when the call returns.

    Cloning here instead makes the returned tensor differ from the argument, so
    the caller's own write-back branch (``if work is not view``) fires. On a
    descriptor that already matches the logical shape this is exactly
    ``to_split_dim``: the probe agrees and the tensor is returned unchanged.
    """
    tensor = to_split_dim(tensor, split_dim)
    if split_dim < 0 or tensor.dim() <= split_dim or tensor.shape[split_dim] <= 1:
        # ``unzip`` refuses a split extent of one, and a negative or
        # out-of-range dim is not a split this helper can reason about.
        return tensor
    try:
        shards = unzip(tensor, split_dim)
    except RuntimeError:
        # A layout this probe cannot part is one the caller cannot part either,
        # since it makes the same call with the same tensor; leave the tensor
        # alone rather than turning a probe into a new failure.
        return tensor
    expected0 = list(tensor.shape)
    expected1 = list(tensor.shape)
    expected0[split_dim] = (tensor.shape[split_dim] + 1) // 2
    expected1[split_dim] = tensor.shape[split_dim] // 2
    if (tuple(shards[0].shape) != tuple(expected0)
            or tuple(shards[1].shape) != tuple(expected1)):
        return torch.ops.lpu.clone_with_split_dim(tensor, split_dim=split_dim)
    return tensor

# SPDX-License-Identifier: Apache-2.0
"""Native stable integer sorting for the model-scoped top-p filter."""

import torch


def stable_sort_keys(keys, values, indices):
    if (
        not keys.is_cuda
        or keys.dtype != torch.int32
        or not keys.is_contiguous()
        or values.shape != keys.shape
        or values.dtype != keys.dtype
        or indices.shape != keys.shape
        or indices.dtype != torch.int64
        or values.device != keys.device
        or indices.device != keys.device
        or not values.is_contiguous()
        or not indices.is_contiguous()
    ):
        raise ValueError("Expected contiguous int32 keys/values and int64 indices")
    return torch.sort(
        keys, dim=-1, descending=False, stable=True, out=(values, indices)
    )

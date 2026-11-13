# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Framework-owned native QSA workspace prepared before graph capture."""

import torch
from flag_gems import arange

from .qsa_packing import _QSA_PACK_ROWS_CACHE


def allocate_qsa_workspace(m, device):
    return dict(
        table=torch.empty((m, 513), device=device, dtype=torch.int32),
        lengths=torch.empty(m, device=device, dtype=torch.int32),
        cu=torch.empty(m + 1, device=device, dtype=torch.int32),
        cuq=arange(m + 1, device=device, dtype=torch.int32),
        query=torch.empty((m, 3, 256), device=device, dtype=torch.bfloat16),
    )


def prepare_native_qsa_workspace(rows, device):
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Native QSA workspace must be prepared before graph capture")
    stream = (
        torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
    )
    owner = _QSA_PACK_ROWS_CACHE.get()
    key = ("native_workspace", rows, str(device), stream)
    if owner is not None and key in owner:
        return owner[key]
    workspace = allocate_qsa_workspace(rows, device)
    if owner is not None:
        owner[key] = workspace
    return workspace


def retain_native_graph_workspace(owner, workspace, query, score_columns):
    """Keep external capture buffers alive after per-step metadata is released.

    The model owns these references for its graph lifetime. Eager steps keep
    their scoped workspace; separate capture streams and score buckets do not
    alias mutable storage.
    """
    if workspace is None or not torch.cuda.is_current_stream_capturing():
        return workspace
    stream = torch.cuda.current_stream(query.device).cuda_stream
    key = (query.shape[0], query.device.index, query.dtype, stream, score_columns)
    return owner.setdefault(key, workspace)

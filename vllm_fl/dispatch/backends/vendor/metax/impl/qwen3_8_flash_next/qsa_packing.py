# SPDX-License-Identifier: Apache-2.0
"""Model shape/metadata policy over library QSA kernels."""

import torch


def packed_attention(
    q, k, v, indices, table, mapping, scale, out, gate, workspace_cache, rowmap=None
):
    eligible = (
        q.device.type == "cuda"
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[1:] == (3, 256)
        and k.ndim == 4
        and k.shape[2:] == (1, 256)
        and v.shape == k.shape
        and indices.shape == (q.shape[0], 2051)
        and k.shape[1] in (16, 400)
        and float(scale) == 0.0625
        and gate is not None
        and gate.dtype == torch.bfloat16
        and gate.shape == q.shape
        and q.stride(2) == 1
        and 0 < q.shape[0] <= 2048
    )
    if not eligible:
        return False
    b = q.shape[0]
    if b <= 64:
        mode = "fold4_skip"
        splits = 32 if b <= 16 else 16 if b <= 33 else 8
        warps, stages = 2, 1
        key = (mode, b, splits, q.device.index)
        candidate = workspace_cache.get(key)
        if candidate is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            from flaggems_vllm.runtime.backend._metax.ops.qsa_packed import Candidate

            candidate = Candidate(q, mode, splits, 16, warps, stages)
            workspace_cache[key] = candidate
    elif rowmap is not None:
        # The metadata owner certifies same-request groups and aligned columns.
        # Prefill/mixed are eager: temporary workspace is not retained per M.
        if torch.cuda.is_current_stream_capturing():
            return False
        from flaggems_vllm.runtime.backend._metax.ops.qsa_packed import Candidate

        splits = 8 if b <= 128 else 4
        warps, stages = (4, 1) if b <= 128 else (2, 3)
        candidate = Candidate(q, "pack4_skip", splits, 16, warps, stages, rows=rowmap)
    else:
        return False
    candidate(q, k, v, indices, table, mapping, out, gate)
    return True

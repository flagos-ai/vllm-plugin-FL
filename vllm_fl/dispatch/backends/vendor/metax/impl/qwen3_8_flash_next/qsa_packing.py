# SPDX-License-Identifier: Apache-2.0
"""Model shape/metadata policy over library QSA kernels."""

import json
from functools import lru_cache
from pathlib import Path

import torch

from .shape_limits import max_prefill_rows


@lru_cache(maxsize=1)
def _long_context_settings():
    path = (
        Path(__file__).resolve().parents[5]
        / "config/qwen3_8_flash_next_metax_tuning.json"
    )
    return json.loads(path.read_text())["qsa_long_context"]


def packed_attention(
    q,
    k,
    v,
    indices,
    table,
    mapping,
    scale,
    out,
    gate,
    workspace_cache,
    rowmap=None,
    *,
    score_columns=None,
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
        and 0 < q.shape[0] <= max_prefill_rows()
        and (
            q.shape[0] <= 2048 or (score_columns is not None and score_columns <= 17600)
        )
    )
    if not eligible:
        return False
    b = q.shape[0]
    if b <= 64:
        mode = "fold4_skip"
        splits = 32 if b <= 16 else 16 if b <= 33 else 8
        if b == 1 and score_columns is not None and 8192 < score_columns <= 17600:
            splits = _long_context_settings()["decode_single_splits"]
        if score_columns is not None:
            for preset in _long_context_settings().get("decode_shape_overrides", []):
                if (
                    b == preset["rows"]
                    and preset["min_score_columns"]
                    <= score_columns
                    <= preset["max_score_columns"]
                ):
                    splits = preset["splits"]
                    break
        warps, stages = 2, 1
        key = (mode, b, splits, q.device.index)
        candidate = workspace_cache.get(key)
        if candidate is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            from flaggems_vllm.runtime.backend._metax.ops.qsa_packed import Candidate

            candidate = Candidate(
                q, mode, splits, 16, warps, stages, collect_metadata=False
            )
            workspace_cache[key] = candidate
    elif rowmap is not None:
        # The metadata owner certifies same-request groups and aligned columns.
        # Prefill/mixed are eager: temporary workspace is not retained per M.
        if torch.cuda.is_current_stream_capturing():
            return False
        from flaggems_vllm.runtime.backend._metax.ops.qsa_packed import Candidate

        splits = 8 if b <= 128 else 4
        warps, stages = (4, 1) if b <= 128 else (2, 3)
        candidate = Candidate(
            q,
            "pack4_skip",
            splits,
            16,
            warps,
            stages,
            rows=rowmap,
            collect_metadata=False,
        )
    elif score_columns is not None and 512 < score_columns <= 17600:
        # Long-context queries can select different KV columns. Fold each
        # query independently; never reuse the short-context row packing.
        if torch.cuda.is_current_stream_capturing():
            return False
        from flaggems_vllm.runtime.backend._metax.ops.qsa_packed import Candidate

        preset = next(
            p for p in _long_context_settings()["prefill"] if b <= p["max_rows"]
        )
        candidate = Candidate(
            q,
            "fold4_skip",
            preset["splits"],
            preset["block_n"],
            preset["warps"],
            preset["stages"],
            collect_metadata=False,
        )
    else:
        return False
    candidate(q, k, v, indices, table, mapping, out, gate)
    return True

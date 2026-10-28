# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Default-off orchestration only; all numerical work lives in FlagGems-vllm."""

from vllm_fl.utils import use_flaggems_op


def maybe_mrope_direct_cache(rotary, positions, q_pe, k_pe):
    if not use_flaggems_op("mrope_strided_cache", default=False):
        return None
    if (
        positions.ndim != 2
        or q_pe.ndim != 3
        or k_pe.ndim != 3
        or getattr(rotary, "head_size", None) != 64
        or getattr(rotary, "rotary_dim", None) != 64
        or getattr(rotary, "is_neox_style", None) is not False
    ):
        return None
    sections = getattr(rotary, "mrope_section", None)
    interleaved = getattr(rotary, "mrope_interleaved", None)
    cache = getattr(rotary, "cos_sin_cache", None)
    if cache is None:
        return None
    from flaggems_vllm.ops.mrope_strided_cache import (
        can_mrope_strided_cache,
        mrope_strided_cache,
    )

    if not can_mrope_strided_cache(q_pe, k_pe, cache, positions, sections, interleaved):
        return None
    return mrope_strided_cache(q_pe, k_pe, cache, positions, sections, interleaved)

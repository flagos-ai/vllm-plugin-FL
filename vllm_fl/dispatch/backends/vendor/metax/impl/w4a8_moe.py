# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Existing vendor dispatch bridge to the public FGV packed W4A8 contract."""

import inspect


def is_available():
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "metax":
        return False
    try:
        import flaggems_vllm
        from mctlassEx import FusedMoeGEMM
    except (ImportError, OSError):
        return False
    function = getattr(flaggems_vllm, "fused_experts_w4a8", None)
    return (
        callable(function)
        and callable(getattr(FusedMoeGEMM, "get_kernel_m", None))
        and {"hidden_states", "w1", "w2", "topk_ids", "clamp_limit"}.issubset(
            inspect.signature(function).parameters
        )
    )


def w4a8_moe_experts(*args, **kwargs):
    from flaggems_vllm import fused_experts_w4a8

    return fused_experts_w4a8(*args, **kwargs)

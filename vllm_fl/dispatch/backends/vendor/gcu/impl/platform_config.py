# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU-specific adjustments to vLLM platform configuration.

Called from ``PlatformFL.apply_config_platform_defaults`` for the GCU/Enflame
vendor only, so no other backend sees these changes.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _default_capture_sizes() -> list[int]:
    """vLLM's default capture coverage (1..512), built explicitly."""
    return sorted(
        [1, 2, 4]
        + list(range(8, 256, 8))
        + list(range(256, 513, 16))
    )


def adjust_compilation_config_for_gcu(model_config, compilation_config) -> None:
    """Ensure CUDA-graph capture coverage for hybrid+MoE models on GCU.

    Root cause (verified 2026-11-21): with a small ``max_cudagraph_capture_size``
    (e.g. the gate CI list ``[1,2,4,8,16,32]``), any batch with more tokens than
    the cap takes the compiled fallback path; the GCU-compiled artifact of that
    configuration produces NaN in the GDN layers (hidden states all NaN ->
    argmax 0 -> token-0 ``!!!``).  Extending the coverage to vLLM's default
    (max 512) makes those batches run padded PIECEWISE graphs and is numerically
    correct, while a config compiled with ``cudagraph_mode=NONE`` is also
    correct.

    ``VLLM_FL_GCU_KEEP_CUDAGRAPH_CAP=1`` keeps the user's cap and instead
    disables graph replay (compiled eager path) to stay correct.
    """
    if compilation_config is None or model_config is None:
        return
    try:
        from vllm.config import CUDAGraphMode, CompilationMode
    except Exception:  # noqa: BLE001 - older/newer vLLM layout
        return

    if not (
        getattr(model_config, "is_hybrid", False)
        and getattr(model_config, "is_moe", False)
        and compilation_config.mode != CompilationMode.NONE
        and compilation_config.cudagraph_mode not in (None, CUDAGraphMode.NONE)
    ):
        return

    sizes = compilation_config.cudagraph_capture_sizes
    if not sizes or max(sizes) >= 512:
        return

    if os.environ.get("VLLM_FL_GCU_KEEP_CUDAGRAPH_CAP") == "1":
        logger.warning(
            "GCU hybrid+MoE model: capture sizes are capped at %s (<512) and "
            "VLLM_FL_GCU_KEEP_CUDAGRAPH_CAP=1 is set; disabling CUDA-graph "
            "replay to avoid the known GDN-NaN compiled fallback artifact.",
            max(sizes),
        )
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        return

    default_sizes = sorted(set(_default_capture_sizes() + [int(s) for s in sizes]))
    logger.warning(
        "GCU hybrid+MoE model: capture sizes cap %s (<512) triggers a known "
        "GCU compiled-fallback NaN bug in the GDN layers; extending capture "
        "coverage to the vLLM default (max %s).",
        max(sizes),
        default_sizes[-1],
    )
    compilation_config.cudagraph_capture_sizes = default_sizes
    compilation_config.max_cudagraph_capture_size = default_sizes[-1]

# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU launch configuration overrides for vLLM causal-conv kernels."""

from __future__ import annotations

import importlib
import logging

from vllm_fl.patches.triton_kernel import patch_kernel_launch_meta

logger = logging.getLogger(__name__)


CAUSAL_CONV1D_FWD_CONFIG = {
    "BLOCK_N": 2048,
    # vLLM already launches the forward kernel with num_stages=2.
    "num_stages": 2,
}
CAUSAL_CONV1D_UPDATE_CONFIG = {"BLOCK_N": 1024}


def apply_causal_conv1d_gcu_patch() -> None:
    """Apply the S60-tuned causal-conv launch configurations to vLLM."""
    causal_conv1d = importlib.import_module(
        "vllm.model_executor.layers.mamba.ops.causal_conv1d"
    )
    patch_kernel_launch_meta(
        causal_conv1d,
        "_causal_conv1d_fwd_kernel",
        CAUSAL_CONV1D_FWD_CONFIG,
    )
    patch_kernel_launch_meta(
        causal_conv1d,
        "_causal_conv1d_update_kernel",
        CAUSAL_CONV1D_UPDATE_CONFIG,
    )
    logger.info(
        "Patched causal-conv launch configs for GCU (prefill=%s, decode=%s)",
        CAUSAL_CONV1D_FWD_CONFIG,
        CAUSAL_CONV1D_UPDATE_CONFIG,
    )

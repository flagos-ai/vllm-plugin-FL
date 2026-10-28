# SPDX-License-Identifier: Apache-2.0
"""Explicit quantized-indexer policy shared by HYV4 framework adapters."""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def use_int8_indexer():
    mode = os.environ.get("HY4_INDEXER_DTYPE", "auto").lower()
    if mode not in ("auto", "int8", "fp8"):
        raise ValueError("HY4_INDEXER_DTYPE must be auto, int8 or fp8")
    # The supplied FP8 indexer uses DeepGEMM, requiring NVIDIA SM90+.
    # Other devices conservatively use the portable INT8 path.
    native = (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] >= 9
    )
    if mode == "fp8" and not native:
        raise ValueError("HYV4 FP8 indexer is unsupported on this device; use INT8")
    selected = mode == "int8" or (mode == "auto" and not native)
    logger.info(
        "HYV4 indexer dtype=%s (requested=%s)", "int8" if selected else "fp8", mode
    )
    return selected

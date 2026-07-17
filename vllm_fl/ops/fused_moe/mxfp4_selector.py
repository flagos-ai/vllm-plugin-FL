# Copyright (c) 2026 BAAI. All rights reserved.
"""Backend selection for MXFP4 MoE methods."""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

from vllm_fl.dispatch import BackendImplKind, get_default_manager

from vllm_fl.quantization.mxfp4.mxfp4_flaggems import (
    FlagGemsMxfp4MoEMethod,
    flaggems_mxfp4_compatible,
)
from vllm_fl.quantization.mxfp4.mxfp4_reference import ReferenceMxfp4MoEMethod


logger = init_logger(__name__)


def select_fixed_mxfp4_method(
    method: Mxfp4MoEMethod,
    moe: FusedMoEConfig,
) -> Mxfp4MoEMethod:
    """Fix the weight layout and experts from the dispatch policy."""
    manager = get_default_manager()
    try:
        candidates = manager.resolve_candidates("fused_marlin_moe")
    except RuntimeError as error:
        if "No implementation selected" not in str(error):
            raise
        logger.warning(
            "No dispatched MXFP4 backend available; using Torch reference"
        )
        return ReferenceMxfp4MoEMethod.from_method(method)

    for candidate in candidates:
        if candidate.kind == BackendImplKind.DEFAULT:
            if (
                candidate.impl_id == "default.flagos"
                and flaggems_mxfp4_compatible(moe)
            ):
                logger.info(
                    "MXFP4 backend fixed to FlagGems; preserving native uint8 weights"
                )
                return FlagGemsMxfp4MoEMethod.from_method(method)
            continue

        if candidate.kind == BackendImplKind.VENDOR:
            logger.info(
                "MXFP4 backend fixed to %s; using Marlin weight conversion",
                candidate.impl_id,
            )
            return method

        if candidate.kind == BackendImplKind.REFERENCE:
            logger.info(
                "MXFP4 backend fixed to native Torch reference; "
                "preserving native uint8 weights"
            )
            return ReferenceMxfp4MoEMethod.from_method(method)

    logger.warning("No dispatched MXFP4 backend available; using Torch reference")
    return ReferenceMxfp4MoEMethod.from_method(method)


__all__ = ["select_fixed_mxfp4_method"]

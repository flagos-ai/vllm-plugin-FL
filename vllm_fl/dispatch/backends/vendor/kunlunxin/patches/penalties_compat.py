# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Bind vLLM V2 penalties to the Kunlunxin operator provider."""

# Image pin 8ae5c22: https://github.com/flagos-ai/FlagGems-vllm/pull/1248.
# Native reference blockers: https://github.com/flagos-ai/FlagTree/issues/1700
# and https://github.com/flagos-ai/FlagTree/issues/1701.

import importlib
import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_penalties_provider"


def apply_penalties_compat() -> None:
    try:
        penalties = importlib.import_module("vllm.v1.worker.gpu.sample.penalties")
    except ImportError:
        return

    if getattr(penalties, _PATCH_FLAG, False):
        return

    from flaggems_vllm import v2_apply_penalties, v2_penalty_bincount

    penalties.bincount = v2_penalty_bincount
    penalties.apply_penalties = v2_apply_penalties
    setattr(penalties, _PATCH_FLAG, True)
    logger.info("Bound Kunlunxin Runner V2 penalties to FlagGems-vLLM")

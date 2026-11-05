# SPDX-License-Identifier: Apache-2.0
"""Shared Hy4 indexer dtype, layer-sharing and request-local score policy."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import regex as re
import torch

if TYPE_CHECKING:
    from .config import HYV4Config

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


_WEIGHT_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def compute_skip_topk_layers(config: HYV4Config) -> set[int]:
    """Return the backbone layers that reuse a previous layer's top-k indices.

    A "shared" indexer layer performs sparse attention with the indices computed
    by the closest preceding "full" indexer layer, so it does not build its own
    indexer and its checkpoint indexer weights must be skipped.

    Args:
        config: The model config.

    Returns:
        The set of layer indices that share another layer's top-k indices.

    Raises:
        ValueError: If ``indexer_types`` has the wrong length or an unknown
            entry.

    """
    indexer_types = config.indexer_types
    if len(indexer_types) != config.num_hidden_layers:
        raise ValueError(
            "indexer_types must contain one entry per hidden layer: "
            f"expected {config.num_hidden_layers}, got {len(indexer_types)}."
        )
    invalid_types = sorted(set(indexer_types) - {"full", "shared"})
    if invalid_types:
        raise ValueError(
            f"indexer_types only supports 'full' and 'shared', got {invalid_types}."
        )
    return {
        layer_idx
        for layer_idx, indexer_type in enumerate(indexer_types)
        if indexer_type == "shared"
    }


def is_skip_topk_indexer_weight(weight_name: str, skip_topk_layers: set[int]) -> bool:
    """Return whether an indexer weight belongs to a top-k sharing layer.

    Args:
        weight_name: Checkpoint weight name.
        skip_topk_layers: Result of `compute_skip_topk_layers`.

    Returns:
        True when the weight is an indexer weight of a layer that has no
        indexer module and therefore must be dropped.

    """
    if ".indexer." not in weight_name or not skip_topk_layers:
        return False
    match = _WEIGHT_LAYER_INDEX_RE.search(weight_name)
    return match is not None and int(match.group(1)) in skip_topk_layers


def _score_capacity(bound, width, topk):
    """Smallest qualified request-local capacity; no device-value reads.

    These are score columns, not a concatenated KV gather allocation. A
    selector still proves row/table/layout/step and uncaptured execution.
    FULL replay and unproven/decode metadata never enter this function.
    """
    if bound <= 0 or topk <= 0:
        return None
    for capacity in (16384, 32768):
        if max(bound, topk) <= capacity < width:
            return capacity
    return None

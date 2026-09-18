# SPDX-License-Identifier: Apache-2.0
"""Adapt vLLM scheduler metadata without changing its ownership."""

from .decode import FlashMLASchedMeta, flash_mla_with_kvcache


def decode_attention(*, tile_scheduler_metadata, **kwargs):
    original = tile_scheduler_metadata
    local = (
        original
        if isinstance(original, FlashMLASchedMeta)
        else FlashMLASchedMeta(
            have_initialized=original.have_initialized,
            config=original.config,
            tile_scheduler_metadata=original.tile_scheduler_metadata,
            num_splits=original.num_splits,
        )
    )
    result = flash_mla_with_kvcache(tile_scheduler_metadata=local, **kwargs)
    if local is not original:
        original.have_initialized = local.have_initialized
        original.config = local.config
        original.tile_scheduler_metadata = local.tile_scheduler_metadata
        original.num_splits = local.num_splits
    return result

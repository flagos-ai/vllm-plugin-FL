# Copyright (c) 2026 BAAI. All rights reserved.
"""Attach reusable long-prefill KDA metadata on Thead PPU."""

from __future__ import annotations

import logging
import os
from functools import wraps

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import async_tensor_h2d


logger = logging.getLogger(__name__)

_PATCH_MARKER = "_vllm_fl_thead_kda_prefill_metadata_patch"


def _common_attention_metadata(args, kwargs):
    common = kwargs.get("common_attn_metadata")
    if common is not None:
        return common
    if len(args) >= 2:
        return args[1]
    return None


def apply_thead_kda_prefill_metadata_patch(builder_cls=None) -> bool:
    """Precompute KDA shape routing and BT=16 indices on the CPU build path."""
    if current_platform.device_name != "thead":
        return False
    if builder_cls is None:
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadataBuilder,
        )

        builder_cls = GDNAttentionMetadataBuilder

    original = builder_cls.build
    if getattr(original, _PATCH_MARKER, False):
        return False

    @wraps(original)
    def build_with_kda_prefill_metadata(self, *args, **kwargs):
        metadata = original(self, *args, **kwargs)
        common = _common_attention_metadata(args, kwargs)
        if (
            common is None
            or metadata.num_prefills <= 0
            or metadata.num_spec_decodes != 0
        ):
            return metadata

        query_lens_cpu = (
            common.query_start_loc_cpu[1:] - common.query_start_loc_cpu[:-1]
        )
        num_sequences = int(metadata.num_decodes + metadata.num_prefills)
        non_spec_lens_cpu = query_lens_cpu[:num_sequences]
        non_spec_query_start_loc_cpu = torch.empty(
            num_sequences + 1, dtype=torch.int64
        )
        non_spec_query_start_loc_cpu[0] = 0
        torch.cumsum(
            non_spec_lens_cpu,
            dim=0,
            out=non_spec_query_start_loc_cpu[1:],
        )

        from vllm.model_executor.layers.fla.ops.index import (
            prepare_chunk_indices,
        )

        device = common.query_start_loc.device
        metadata.fl_kda_prefill_max_query_len = int(non_spec_lens_cpu.max().item())
        metadata.fl_kda_num_sequences = num_sequences
        metadata.fl_kda_total_tokens = int(non_spec_query_start_loc_cpu[-1].item())
        metadata.fl_kda_cu_seqlens_long = async_tensor_h2d(
            non_spec_query_start_loc_cpu,
            device=device,
        )
        metadata.fl_kda_chunk_indices_16 = async_tensor_h2d(
            prepare_chunk_indices(non_spec_query_start_loc_cpu, 16),
            device=device,
        )
        if os.getenv("VLLM_FL_LOG_KDA_PREFILL_SHAPES") == "1":
            logger.info(
                "FL_KDA_PREFILL_SHAPE requests=%d total_tokens=%d "
                "max_query_len=%d lengths=%s",
                num_sequences,
                metadata.fl_kda_total_tokens,
                metadata.fl_kda_prefill_max_query_len,
                non_spec_lens_cpu.tolist(),
            )
        return metadata

    setattr(build_with_kda_prefill_metadata, _PATCH_MARKER, True)
    build_with_kda_prefill_metadata._vllm_fl_original = original
    builder_cls.build = build_with_kda_prefill_metadata
    logger.info("Installed Thead long-prefill KDA metadata patch")
    return True


__all__ = ["apply_thead_kda_prefill_metadata_patch"]

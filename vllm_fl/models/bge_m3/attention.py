# Copyright (c) 2026 BAAI. All rights reserved.

from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType

from vllm_fl.dispatch import CachedOp
from vllm_fl.dispatch.backends.flaggems.impl.attention import (
    AttentionFLBackend,
    AttentionFLImpl,
    AttentionFLMetadataBuilder,
)

flash_attn_varlen_func = CachedOp("bge_m3_encoder_attention")


class BgeM3AttentionMetadataBuilder(AttentionFLMetadataBuilder):
    # Graph capture is outside the validated eager encoder contract.
    _cudagraph_support = AttentionCGSupport.NEVER


class BgeM3AttentionImpl(AttentionFLImpl):
    def _forward_encoder_attention(
        self, query, key, value, output, attn_metadata, layer
    ):
        """Run packed FP32 attention once for all independent documents."""
        if attn_metadata.causal or self.sliding_window != (-1, -1):
            raise ValueError("BGE-M3 requires full bidirectional attention")
        if self.alibi_slopes is not None or self.logits_soft_cap != 0:
            raise ValueError("BGE-M3 does not use ALiBi or attention soft-capping")
        flash_attn_varlen_func(
            query,
            key,
            value,
            out=output,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_query_len,
            causal=False,
            softmax_scale=self.scale,
        )
        return output


class BgeM3AttentionBackend(AttentionFLBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float32]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.ENCODER_ONLY

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 64

    @staticmethod
    def get_impl_cls():
        return BgeM3AttentionImpl

    @staticmethod
    def get_builder_cls():
        return BgeM3AttentionMetadataBuilder

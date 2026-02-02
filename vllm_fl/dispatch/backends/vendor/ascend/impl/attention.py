# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/v0.13.0rc1/vllm_ascend/attention/attention_v1.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd.

"""
Ascend NPU native attention backend for vllm-plugin-FL.

This module provides native Ascend NPU attention implementation using torch_npu
operators directly, without depending on vllm-ascend package.

Core operators used:
- torch_npu.npu_fused_infer_attention_score: For prefill/chunked-prefill
- torch_npu._npu_paged_attention: For decode
- torch_npu._npu_reshape_and_cache: For KV cache update

These are optimized operators for Huawei Ascend NPUs that provide better
performance than generic implementations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import AttentionCGSupport

from vllm_fl.dispatch.backends.vendor.ascend.impl.attention_mask import (
    AttentionMaskBuilder,
    get_attention_mask_builder,
)

logger = logging.getLogger(__name__)

# Check torch_npu availability and setup NPU compatibility
_TORCH_NPU_AVAILABLE = False
try:
    import torch_npu
    _TORCH_NPU_AVAILABLE = True

    # NPU compatibility: Replace torch.Event and torch.cuda.Stream with NPU versions
    # This is similar to vllm-ascend's _torch_cuda_wrapper approach
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.Event = torch.npu.Event
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        logger.info("NPU compatibility enabled: torch.Event -> torch.npu.Event")
except ImportError as e:
    raise ImportError(
        "torch_npu is required for Ascend attention backend. "
        "Please install torch_npu for NPU support."
    ) from e


def is_torch_npu_available() -> bool:
    """Check if torch_npu is available."""
    return _TORCH_NPU_AVAILABLE


# Ascend platform specific configurations
ASCEND_SAMPLED_TOKEN_IDS_DTYPE = torch.int32  # NPU uses int32, CUDA uses int64


class AscendAttentionState(Enum):
    """Attention state for Ascend backend."""
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """Metadata for Ascend attention."""
    # Basic properties
    attn_mask: Optional[torch.Tensor] = None
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Token counts
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # Sequence lengths
    seq_lens: torch.Tensor = None
    seq_lens_list: List[int] = None
    actual_seq_lengths_q: List[int] = None

    query_start_loc: torch.Tensor = None
    max_query_len: Optional[int] = None

    # KV Cache properties
    block_tables: torch.Tensor = None
    slot_mapping: torch.Tensor = None

    causal: bool = True
    model_runner_type: str = ""


class AscendAttentionMetadataBuilder:
    """Builder for Ascend attention metadata."""

    # ACL graph support - ALWAYS means full graph capture is supported
    aclgraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    reorder_batch_threshold: ClassVar[int] = 1

    @staticmethod
    def get_cudagraph_support(vllm_config, kv_cache_spec) -> AttentionCGSupport:
        """Get CUDAGraph support level for Ascend backend."""
        return AttentionCGSupport.ALWAYS

    # Class-level mask builder cache
    _mask_builder: ClassVar[Optional[AttentionMaskBuilder]] = None
    _mask_builder_device: ClassVar[Optional[torch.device]] = None

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendAttentionBackend.get_supported_block_size()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill

    def _get_mask_builder(self) -> AttentionMaskBuilder:
        """Get or create the attention mask builder (cached at class level)."""
        cls = AscendAttentionMetadataBuilder
        if cls._mask_builder is None or cls._mask_builder_device != self.device:
            cls._mask_builder = AttentionMaskBuilder(self.device)
            cls._mask_builder_device = self.device
        return cls._mask_builder

    def _make_attention_mask(
        self,
        attn_state: AscendAttentionState,
    ) -> Optional[torch.Tensor]:
        """
        Create attention mask based on attention state.

        Args:
            attn_state: Current attention state.

        Returns:
            Attention mask tensor, or None for decode-only.
        """
        # Decode-only doesn't need mask (uses paged attention)
        if attn_state == AscendAttentionState.DecodeOnly:
            return None

        mask_builder = self._get_mask_builder()

        # Pooling model uses general attention mask
        if self.model_config.runner_type == "pooling":
            return mask_builder.get_attn_mask(2048, torch.bool)

        # MLA attention
        if self.model_config.use_mla:
            # TODO: Add pcp_size check if needed
            return mask_builder.get_mla_mask(torch.float16)

        # Default: chunked prefill / split-fuse mask
        return mask_builder.get_splitfuse_attn_mask()

    def reorder_batch(self, input_batch, scheduler_output) -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        model: Optional[nn.Module] = None,
    ):
        """Build AscendMetadata from common attention metadata."""
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:num_reqs + 1]

        # Split decodes and prefills
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            self._split_decodes_and_prefills(common_attn_metadata)

        block_table = common_attn_metadata.block_table_tensor
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]

        # Determine attention state
        attn_state = self._determine_attn_state(
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens
        )

        # Create attention mask based on state
        attn_mask = self._make_attention_mask(attn_state)

        query_start_loc = query_start_loc_cpu.pin_memory().to(
            self.device, non_blocking=True)

        return AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist() if hasattr(seq_lens, 'tolist') else list(seq_lens),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=getattr(common_attn_metadata, 'causal', True),
            model_runner_type=self.model_config.runner_type,
        )

    def _determine_attn_state(
        self,
        num_decodes: int,
        num_prefills: int,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> AscendAttentionState:
        """Determine attention state based on batch composition."""
        if num_prefills == 0:
            return AscendAttentionState.DecodeOnly
        elif num_decodes == 0 and num_prefill_tokens > 0:
            # Pure prefill - check if cache hit or no cache
            # For simplicity, use ChunkedPrefill as default
            return AscendAttentionState.ChunkedPrefill
        else:
            # Mixed decode and prefill
            return AscendAttentionState.ChunkedPrefill

    def _split_decodes_and_prefills(self, common_attn_metadata):
        """Split batch into decode and prefill requests."""
        max_query_len = common_attn_metadata.max_query_len
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc_cpu

        if max_query_len <= self.decode_threshold:
            return num_reqs, 0, num_tokens, 0

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        is_prefill = query_lens > self.decode_threshold
        if not torch.any(is_prefill):
            return num_reqs, 0, num_tokens, 0

        first_prefill = is_prefill.int().argmax(dim=-1).item()
        num_decodes = first_prefill
        num_prefills = num_reqs - num_decodes
        num_decode_tokens = query_start_loc[first_prefill].item()
        num_prefill_tokens = num_tokens - num_decode_tokens
        return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        model: Optional[nn.Module] = None,
    ):
        """Build metadata for CUDA graph capture (ACL graph on Ascend)."""
        return self.build_for_graph_capture(
            common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
            model=model,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        model: Optional[nn.Module] = None,
    ):
        """Build metadata for graph capture."""
        if attn_state == AscendAttentionState.DecodeOnly:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently only support building dummy metadata for DecodeOnly state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        """
        Cascade attention is not supported for Ascend backend.

        Cascade attention is a CUDA-specific optimization that splits
        attention computation for shared prefixes. Ascend NPU uses
        different optimizations.
        """
        return False


class AscendAttentionBackend(AttentionBackend):
    """
    Ascend NPU native attention backend.

    Uses torch_npu operators directly for high-performance attention on
    Huawei Ascend NPUs.
    """
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_FL"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> Type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_block_size() -> list[int]:
        return [128]


class AscendAttentionBackendImpl(AttentionImpl):
    """
    Ascend attention implementation using native torch_npu operators.

    Core operators:
    - torch_npu.npu_fused_infer_attention_score: For prefill attention
    - torch_npu._npu_paged_attention: For decode attention
    - torch_npu._npu_reshape_and_cache: For KV cache updates
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        if not _TORCH_NPU_AVAILABLE:
            raise RuntimeError(
                "torch_npu is required for Ascend attention backend. "
                "Please install it with: pip install torch_npu"
            )

        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window

        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(
                alibi_slopes,
                dtype=torch.float32,
                device="npu"
            )
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None

    def _get_fia_params(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
    ):
        """Get parameters for fused_infer_attention."""
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape
            key = self.key_cache.view(num_block, block_size, -1)
            value = self.value_cache.view(num_block, block_size, -1)
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape
            key = self.key_cache.view(num_block, block_size, -1)
            value = self.value_cache.view(num_block, block_size, -1)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        else:
            # ChunkedPrefill
            num_block, block_size, _, _ = self.key_cache.shape
            key = self.key_cache.view(num_block, block_size, -1)
            value = self.value_cache.view(num_block, block_size, -1)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list

        return key, value, block_size, block_table, actual_seq_lengths_kv

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
    ):
        """Reshape and cache key/value tensors."""
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            # torch_npu requires int32 for slot_indices
            # TODO(yxa): block_table.py: CUDA uses int64, NPU uses int32.
            if slots.dtype != torch.int32:
                slots = slots.to(torch.int32)
            # Use torch_npu reshape_and_cache
            torch_npu._npu_reshape_and_cache(
                key=key[:attn_metadata.num_actual_tokens],
                value=value[:attn_metadata.num_actual_tokens],
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                slot_indices=slots[:attn_metadata.num_actual_tokens]
            )
        return key, value

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using fused_infer_attention_score."""
        key, value, block_size, block_table, actual_seq_lengths_kv = \
            self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            key = key[:num_tokens]
            value = value[:num_tokens]

        # Determine sparse_mode based on mask availability
        # sparse_mode=3 requires attn_mask; sparse_mode=0 does not
        # sparse_mode = 3 if attn_metadata.attn_mask is not None else 0

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )

        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using paged attention for decode."""
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for encoder-only attention."""
        assert attn_metadata is not None

        if attn_metadata.causal:
            # Use sparse_mode 3 in causal scenario
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                sparse_mode=3,
                atten_mask=attn_metadata.attn_mask,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
            )[0]
        else:
            # Use default sparse_mode 0 in normal scenario
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
            )[0]

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        """Forward implementation dispatching to appropriate attention method."""
        num_tokens = query.shape[0]

        # Use paged attention for decode-only state
        if (attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and self.sliding_window is None):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(
                query, key, value, attn_metadata, output)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Ascend attention.

        Args:
            layer: AttentionLayer containing scale factors
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention
            output: Pre-allocated output tensor
            output_scale: Optional output quantization scale
            output_block_scale: Optional output block quantization scale

        Returns:
            Output tensor of shape [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not yet supported "
                "for AscendAttentionBackendImpl"
            )

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0

        attn_type = self.attn_type
        if attn_type not in [AttentionType.DECODER, AttentionType.ENCODER_ONLY]:
            raise NotImplementedError(
                "Encoder/Decoder cross-attention is not implemented for "
                "AscendAttentionBackendImpl"
            )

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Reshape and cache KV
        key, value = self.reshape_and_cache(key, value, kv_cache, attn_metadata)

        # Handle pooling model branch (encoder attention)
        if attn_metadata.model_runner_type == "pooling":
            attn_output = self._forward_encoder_attention(
                query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output

        # Standard forward
        output = self.forward_impl(
            query, key, value, kv_cache, attn_metadata, output)
        return output


# MLA Backend placeholder - can be extended later
class AscendMLABackend:
    """
    Ascend MLA (Multi-head Latent Attention) backend placeholder.

    This is a minimal implementation. Full MLA support would require
    additional implementation based on the specific MLA algorithm.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Ascend MLA attention backend is not yet fully implemented. "
            "Please use standard attention backend by setting use_mla=False"
        )


__all__ = [
    "AscendAttentionBackend",
    "AscendAttentionBackendImpl",
    "AscendAttentionMetadataBuilder",
    "AscendMetadata",
    "AscendAttentionState",
    "AscendMLABackend",
    "is_torch_npu_available",
]

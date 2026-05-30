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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import AttentionCGSupport, CommonAttentionMetadata

from vllm_fl.dispatch.backends.vendor.ascend.impl.attention_mask import (
    AttentionMaskBuilder,
)

logger = logging.getLogger(__name__)

# Check torch_npu availability and setup NPU compatibility
_TORCH_NPU_AVAILABLE = False
try:
    import torch_npu
    _TORCH_NPU_AVAILABLE = True

    # NPU compatibility: Replace torch.Event and torch.cuda.Stream with NPU versions
    # This is similar to vllm-ascend's _torch_cuda_wrapper approach
    if hasattr(torch, "npu") and torch.npu.is_available():
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
    attn_state: AscendAttentionState = AscendAttentionState.PrefillNoCache

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


@dataclass
# class AscendCommonLongSequenceMetadata:
class AscendPrefillContextParallelMetadata:
    pcp_allgather_restore_idx: torch.Tensor = None

    num_actual_tokens_pcp_padded: int = 0

    num_computed_tokens_of_pcp_dcp: Optional[list[list[list[int]]]] = None

    q_head_idx_tensor: torch.Tensor = None

    q_tail_idx_tensor: torch.Tensor = None

    kv_with_q_head_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_head_mask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_mask_idx_tensor: torch.Tensor = None

    attn_mask_seqlens: torch.Tensor = None

    head_attn_nomask_seqlens: torch.Tensor = None

    tail_attn_nomask_seqlens: torch.Tensor = None

    q_full_idx: torch.Tensor = None

    # original query_lens before pcp split
    query_lens_pcp_full_cpu: torch.Tensor = None

    # original max_query_len before pcp split
    max_query_len_pcp_full: int = 0


@dataclass
class AscendCommonAttentionMetadata(CommonAttentionMetadata):
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both NPU and CPU versions.
    """

    seq_lens_cpu: torch.Tensor = None
    num_computed_tokens_cpu: torch.Tensor = None

    decode_token_per_req: int = 1
    """decode token number per request"""

    actual_seq_lengths_q: list[int] = field(default_factory=list)

    positions: torch.Tensor = None

    attn_state: Any = None

    graph_pad_size: int = -1

    # num_input_tokens refers to total number of tokens including
    # padding tokens. It is used to handle some padding operations.
    num_input_tokens: int = 0

    prefill_context_parallel_metadata: Optional[AscendPrefillContextParallelMetadata] = None

    # TODO: Remove it when vLLM no longer uses this function.
    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> "AscendCommonAttentionMetadata":
        # This only use to eagle now. It will be use to enforce_eager in future.
        return AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=self.seq_lens_cpu[:num_actual_reqs],
            num_computed_tokens_cpu=self.num_computed_tokens_cpu[:num_actual_reqs],
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            decode_token_per_req=self.decode_token_per_req,
            # NOTE: keep all tokens for block_table_tensor and slot_mapping otherwise
            # there will be error about shape mismatch during reshape and cache.
            # This is really strange since vLLM slices them as well
            block_table_tensor=self.block_table_tensor,
            slot_mapping=self.slot_mapping,
            causal=self.causal,
            actual_seq_lengths_q=self.actual_seq_lengths_q[:num_actual_tokens],
            positions=self.positions,
            attn_state=self.attn_state,
            graph_pad_size=-1,  # It should be -1 when not run in fullgraph mode.
            num_input_tokens=self.num_input_tokens,
            prefill_context_parallel_metadata=self.prefill_context_parallel_metadata,
            max_seq_len=self.max_seq_len,
        )


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
            return AscendAttentionState.PrefillNoCache
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
            # num_block, block_size, _, _ = self.key_cache.shape
            # key = self.key_cache.view(num_block, block_size, -1)
            # value = self.value_cache.view(num_block, block_size, -1)
            key = self.key_cache.view(-1, block_size, 256)
            value = self.value_cache.view(-1, block_size, 256)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        else:
            # ChunkedPrefill
            # num_block, block_size, _, _ = self.key_cache.shape
            # key = self.key_cache.view(num_block, block_size, -1)
            # value = self.value_cache.view(num_block, block_size, -1)
            key = self.key_cache.view(-1, block_size, 256)
            value = self.value_cache.view(-1, block_size, 256)
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
        if attn_metadata != AscendAttentionState.DecodeOnly:
            kv_cache = [i.contiguous() for i in kv_cache]
        if key is not None and value is not None:
            key = key.contiguous()
            value = value.contiguous()
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


@dataclass
class AscendMLAMetadata:

    # Basic properties
    attn_mask: Optional[torch.Tensor] = None
    attn_state: AscendAttentionState = AscendAttentionState.PrefillNoCache

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


class AscendMLAMetadataBuilder:

    aclgraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    reorder_batch_threshold: ClassVar[int] = 1

    @staticmethod
    def get_cudagraph_support(vllm_config, kv_cache_spec) -> AttentionCGSupport:
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
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            self.decode_threshold += \
                self.speculative_config.num_speculative_tokens

    def _get_mask_builder(self) -> AttentionMaskBuilder:
        cls = AscendMLAMetadataBuilder
        if cls._mask_builder is None or cls._mask_builder_device != self.device:
            cls._mask_builder = AttentionMaskBuilder(self.device)
            cls._mask_builder_device = self.device
        return cls._mask_builder

    def _make_attention_mask(
        self,
        attn_state: AscendAttentionState,
    ) -> Optional[torch.Tensor]:
        if attn_state == AscendAttentionState.DecodeOnly:
            return None
        mask_builder = self._get_mask_builder()
        return mask_builder.get_mla_mask(torch.float16)

    def reorder_batch(self, input_batch, scheduler_output) -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        model: Optional[nn.Module] = None,
    ) -> AscendMLAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[
            :num_reqs + 1
        ]

        # Split decodes / prefills
        num_decodes, num_prefills, num_decode_tokens, _ = \
            self._split_decodes_and_prefills(common_attn_metadata)

        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]

        # Determine attention state
        attn_state = self._determine_attn_state(
            num_decodes, num_prefills, num_decode_tokens
        )

        attn_mask = self._make_attention_mask(attn_state)

        query_start_loc = query_start_loc_cpu.pin_memory().to(
            self.device, non_blocking=True
        )

        return AscendMLAMetadata(
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist() if hasattr(seq_lens, 'tolist') else list(seq_lens),
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            query_start_loc=query_start_loc,
            max_query_len=common_attn_metadata.max_query_len,
            block_tables=block_table,
            slot_mapping=slot_mapping,
        )

    def _determine_attn_state(
        self,
        num_decodes: int,
        num_prefills: int,
        num_decode_tokens: int,
    ) -> AscendAttentionState:
        if num_prefills == 0:
            return AscendAttentionState.DecodeOnly
        elif num_decodes == 0:
            return AscendAttentionState.PrefillNoCache
        else:
            return AscendAttentionState.ChunkedPrefill

    def _split_decodes_and_prefills(self, common_attn_metadata):
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
        return (num_decodes, num_prefills, num_decode_tokens,
                num_prefill_tokens)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        model: Optional[nn.Module] = None,
    ):
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
        if attn_state == AscendAttentionState.DecodeOnly:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently only support building dummy metadata "
                "for DecodeOnly state"
            )
        attn_metadata.attn_state = attn_state
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


# ---------------------------------------------------------------------------
# MLA Backend
# ---------------------------------------------------------------------------

class AscendMLABackend(AttentionBackend):
    """
    Ascend NPU MLA attention backend.

    KV cache layout: (num_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
    Single compressed vector per token — only 1 KV "head".
    """
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MLA_FL"

    @staticmethod
    def get_impl_cls() -> Type["AscendMLAImpl"]:
        return AscendMLAImpl

    @staticmethod
    def get_builder_cls() -> Type[AscendMLAMetadataBuilder]:
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,   # 1 for MLA
        head_size: int,      # kv_lora_rank + qk_rope_head_dim for MLA
        cache_dtype_str: str = "auto",
    ) -> Tuple[int, ...]:
        # MLA: single compressed KV vector per token
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> Tuple[int, ...]:
        return (1, 0, 2, 3) if include_num_layers_dimension else (0, 1, 2)

    @staticmethod
    def get_supported_block_sizes() -> List[int]:
        return [64]

    @classmethod
    def get_supported_head_sizes(cls) -> List[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        block_size_in_bytes: int,
    ) -> None:
        for src, dst in zip(src_kv_cache, dst_kv_cache):
            dst.copy_(src)


class AscendMLAImpl:
    """Ascend NPU MLA attention implementation.

    MLA algorithm overview:
    - Prefill: project compressed kv_c through kv_b_proj to full K/V heads,
      then run standard multi-head attention (compute-friendly).
    - Decode: absorb W_UK into Q, then run single-KV-head attention against
      the compressed cache (data-movement-friendly).
    """

    can_return_lse_for_decode: bool = True
    supports_quant_query_input: bool = False

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
        # MLA-specific arguments
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        qk_head_dim: int = 192,
        v_head_dim: int = 128,
        kv_b_proj=None,     # ColumnParallelLinear – used in prefill
        indexer: Optional[object] = None,
        q_pad_num_heads: Optional[int] = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "KV sharing is not supported for Ascend MLA backend"
            )
        unsupported = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported):
            raise NotImplementedError(
                "AscendMLAImpl does not support alibi_slopes, "
                "sliding_window, or logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Only DECODER attention is supported for AscendMLAImpl"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA dimensions
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.indexer = indexer
        self.q_pad_num_heads = q_pad_num_heads

        # W_UV will be extracted in process_weights_after_loading
        self.W_UV = None  # [N, Lkv, V]

        # W_UK_T will be extracted in process_weights_after_loading
        # for decode Q absorption: q_nope @ W_UK_T -> q_latent
        self.W_UK_T = None  # [N, P, Lkv]

        # DCP / PCP (not yet supported, set defaults)
        self.dcp_world_size = 1
        self.dcp_rank = 0
        self.pcp_world_size = 1
        self.pcp_rank = 0
        self.total_cp_world_size = 1
        self.total_cp_rank = 0

        self.need_to_return_lse_for_decode = False

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """
        Extract W_UK_T and W_UV from kv_b_proj for decode.

        kv_b_proj weight shape: [N*(P+V), Lkv] (ColumnParallelLinear)
        Transpose to [Lkv, N*(P+V)], then split into W_UK [Lkv, N, P]
        and W_UV [Lkv, N, V].

        For decode:
          - W_UK_T: [N, P, Lkv] — absorb into Q: q_nope @ W_UK_T -> q_latent
          - W_UV:   [N, Lkv, V] — v_up projection: attn_out @ W_UV -> output
        """
        # Get dequantized weight
        weight = self.kv_b_proj.weight
        if hasattr(self.kv_b_proj, 'quant_method') and \
                self.kv_b_proj.quant_method is not None:
            try:
                weight = self.kv_b_proj.quant_method.get_weight(
                    self.kv_b_proj.weight, act_dtype
                )
            except Exception:
                logger.warning(
                    "AscendMLAImpl: failed to dequantize kv_b_proj weight, "
                    "using raw weight. This may produce incorrect results "
                    "with quantized models."
                )

        # weight: [out_features, in_features] = [N*(P+V), Lkv]
        # Transpose to [Lkv, N*(P+V)]
        weight_t = weight.t().to(act_dtype)
        weight_t = weight_t.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = weight_t.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # W_UK_T: [Lkv, N, P] -> permute to [N, P, Lkv] for bmm
        # decode: q_nope [N, B, P] @ W_UK_T [N, P, Lkv] -> q_latent [N, B, Lkv]
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()

        # W_UV: [Lkv, N, V] -> [N, Lkv, V] for bmm in v_up projection
        self.W_UV = W_UV.transpose(0, 1).contiguous()

    def reshape_and_cache_mla(
        self,
        kv_c: torch.Tensor,       # [num_tokens, kv_lora_rank]
        k_pe: torch.Tensor,       # [num_tokens, qk_rope_head_dim]
        kv_cache: torch.Tensor,   # [num_blocks, block_size, D]
        slot_mapping: torch.Tensor,  # [num_tokens]
    ) -> None:
        """
        Write compressed KV latent + rope key into MLA KV cache.
        """
        num_tokens = kv_c.shape[0]
        kv_c = kv_c.contiguous()
        k_pe = k_pe.contiguous()

        # Concatenate [kv_c | k_pe] along last dim  →  [T, D]
        kv_concat = torch.cat([kv_c, k_pe], dim=-1)  # [T, D]

        # Flatten cache to 2-D for slot-based indexing
        # Use view (not reshape) to guarantee this is a view of kv_cache,
        # so scatter_ modifies the original tensor in-place.
        cache_flat = kv_cache.view(-1, kv_cache.shape[-1])  # [num_slots, D]

        slots = slot_mapping[:num_tokens]

        # scatter_ (CANN-friendly, avoids MLIR crash on assignment indexing)
        # Build index: [T, D] — expand slot indices to match source shape
        idx = slots.unsqueeze(1).expand(-1, kv_concat.shape[-1])  # [T, D]
        cache_flat.scatter_(0, idx, kv_concat)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,                    # [num_tokens, N, D]
        key: torch.Tensor,                      # [num_tokens, kv_lora_rank] (kv_c_normed)
        value: torch.Tensor,                    # [num_tokens, qk_rope_head_dim] (k_pe)
        kv_cache: torch.Tensor,                 # [num_blocks, block_size, D]
        attn_metadata: AscendMLAMetadata,
        output: Optional[torch.Tensor] = None,  # [num_tokens, N * v_head_dim]
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for MLA attention.

        Args:
            layer: AttentionLayer containing scale factors.
            query: Query tensor [num_tokens, N, qk_head_dim].
                   For decode, Q is already absorbed (N, Lkv+R).
            key: Compressed KV latent (kv_c_normed) [num_tokens, kv_lora_rank].
            value: Decoupled rope key (k_pe) [num_tokens, qk_rope_head_dim].
            kv_cache: MLA KV cache [num_blocks, block_size, D].
            attn_metadata: Metadata for MLA attention.
            output: Pre-allocated output tensor.

        Returns:
            Output tensor.
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output.fill_(0)

        # Write compressed KV + k_pe to cache
        if kv_cache.numel() > 0 and key is not None and value is not None:
            k_pe_flat = value[:attn_metadata.num_actual_tokens].reshape(
                -1, self.qk_rope_head_dim)
            self.reshape_and_cache_mla(
                key[:attn_metadata.num_actual_tokens],
                k_pe_flat,
                kv_cache,
                attn_metadata.slot_mapping,
            )

        # Dispatch to prefill / decode
        output = self.forward_impl(
            query, key, value, kv_cache, attn_metadata, output)
        return output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMLAMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch to prefill or decode based on attn_state."""
        num_mqa_tokens = attn_metadata.num_decode_tokens
        num_mha_tokens = query.shape[0] - num_mqa_tokens

        # ---- Prefill (compute-friendly MHA path) ----
        if num_mha_tokens > 0:
            self._forward_prefill(
                q=query[num_mqa_tokens:],
                kv_c_normed=key[num_mqa_tokens:],
                k_pe=value[num_mqa_tokens:],
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output[num_mqa_tokens:],
            )

        # ---- Decode (data-movement-friendly MQA path) ----
        if num_mqa_tokens > 0:
            self._forward_decode(
                q=query[:num_mqa_tokens],
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output[:num_mqa_tokens],
            )

        return output

    def _forward_prefill(
        self,
        q: torch.Tensor,                   # [T_prefill, N, qk_head_dim]
        kv_c_normed: torch.Tensor,         # [T_prefill, kv_lora_rank]
        k_pe: torch.Tensor,                # [T_prefill, qk_rope_head_dim]
        kv_cache: torch.Tensor,            # [num_blocks, block_size, D]
        attn_metadata: AscendMLAMetadata,
        output: torch.Tensor,              # [T_prefill, N * v_head_dim]
    ) -> None:
        """
        MHA-style prefill forward.

        1. Project compressed kv_c through kv_b_proj to get k_nope & v.
        2. Concat k_nope with k_pe to form full K.
        3. Run varlen SDPA (causal) across all prefill requests.
        4. Write output.
        """
        num_prefill_tokens = q.shape[0]

        # ----- Step 1: project kv_c -> k_nope, v -----
        kv_proj = self.kv_b_proj(kv_c_normed)[0]   # [T, N*(P+V)]
        kv_proj = kv_proj.view(
            num_prefill_tokens,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, v = kv_proj.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # ----- Step 2: concat k_nope + k_pe -----
        # k_pe may be [T, 1, 1, R] or [T, 1, R] or [T, R], squeeze to [T, R]
        k_pe_2d = k_pe.reshape(num_prefill_tokens, self.qk_rope_head_dim)
        k_pe_3d = k_pe_2d.unsqueeze(1).expand(-1, self.num_heads, -1)  # [T, N, R]
        k = torch.cat([k_nope, k_pe_3d], dim=-1)  # [T, N, P+R]

        # ----- Step 3: varlen attention (causal), write directly into output -----
        cu_q = attn_metadata.query_start_loc
        max_q = attn_metadata.max_query_len

        self._native_flash_attn_varlen(
            q=q,
            k=k,
            v=v,
            output=output,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_q,
            max_seqlen_q=max_q,
            max_seqlen_k=max_q,
            softmax_scale=self.scale,
            causal=True,
        )

    def _forward_decode(
        self,
        q: torch.Tensor,                   # [T_decode, N, qk_head_dim]
        kv_cache: torch.Tensor,            # [num_blocks, block_size, D]
        attn_metadata: AscendMLAMetadata,
        output: torch.Tensor,              # [T_decode, N * v_head_dim]
    ) -> None:
        """
        MQA-style decode forward.

        In v0.13.0, Q arrives as [T_decode, N, qk_head_dim].
        We must:
          1. Split Q into q_nope [T, N, P] and q_pe [T, N, R]
          2. Absorb W_UK into q_nope: q_nope @ W_UK_T -> q_latent [N, T, Lkv]
          3. Concat q_latent and q_pe -> absorbed_q [T, N, Lkv+R]
          4. Run MQA attention: absorbed_q @ kv_cache^T -> softmax -> @ kv_c
          5. v_up projection: attn_out @ W_UV -> output

        The KV cache stores [kv_c || k_pe] per token with 1 KV head.
        """
        block_table = attn_metadata.block_tables
        seq_lens_list = attn_metadata.seq_lens_list
        num_decodes = attn_metadata.num_decodes

        num_blocks, block_size, cache_dim = kv_cache.shape
        device = q.device
        dtype = q.dtype

        # ---- Step 1: Split Q into q_nope and q_pe ----
        q_nope, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )  # q_nope: [T, N, P], q_pe: [T, N, R]

        # ---- Step 2: Absorb W_UK into q_nope ----
        # [N, T, P] @ [N, P, Lkv] -> [N, T, Lkv]
        q_nope_t = q_nope.transpose(0, 1)  # [N, T, P]
        q_latent = torch.bmm(q_nope_t, self.W_UK_T.to(dtype))  # [N, T, Lkv]
        q_latent = q_latent.transpose(0, 1)  # [T, N, Lkv]

        # ---- Step 3: Concat q_latent + q_pe -> absorbed_q ----
        absorbed_q = torch.cat([q_latent, q_pe], dim=-1)  # [T, N, Lkv+R]

        # Flatten cache for structured indexing
        cache_flat = kv_cache.reshape(-1, cache_dim)  # [num_slots, D]

        # Process each decode request
        q_offset = 0
        for i in range(num_decodes):
            s_i = seq_lens_list[i]

            # Number of query tokens for this request
            if i + 1 < num_decodes:
                q_len = 1
            else:
                q_len = absorbed_q.shape[0] - q_offset
                if num_decodes > 1:
                    q_len = 1

            q_i = absorbed_q[q_offset:q_offset + q_len]  # [q_len, N, Lkv+R]
            q_offset += q_len

            if s_i == 0:
                continue

            # Gather cached KV via block_table using index_select
            num_blks = (s_i + block_size - 1) // block_size
            blk_ids_i = block_table[i, :num_blks]  # [num_blks]

            blk_offsets = torch.arange(block_size, device=device, dtype=torch.int32)  # [bs]
            blk_starts = blk_ids_i * block_size  # [num_blks]
            indices = blk_starts.unsqueeze(1) + blk_offsets.unsqueeze(0)  # [num_blks, bs]
            indices = indices.reshape(-1)[:s_i]  # [s_i]

            kv_i = torch.index_select(cache_flat, 0, indices)  # [s_i, D]

            # V = kv_c (compressed latent), K = [kv_c, k_pe] = full cache row
            kv_c_i = kv_i[:, :self.kv_lora_rank]  # [s_i, Lkv]

            # q_i: [q_len, N, Lkv+R]
            # Batch over heads: [N, q_len, Lkv+R] @ [N, Lkv+R, s_i] -> [N, q_len, s_i]
            q_heads = q_i.permute(1, 0, 2)          # [N, q_len, Lkv+R]
            kv_t = kv_i.t().unsqueeze(0).expand(
                self.num_heads, -1, -1)              # [N, D, s_i]

            scores = torch.bmm(q_heads, kv_t.to(dtype)) * self.scale

            # Numerically stable softmax
            scores_max = scores.amax(dim=-1, keepdim=True)
            exp_scores = torch.exp(scores - scores_max)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)
            attn_w = exp_scores / sum_exp

            # Output: attn_w @ kv_c
            # [N, q_len, s_i] @ [N, s_i, Lkv] -> [N, q_len, Lkv]
            kv_c_for_v = kv_c_i.unsqueeze(0).expand(
                self.num_heads, -1, -1)               # [N, s_i, Lkv]

            o_i = torch.bmm(attn_w.to(dtype), kv_c_for_v.to(dtype))
            # o_i: [N, q_len, Lkv]

            # v_up projection: [N, q_len, Lkv] @ [N, Lkv, V] -> [N, q_len, V]
            o_i = torch.bmm(o_i, self.W_UV.to(dtype))  # [N, q_len, V]

            o_i = o_i.permute(1, 0, 2)  # [q_len, N, V]

            # Write to output buffer: [q_len, N*V]
            output[q_offset - q_len:q_offset].copy_(
                o_i.reshape(q_len, self.num_heads * self.v_head_dim)
            )

    def _native_flash_attn_varlen(
        self,
        q: torch.Tensor,       # [total_T, N, Dq]
        k: torch.Tensor,       # [total_T, N, Dk]
        v: torch.Tensor,       # [total_T, N, Dv]
        output: torch.Tensor,  # [total_T, N * v_head_dim]  — written in-place
        cu_seqlens_q: torch.Tensor,  # [B+1]
        cu_seqlens_k: torch.Tensor,  # [B+1]
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
    ) -> None:
        """Varlen attention using pure matmul + softmax.

        Splits the concatenated token tensors into per-sequence, computes
        attention manually, and writes results directly into output.
        """
        cu_q = cu_seqlens_q.cpu().numpy().tolist()
        cu_k = cu_seqlens_k.cpu().numpy().tolist()
        num_seqs = len(cu_q) - 1

        v_head_dim = v.shape[-1]

        for i in range(num_seqs):
            sq = cu_q[i + 1] - cu_q[i]
            sk = cu_k[i + 1] - cu_k[i]
            if sq == 0:
                continue

            qi = q[cu_q[i]:cu_q[i + 1]]   # [Sq, N, Dq]
            ki = k[cu_k[i]:cu_k[i + 1]]   # [Sk, N, Dk]
            vi = v[cu_k[i]:cu_k[i + 1]]   # [Sk, N, Dv]

            # Batch over heads: [N, Sq, Dq] @ [N, Dk, Sk] -> [N, Sq, Sk]
            qi = qi.transpose(0, 1)   # [N, Sq, Dq]
            ki = ki.transpose(0, 1)   # [N, Sk, Dk]
            vi = vi.transpose(0, 1)   # [N, Sk, Dv]

            scores = torch.bmm(qi, ki.transpose(-2, -1)) * softmax_scale

            # Causal mask: lower-triangle is allowed, upper is masked
            # Avoid torch.ones/torch.zeros on NPU — they can trigger
            # CANN TritonToStructured pointer-analysis crash.
            # Instead, build mask from arange comparison (structured op).
            if causal:
                row_idx = torch.arange(sq, device=q.device, dtype=torch.int32)
                col_idx = torch.arange(sk, device=q.device, dtype=torch.int32)
                # tril: row >= col
                causal_mask = row_idx.unsqueeze(1) >= col_idx.unsqueeze(0)  # [Sq, Sk]
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            # Numerically stable softmax
            scores_max = scores.amax(dim=-1, keepdim=True)
            exp_scores = torch.exp(scores - scores_max)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)
            attn_w = exp_scores / sum_exp

            # Output: [N, Sq, Sk] @ [N, Sk, Dv] -> [N, Sq, Dv]
            out_i = torch.bmm(attn_w.to(q.dtype), vi)

            # Write directly into output: [N, Sq, Dv] -> [Sq, N*V]
            out_slice = output[cu_q[i]:cu_q[i + 1]]  # [Sq, N*V]
            out_slice.copy_(out_i.transpose(0, 1).reshape(sq, -1))


__all__ = [
    "AscendAttentionBackend",
    "AscendAttentionBackendImpl",
    "AscendAttentionMetadataBuilder",
    "AscendMetadata",
    "AscendAttentionState",
    "AscendMLABackend",
    "AscendMLAMetadataBuilder",
    "AscendMLAMetadata",
    "AscendMLAImpl",
    "is_torch_npu_available",
]
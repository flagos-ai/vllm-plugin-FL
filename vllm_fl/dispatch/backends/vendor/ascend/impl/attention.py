# Copyright (c) 2026 BAAI. All rights reserved.
# Ordinary attention task capture/update and routing adapted from
# vllm-ascend v0.24.0rc1, vllm_ascend/attention/attention_v1.py.
# FL metadata integration follows the matching vLLM v0.24.0 contract.
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
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import accumulate
from typing import Any, ClassVar, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)

from vllm_fl.compilation.graph_params import RswaGraphBuffers
from vllm_fl.models.ocr_runtime import get_ocr_rswa_window, is_unlimited_ocr

from .attention_mask import AttentionMaskBuilder

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
_FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE = 512
_SWA_INT_MAX = 2147483647


def using_paged_attention(
    runtime_shape: int,
    vllm_config: VllmConfig,
    head_size: int | None = None,
) -> bool:
    """Match the vLLM-Ascend 0.24 regular-attention PA routing policy."""
    if vllm_config.speculative_config is not None:
        return False
    if head_size == _FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE:
        return True
    if (
        vllm_config.compilation_config.cudagraph_mode
        != CUDAGraphMode.FULL_DECODE_ONLY
    ):
        return False

    additional_config = vllm_config.additional_config or {}
    pa_shape_list = additional_config.get("pa_shape_list", [])
    if not isinstance(pa_shape_list, (list, tuple)) or any(
        not isinstance(size, int) or isinstance(size, bool)
        for size in pa_shape_list
    ):
        raise TypeError("additional_config.pa_shape_list must contain integers")
    return runtime_shape in pa_shape_list


@dataclass
class RswaAttentionGroup:
    request_start: int
    request_end: int
    token_start: int
    token_end: int
    actual_seq_lengths_q: list[int]
    attn_mask: torch.Tensor


def _build_grouped_rswa_masks(
    builder,
    prefix_lens,
    seq_lens,
    query_lens,
    num_decodes,
    num_decode_tokens,
    max_kv_len,
    rswa_window,
):
    """Group only ordered, unpadded mixed batches; never allocate the full mask."""
    lengths = query_lens.tolist()
    count = len(lengths)
    if not 0 < num_decodes < count:
        return None
    if (
        any(q != 1 for q in lengths[:num_decodes])
        or any(q <= 1 for q in lengths[num_decodes:])
        or sum(lengths[:num_decodes]) != num_decode_tokens
    ):
        return None
    if prefix_lens.numel() != count or seq_lens.numel() != count:
        return None
    groups = []
    token_start = 0
    for start, end in ((0, num_decodes), (num_decodes, count)):
        cumulative = list(accumulate(lengths[start:end]))
        token_end = token_start + cumulative[-1]
        mask = builder.get_rswa_attn_mask(
            prefix_lens=prefix_lens[start:end],
            seq_lens=seq_lens[start:end],
            query_lens=query_lens[start:end],
            max_kv_len=max_kv_len,
            max_query_len=max(lengths[start:end]),
            rswa_window=rswa_window,
        )
        groups.append(
            RswaAttentionGroup(
                start, end, token_start, token_end, cumulative, mask
            )
        )
        token_start = token_end
    return tuple(groups)


def _rswa_mask_request_lengths(
    prefix_lens: torch.Tensor,
    query_lens: torch.Tensor,
    num_actual_reqs: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exclude runner-declared padding from the mask, preserving FIA inputs."""
    if num_actual_reqs is None:
        return prefix_lens, query_lens
    num_reqs = query_lens.numel()
    if type(num_actual_reqs) is not int or not 0 <= num_actual_reqs <= num_reqs:
        raise ValueError(
            f"R-SWA actual request count must be in [0, {num_reqs}], "
            f"got {num_actual_reqs!r}"
        )
    if num_actual_reqs == num_reqs:
        return prefix_lens, query_lens
    # FULL decode padding has query extents but must not expose any keys.
    prefix_lens = prefix_lens.clone()
    query_lens = query_lens.clone()
    prefix_lens[num_actual_reqs:] = 0
    query_lens[num_actual_reqs:] = 0
    return prefix_lens, query_lens


class AscendAttentionState(Enum):
    """Attention state for Ascend backend."""
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


def classify_ascend_attention_state(
    num_computed_tokens: Sequence[int],
    num_scheduled_tokens: Sequence[int],
    num_valid_tokens: Sequence[int],
    enable_chunked_prefill: bool,
    speculative_method: str | None,
) -> AscendAttentionState:
    """Return rc1's stored/common-metadata phase for the current Ascend batch.

    Mirrors ModelRunner._build_attn_state, including the non-MTP speculative
    phase restoration before metadata is built. Initial one-token prompts
    precede the one-token decode check because they have no computed KV.
    """
    if all(tokens == 0 for tokens in num_computed_tokens):
        return AscendAttentionState.PrefillNoCache
    if all(tokens == 1 for tokens in num_scheduled_tokens):
        return (
            AscendAttentionState.SpecDecoding
            if speculative_method == "mtp"
            else AscendAttentionState.DecodeOnly
        )
    if all(tokens == 1 for tokens in num_valid_tokens):
        # rc1 returns SpecDecoding from its local classifier, but stores
        # ChunkedPrefill for every non-MTP speculative method before common
        # metadata is built. Return the stored/common-metadata value here.
        return (
            AscendAttentionState.SpecDecoding
            if speculative_method == "mtp"
            else AscendAttentionState.ChunkedPrefill
        )
    if enable_chunked_prefill:
        return AscendAttentionState.ChunkedPrefill
    return AscendAttentionState.PrefillCacheHit


def get_ascend_dummy_attention_state(
    create_mixed_batch: bool = False,
    speculative_method: str | None = None,
    use_mla: bool = False,
) -> AscendAttentionState:
    """Match rc1's explicit phase for capture, warmup, and idle dummies."""
    if create_mixed_batch:
        raise NotImplementedError(
            "create_mixed_batch is used for warmup deepgemm; Ascend "
            "does not support it"
        )
    if speculative_method == "mtp":
        return (
            AscendAttentionState.SpecDecoding
            if use_mla else AscendAttentionState.ChunkedPrefill
        )
    return AscendAttentionState.DecodeOnly


@dataclass(frozen=True)
class PagedAttentionGraphParam:
    layer_name: str
    query: Any
    key_cache: Any
    value_cache: Any
    num_kv_heads: int
    num_heads: int
    scale: float
    block_table: Any
    seq_lens: Any
    output: Any


@dataclass(frozen=True)
class FusedInferAttentionGraphParam:
    layer_name: str
    query: Any
    key: Any
    value: Any
    block_table: Any
    attn_mask: Any
    block_size: int
    actual_seq_lengths_kv: Any
    actual_seq_lengths_q: Any
    num_kv_heads: int
    num_heads: int
    scale: float
    output: Any
    softmax_lse: Any
    sparse_mode: int
    pre_tokens: int
    next_tokens: int
    rswa_slot_mapping: torch.Tensor | None = None


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

    rswa_window: int | None = None
    rswa_groups: tuple[RswaAttentionGroup, ...] | None = None

    causal: bool = True
    model_runner_type: str = ""




class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """Builder for Ascend attention metadata."""

    # ACL graph support - ALWAYS means full graph capture is supported
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self._mask_builder = AttentionMaskBuilder(
            device, ocr_mode=is_unlimited_ocr(self.model_config)
        )
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill

    def _get_mask_builder(self) -> AttentionMaskBuilder:
        return self._mask_builder

    def _make_attention_mask(
        self,
        attn_state: AscendAttentionState,
    ) -> Optional[torch.Tensor]:
        """
        Create attention mask based on attention state.

        Args:
            attn_state: Current attention state.

        Returns:
            Attention mask tensor for the selected regular/MLA path.
        """
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
        fast_build: bool = False,
    ):
        """Build AscendMetadata from common attention metadata."""
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:num_reqs + 1]

        # Split decodes and prefills
        num_decodes, num_prefills, num_decode_tokens, _ = \
            self._split_decodes_and_prefills(common_attn_metadata)

        block_table = common_attn_metadata.block_table_tensor
        # The private buffer is the authoritative CPU view in current vLLM;
        # async/spec paths may leave the public compatibility field unset.
        if getattr(common_attn_metadata, "_seq_lens_cpu", None) is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]

        # rc1's runner owns the phase: batch composition alone cannot tell
        # initial prefill from a later chunk or a prefix-cache hit.
        attn_state = common_attn_metadata.attn_state
        assert attn_state is not None, "Ascend metadata requires an attention state"

        # Create attention mask based on state
        attn_mask = self._make_attention_mask(attn_state)

        query_start_loc = query_start_loc_cpu.pin_memory().to(
            self.device, non_blocking=True)

        actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
        seq_lens_list = seq_lens.tolist()
        # FULL graph padding can add a dummy query request while the fixed
        # seq-length/block-table buffers remain at max_num_reqs. FIA derives
        # its batch size from the query prefix sums, so pad both KV metadata
        # views to the same request count. The dummy row never writes KV cache.
        num_reqs_fia = len(actual_seq_lengths_q)
        if len(seq_lens_list) < num_reqs_fia:
            padding_len = num_reqs_fia - len(seq_lens_list)
            seq_lens_list += [1] * padding_len
            seq_lens = torch.cat(
                [seq_lens, seq_lens.new_ones(padding_len)]
            )
        if block_table is not None and block_table.shape[0] < num_reqs_fia:
            block_table = torch.cat(
                [
                    block_table,
                    block_table.new_zeros(
                        (
                            num_reqs_fia - block_table.shape[0],
                            block_table.shape[1],
                        )
                    ),
                ],
                dim=0,
            )

        rswa_window = get_ocr_rswa_window(self.model_config)
        rswa_groups = None
        if rswa_window is not None:
            rswa_prefix_lens = getattr(common_attn_metadata, "rswa_prefix_lens", None)
            max_kv_len = self.model_config.max_model_len
            if type(rswa_window) is not int or not 0 < rswa_window <= max_kv_len:
                raise ValueError("R-SWA window must be within the model context length")
            if not getattr(common_attn_metadata, "causal", True):
                raise ValueError("R-SWA requires causal attention")
            if (
                not isinstance(rswa_prefix_lens, torch.Tensor)
                or rswa_prefix_lens.numel() < num_reqs
            ):
                raise RuntimeError("R-SWA requires a prompt length for each request")
            if getattr(self.vllm_config.quant_config, "enable_c8_quant", False):
                raise NotImplementedError(
                    "R-SWA does not support C8 KV-cache quantization"
                )

            rswa_prefix_lens = rswa_prefix_lens[:num_reqs_fia]
            if rswa_prefix_lens.shape[0] < num_reqs_fia:
                padding_len = num_reqs_fia - rswa_prefix_lens.shape[0]
                rswa_prefix_lens = torch.cat(
                    [rswa_prefix_lens, rswa_prefix_lens.new_zeros(padding_len)]
                )
            query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            num_actual_reqs = getattr(
                common_attn_metadata, "rswa_num_actual_reqs", None
            )
            rswa_prefix_lens, rswa_query_lens = _rswa_mask_request_lengths(
                rswa_prefix_lens, query_lens, num_actual_reqs
            )
            grouped = os.environ.get("VLLM_FL_RSWA_GROUPED_MASK", "1")
            if grouped not in ("0", "1"):
                raise ValueError("VLLM_FL_RSWA_GROUPED_MASK must be 0 or 1")
            if (
                grouped == "1"
                and attn_state == AscendAttentionState.ChunkedPrefill
                and self.vllm_config.speculative_config is None
                and getattr(common_attn_metadata, "graph_pad_size", -1) < 0
                and num_actual_reqs in (None, num_reqs_fia)
                and bool(actual_seq_lengths_q)
                and actual_seq_lengths_q[-1] == num_actual_tokens
            ):
                rswa_groups = _build_grouped_rswa_masks(
                    self._get_mask_builder(),
                    rswa_prefix_lens,
                    seq_lens[:num_reqs_fia],
                    rswa_query_lens,
                    num_decodes,
                    num_decode_tokens,
                    max_kv_len,
                    rswa_window,
                )
            if rswa_groups is not None:
                attn_mask = None
                if not getattr(self, "_rswa_grouped_logged", False):
                    logger.info(
                        "FL R-SWA grouped mask active: decode_requests=%s, prefill_requests=%s",
                        num_decodes,
                        num_reqs_fia - num_decodes,
                    )
                    self._rswa_grouped_logged = True
            else:
                attn_mask = self._get_mask_builder().get_rswa_attn_mask(
                    prefix_lens=rswa_prefix_lens,
                    seq_lens=seq_lens[:num_reqs_fia],
                    query_lens=rswa_query_lens,
                    max_kv_len=max_kv_len,
                    max_query_len=common_attn_metadata.max_query_len,
                    rswa_window=rswa_window,
                )

        return AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens_list,
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=actual_seq_lengths_q,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=getattr(common_attn_metadata, 'causal', True),
            model_runner_type=self.model_config.runner_type,
            rswa_window=rswa_window,
            rswa_groups=rswa_groups,
        )

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


@register_backend(AttentionBackendEnum.CUSTOM)
class AscendAttentionBackend(AttentionBackend):
    """
    Ascend NPU native attention backend.

    Uses torch_npu operators directly for high-performance attention on
    Huawei Ascend NPUs.
    """
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

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
    def get_supported_kernel_block_sizes() -> list[int]:
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
        self.sinks = kwargs.get("sinks")

    @staticmethod
    def _weak_ref_tensor(tensor: Any) -> Any:
        if tensor is None:
            return None
        from vllm_fl.compilation.graph import weak_ref_tensors

        return weak_ref_tensors(tensor)

    @staticmethod
    def stage_rswa_graph_inputs(forward_context: Any, num_tokens: int) -> None:
        """Validate live decode metadata and update the runner-owned inputs once."""
        params = forward_context.rswa_graph_params
        params.require_shape(num_tokens)
        buffers = params.rswa_buffers
        metadata = forward_context.attn_metadata
        if not isinstance(metadata, dict) or not metadata:
            raise RuntimeError("R-SWA graphs require one shared decode metadata group")
        live = next(iter(metadata.values()))
        if any(other is not live for other in metadata.values()):
            raise RuntimeError("R-SWA graphs require one shared decode metadata group")
        if (
            live.attn_state != AscendAttentionState.DecodeOnly
            or live.rswa_window is None
            or live.rswa_groups is not None
            or live.attn_mask is None
            or live.attn_mask.ndim != 4
            or live.block_tables is None
            or live.block_tables.ndim != 2
            or live.slot_mapping is None
        ):
            raise RuntimeError("R-SWA graphs require paged R-SWA decode metadata")

        kv_width = live.attn_mask.shape[-1]
        if (
            live.attn_mask.shape != (num_tokens, 1, 1, kv_width)
            or not 0 < kv_width <= buffers.attn_mask.shape[-1]
            or live.actual_seq_lengths_q != list(range(1, num_tokens + 1))
            or len(live.seq_lens_list) != num_tokens
            or any(not 0 <= length <= kv_width for length in live.seq_lens_list)
        ):
            raise RuntimeError(
                "R-SWA graph mask or sequence lengths exceed the decode bucket"
            )

        table_width = live.block_tables.shape[-1]
        (block_size,) = AscendAttentionBackend.get_supported_kernel_block_sizes()
        if (
            live.block_tables.shape != (num_tokens, table_width)
            or not cdiv(kv_width, block_size)
            <= table_width
            <= buffers.block_table.shape[-1]
        ):
            raise RuntimeError("R-SWA graph block table exceeds the decode bucket")
        slots = live.slot_mapping
        if (
            slots.shape != (num_tokens,)
            or slots.dtype not in (torch.int32, torch.int64)
            or slots.device != buffers.block_table.device
            or not slots.is_contiguous()
        ):
            raise RuntimeError("R-SWA graph requires the runner's fixed slot mapping")

        if getattr(forward_context, "capturing", False):
            # A reduced model length may use a narrower, contiguous view of
            # the original reservation. Replay uses the exact views captured here.
            mask = buffers.attn_mask.view(-1)[: live.attn_mask.numel()].view_as(
                live.attn_mask
            )
            table = buffers.block_table.view(-1)[: live.block_tables.numel()].view_as(
                live.block_tables
            )
            forward_context.rswa_graph_inputs = (mask, table)
        else:
            captured = params.attn_params[num_tokens]
            if (
                not captured
                or not len(captured)
                == len(params.events[num_tokens])
                == len(params.handles[num_tokens])
                or any(
                    not isinstance(param, FusedInferAttentionGraphParam)
                    or param.layer_name not in metadata
                    for param in captured
                )
            ):
                raise RuntimeError("R-SWA graph attention task state is incomplete")
            mask, table = captured[0].attn_mask, captured[0].block_table
            captured_slots = captured[0].rswa_slot_mapping
            if (
                captured_slots is None
                or slots.dtype != captured_slots.dtype
                or slots.data_ptr() != captured_slots.data_ptr()
            ):
                raise RuntimeError("R-SWA graph slot mapping changed address or dtype")

        # Validate both inputs before mutating either of them or enqueuing replay.
        for captured, current, label in (
            (mask, live.attn_mask, "mask"),
            (table, live.block_tables, "block table"),
        ):
            if (
                captured.shape != current.shape
                or captured.dtype != current.dtype
                or captured.device != current.device
            ):
                raise RuntimeError(f"R-SWA graph {label} layout changed")
        mask.copy_(live.attn_mask)
        table.copy_(live.block_tables)

    @staticmethod
    def update_graph_params(
        update_stream: Any,
        forward_context: Any,
        num_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        """Rebind live regular-attention metadata after FULL replay enqueue."""
        from vllm_fl.compilation.graph_params import get_graph_params

        rswa_params = getattr(forward_context, "rswa_graph_params", None)
        graph_params = rswa_params if rswa_params is not None else get_graph_params()
        graph_params.require_shape(num_tokens)
        params = graph_params.attn_params[num_tokens]
        handles = graph_params.handles[num_tokens]
        events = graph_params.events[num_tokens]
        if not params:
            raise RuntimeError(
                f"No Ascend attention task was captured for graph size {num_tokens}"
            )
        if not (len(params) == len(handles) == len(events)):
            raise RuntimeError(
                "Incomplete Ascend attention graph state: "
                f"params={len(params)}, handles={len(handles)}, events={len(events)}"
            )

        metadata = forward_context.attn_metadata
        if any(
            not isinstance(
                param, (PagedAttentionGraphParam, FusedInferAttentionGraphParam)
            )
            for param in params
        ):
            raise RuntimeError(
                f"Unknown Ascend attention task parameters (graph size {num_tokens})"
            )

        with torch.npu.stream(update_stream):
            for param, handle, event in zip(params, handles, events):
                if param.layer_name not in metadata:
                    raise RuntimeError(
                        "Live Ascend attention metadata is missing captured layer "
                        f"{param.layer_name!r}"
                    )
                live = metadata[param.layer_name]
                if live.attn_state != AscendAttentionState.DecodeOnly:
                    raise NotImplementedError(
                        "Ascend FULL graph task update currently supports only "
                        "regular DecodeOnly attention"
                    )
                if isinstance(param, PagedAttentionGraphParam):
                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=param.query,
                        key_cache=param.key_cache,
                        value_cache=param.value_cache,
                        num_kv_heads=param.num_kv_heads,
                        num_heads=param.num_heads,
                        scale_value=param.scale,
                        block_table=live.block_tables,
                        context_lens=live.seq_lens,
                        out=param.output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=param.query,
                        key_cache=param.key_cache,
                        value_cache=param.value_cache,
                        num_kv_heads=param.num_kv_heads,
                        num_heads=param.num_heads,
                        scale_value=param.scale,
                        block_table=live.block_tables,
                        context_lens=live.seq_lens,
                        out=param.output,
                        workspace=workspace,
                    )
                else:
                    workspace = graph_params.workspaces[num_tokens]
                    if workspace is None:
                        raise RuntimeError(
                            f"Missing FIA workspace for graph size {num_tokens}"
                        )
                    # Match rc1's model-level SWA guard, including global
                    # attention layers in a model with sliding_window metadata.
                    # Rebinding their captured table can corrupt FULL replay.
                    block_table = param.block_table
                    if rswa_params is None and not hasattr(
                        vllm_config.model_config.hf_text_config, "sliding_window"
                    ):
                        block_table = live.block_tables
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=param.query,
                        key=param.key,
                        value=param.value,
                        atten_mask=param.attn_mask,
                        block_table=block_table,
                        input_layout="TND",
                        block_size=param.block_size,
                        actual_seq_lengths=live.actual_seq_lengths_q,
                        actual_seq_lengths_kv=live.seq_lens_list,
                        num_key_value_heads=param.num_kv_heads,
                        num_heads=param.num_heads,
                        scale=param.scale,
                        sparse_mode=param.sparse_mode,
                        pre_tokens=param.pre_tokens,
                        next_tokens=param.next_tokens,
                        workspace=workspace,
                        out=[param.output, param.softmax_lse],
                    )
                torch.npu.graph_task_update_end(update_stream)
                event.record(update_stream)

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
        layer: AttentionLayer | None = None,
    ) -> torch.Tensor:
        """Forward pass using fused_infer_attention_score."""
        forward_context = get_forward_context()
        if getattr(forward_context, "capturing", False):
            return self._capture_fused_infer_attention(
                query, key, value, attn_metadata, output, layer
            )

        key, value, block_size, block_table, actual_seq_lengths_kv = (
            self._get_fia_params(key, value, attn_metadata)
        )

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            key = key[:num_tokens]
            value = value[:num_tokens]

        mask_args = {}
        sparse_mode = 0
        if attn_metadata.rswa_window is not None:
            mask_args.update(
                atten_mask=attn_metadata.attn_mask,
                pre_tokens=_SWA_INT_MAX,
                next_tokens=_SWA_INT_MAX,
            )
        elif attn_metadata.causal:
            sparse_mode = 4 if self.sliding_window is not None else 3
            mask_args["atten_mask"] = attn_metadata.attn_mask
            if self.sliding_window is not None:
                mask_args.update(pre_tokens=self.sliding_window, next_tokens=0)
        fia_args = dict(
            key=key,
            value=value,
            input_layout="TND",
            block_size=block_size,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            **mask_args,
        )
        if attn_metadata.rswa_groups is not None:
            if (
                block_table is None
                or attn_metadata.attn_state != AscendAttentionState.ChunkedPrefill
            ):
                raise RuntimeError("Grouped R-SWA requires cache-backed ChunkedPrefill")
            for group in attn_metadata.rswa_groups:
                requests = slice(group.request_start, group.request_end)
                tokens = slice(group.token_start, group.token_end)
                fia_args["atten_mask"] = group.attn_mask
                result, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query[tokens],
                    block_table=block_table[requests].contiguous(),
                    actual_seq_lengths=group.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv[requests],
                    **fia_args,
                )
                output[tokens] = result.view(-1, self.num_heads, self.head_size)
        else:
            result, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                block_table=block_table,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                **fia_args,
            )
            output[:num_tokens] = result.view(
                num_tokens, self.num_heads, self.head_size
            )
        return output

    def _require_regular_graph_capture(
        self,
        attn_metadata: AscendMetadata,
        layer: AttentionLayer | None,
    ) -> str:
        if self.vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Ascend FULL graph task capture does not yet support "
                "speculative attention in FL"
            )
        if self.vllm_config.quant_config is not None:
            raise NotImplementedError(
                "Ascend FULL graph task capture does not yet support "
                "quantized attention in FL"
            )
        if self.sinks is not None:
            raise NotImplementedError(
                "Ascend FULL graph task capture currently supports regular "
                "attention without sinks"
            )
        if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            raise NotImplementedError(
                "Ascend FULL graph task capture currently supports only "
                "regular DecodeOnly attention"
            )
        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str) or not layer_name:
            raise RuntimeError(
                "Ascend FULL graph task capture requires AttentionLayer.layer_name"
            )
        return layer_name

    def _capture_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer | None,
    ) -> torch.Tensor:
        """Capture current-rc1 regular FIA as an updatable NPU task."""
        from vllm_fl.compilation.graph_params import get_graph_params

        layer_name = self._require_regular_graph_capture(attn_metadata, layer)
        key, value, block_size, block_table, actual_seq_lengths_kv = (
            self._get_fia_params(key, value, attn_metadata)
        )
        num_tokens = query.shape[0]
        attn_mask = attn_metadata.attn_mask
        is_rswa = attn_metadata.rswa_window is not None
        if is_rswa:
            context = get_forward_context()
            graph_params = getattr(context, "rswa_graph_params", None)
            if graph_params is None or not hasattr(context, "rswa_graph_inputs"):
                raise RuntimeError(
                    "R-SWA graph inputs and workspace must be reserved before capture"
                )
            attn_mask, block_table = context.rswa_graph_inputs
            sparse_mode = 0
            pre_tokens = next_tokens = _SWA_INT_MAX
        else:
            graph_params = get_graph_params()
            sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0
            pre_tokens = self.sliding_window or _SWA_INT_MAX
            next_tokens = 0 if self.sliding_window else _SWA_INT_MAX
        graph_params.require_shape(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        workspace = graph_params.workspaces[num_tokens]
        if is_rswa and workspace is None:
            raise RuntimeError("R-SWA FIA workspace was not prepared before capture")
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.scale,
            )
            graph_params.workspaces[num_tokens] = workspace

        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            FusedInferAttentionGraphParam(
                layer_name=layer_name,
                query=self._weak_ref_tensor(query),
                key=self._weak_ref_tensor(key),
                value=self._weak_ref_tensor(value),
                block_table=block_table
                if is_rswa
                else self._weak_ref_tensor(block_table),
                attn_mask=attn_mask if is_rswa else self._weak_ref_tensor(attn_mask),
                block_size=block_size,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                output=self._weak_ref_tensor(output),
                softmax_lse=self._weak_ref_tensor(softmax_lse),
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                rswa_slot_mapping=attn_metadata.slot_mapping if is_rswa else None,
            )
        )
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            workspace=workspace,
            out=[output, softmax_lse],
        )
        graph_params.handles[num_tokens].append(torch.npu.graph_task_group_end(stream))
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        layer: AttentionLayer | None = None,
    ) -> torch.Tensor:
        """Forward pass using paged attention for decode."""
        forward_context = get_forward_context()
        if getattr(forward_context, "capturing", False):
            return self._capture_paged_attention(
                query, attn_metadata, output, layer
            )
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

    def _capture_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer | None,
    ) -> torch.Tensor:
        """Capture an explicitly selected regular PA task for later rebinding."""
        from vllm_fl.compilation.graph_params import get_graph_params

        layer_name = self._require_regular_graph_capture(attn_metadata, layer)
        num_tokens = query.shape[0]
        graph_params = get_graph_params()
        graph_params.require_shape(num_tokens)
        # PA sizes its own workspace. Reserve the shared slot for FIA, which
        # needs its maximum workspace retained for task updates after capture.
        workspace = torch_npu._npu_paged_attention_get_workspace(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )

        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            PagedAttentionGraphParam(
                layer_name=layer_name,
                query=self._weak_ref_tensor(query),
                key_cache=self._weak_ref_tensor(self.key_cache),
                value_cache=self._weak_ref_tensor(self.value_cache),
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                block_table=self._weak_ref_tensor(attn_metadata.block_tables),
                seq_lens=attn_metadata.seq_lens,
                output=self._weak_ref_tensor(output),
            )
        )
        torch.npu.graph_task_group_begin(stream)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
            workspace=workspace,
        )
        graph_params.handles[num_tokens].append(
            torch.npu.graph_task_group_end(stream)
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
        layer: AttentionLayer | None = None,
    ):
        """Forward implementation dispatching to appropriate attention method."""
        num_tokens = query.shape[0]

        # Use paged attention for decode-only state
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is None
            and attn_metadata.rswa_window is None
            and using_paged_attention(num_tokens, self.vllm_config, self.head_size)
        ):
            output = self.forward_paged_attention(
                query, attn_metadata, output, layer
            )
        else:
            output = self.forward_fused_infer_attention(
                query, key, value, attn_metadata, output, layer)

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
        if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
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
            query, key, value, kv_cache, attn_metadata, output, layer)
        return output


def reserve_rswa_graph_buffers(
    vllm_config: VllmConfig, device: torch.device
) -> RswaGraphBuffers | None:
    """Use the installed FIA operator's maximum-workspace query before capture."""
    config = vllm_config
    compilation = config.compilation_config
    if (
        compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
        or config.speculative_config is not None
        or config.quant_config is not None
        or config.parallel_config.decode_context_parallel_size != 1
        or config.parallel_config.prefill_context_parallel_size != 1
    ):
        raise NotImplementedError(
            "R-SWA FULL graphs require unquantized single-token decode without context parallelism"
        )

    capture_sizes = frozenset(
        size
        for size in (compilation.cudagraph_capture_sizes or ())
        if 0 < size <= config.scheduler_config.max_num_seqs
    )
    if not capture_sizes:
        return None

    workspace_api = getattr(
        torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", None
    )
    fia = getattr(torch_npu, "npu_fused_infer_attention_score", None)
    graph_apis = (
        "ExternalEvent",
        "graph_task_group_begin",
        "graph_task_group_end",
        "graph_task_update_begin",
        "graph_task_update_end",
    )
    if (
        not callable(workspace_api)
        or not callable(getattr(fia, "out", None))
        or not all(callable(getattr(torch.npu, name, None)) for name in graph_apis)
    ):
        raise RuntimeError(
            "R-SWA FULL decode requires torch_npu FIA workspace and out APIs"
        )

    layouts = set()
    for layer in compilation.static_forward_context.values():
        impl = getattr(layer, "impl", None)
        if (
            isinstance(impl, AscendAttentionBackendImpl)
            and impl.attn_type == AttentionType.DECODER
        ):
            layouts.add((impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale))
    if not layouts:
        raise RuntimeError("R-SWA has no native Ascend decoder attention layers")

    max_tokens = max(capture_sizes)
    max_kv = config.model_config.max_model_len
    (block_size,) = AscendAttentionBackend.get_supported_kernel_block_sizes()
    max_blocks = cdiv(max_kv, block_size)
    dtype = config.model_config.dtype
    mask = torch.empty((max_tokens, 1, 1, max_kv), dtype=torch.int8, device=device)
    table = torch.empty((max_tokens, max_blocks), dtype=torch.int32, device=device)
    workspace = None
    for num_heads, num_kv_heads, head_size, scale in layouts:
        # The vendor query examines tensor descriptors and the host sequence
        # lengths. A single context's pages suffice, with no live KV allocation.
        query = torch.empty(
            (max_tokens, num_heads, head_size), dtype=dtype, device=device
        )
        pages = torch.empty(
            (max_blocks, block_size, num_kv_heads * head_size),
            dtype=dtype,
            device=device,
        )
        for size in sorted(capture_sizes, reverse=True):
            sized = workspace_api(
                query=query[:size],
                key=pages,
                value=pages,
                atten_mask=mask[:size],
                block_table=table[:size],
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=list(range(1, size + 1)),
                actual_seq_lengths_kv=[max_kv] * size,
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                sparse_mode=0,
                pre_tokens=_SWA_INT_MAX,
                next_tokens=_SWA_INT_MAX,
                scale=scale,
            )
            if workspace is None or sized.nbytes > workspace.nbytes:
                workspace = sized
            del sized

    # Layers and buckets execute serially on this runner's replay stream.
    # Keep one sufficiently large tensor alive for every captured task.
    return RswaGraphBuffers(capture_sizes, mask, table, workspace)


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

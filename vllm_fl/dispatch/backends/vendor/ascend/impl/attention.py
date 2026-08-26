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
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_fl.dispatch.backends.vendor.ascend.impl.attention_mask import (
    AttentionMaskBuilder,
)
from vllm_fl.compilation.graph import (
    get_ascend_graph_params,
    is_ascend_graph_capturing,
    record_ascend_graph_task,
    update_ascend_graph_params_workspace,
    weak_ref_tensors,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.attention_utils import (
    using_paged_attention,
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
SWA_INT_MAX = 2147483647


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
    positions_cpu: torch.Tensor = None

    attn_state: Any = None

    graph_pad_size: int = -1

    # num_input_tokens refers to total number of tokens including
    # padding tokens. It is used to handle some padding operations.
    num_input_tokens: int = 0

    prefill_context_parallel_metadata: Optional[AscendPrefillContextParallelMetadata] = None
    kvcomp_metadata: Any = None

    # TODO: Remove it when vLLM no longer uses this function.
    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> "AscendCommonAttentionMetadata":
        # This only use to eagle now. It will be use to enforce_eager in future.
        def _slice_reqs(value):
            return value[:num_actual_reqs] if value is not None else None

        return AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=_slice_reqs(self.seq_lens_cpu),
            num_computed_tokens_cpu=_slice_reqs(self.num_computed_tokens_cpu),
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
            positions_cpu=self.positions_cpu,
            attn_state=self.attn_state,
            graph_pad_size=-1,  # It should be -1 when not run in fullgraph mode.
            num_input_tokens=self.num_input_tokens,
            prefill_context_parallel_metadata=self.prefill_context_parallel_metadata,
            seq_lens_cpu_upper_bound=_slice_reqs(self.seq_lens_cpu_upper_bound),
            max_seq_len=self.max_seq_len,
            _seq_lens_cpu=_slice_reqs(self._seq_lens_cpu),
            _num_computed_tokens_cpu=_slice_reqs(
                self._num_computed_tokens_cpu
            ),
            dcp_local_seq_lens=_slice_reqs(self.dcp_local_seq_lens),
            dcp_local_seq_lens_cpu=_slice_reqs(
                self.dcp_local_seq_lens_cpu
            ),
            is_prefilling=_slice_reqs(self.is_prefilling),
            encoder_seq_lens=_slice_reqs(self.encoder_seq_lens),
            encoder_seq_lens_cpu=_slice_reqs(self.encoder_seq_lens_cpu),
            logits_indices_padded=self.logits_indices_padded,
            num_logits_indices=self.num_logits_indices,
        )


class AscendAttentionMetadataBuilder:
    """Builder for Ascend attention metadata."""

    # ACL graph support - ALWAYS means full graph capture is supported
    aclgraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    reorder_batch_threshold: ClassVar[int] = 1
    supports_update_block_table: bool = False

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
            Attention mask tensor selected for the model runner.
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
        # ``seq_lens_cpu`` is intentionally None under async scheduling. The
        # target Ascend path keeps the optimistic pinned buffer available via
        # ``_seq_lens_cpu`` so attention never performs a device-to-host sync.
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        if seq_lens_cpu is None:
            raise RuntimeError(
                "Ascend attention requires seq_lens_cpu or _seq_lens_cpu"
            )
        seq_lens = seq_lens_cpu[:num_reqs]
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
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> Type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Ascend fused_infer_attention_score and paged_attention kernels
        # are validated for block size 128 in vllm-ascend. Allowing the
        # default MultipleOf(1) lets the V1 engine pick unsupported merged
        # storage block sizes (e.g. 784 for Qwen3.5 hybrid models), which
        # causes aclnnFusedInferAttentionScoreV3 to fail with error 561002.
        return [128]

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


register_backend(
    AttentionBackendEnum.CUSTOM,
    "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendAttentionBackend",
)


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

    @staticmethod
    def update_graph_params(
        update_stream: Any,
        forward_context: Any,
        num_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        graph_params = get_ascend_graph_params()
        if (
            graph_params is None
            or num_tokens not in graph_params.attention_params
            or not graph_params.attention_params[num_tokens]
        ):
            return

        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        params = graph_params.attention_params[num_tokens]
        handles = graph_params.handles[num_tokens]
        events = graph_params.events[num_tokens]
        use_pa = using_paged_attention(num_tokens, vllm_config)

        with torch.npu.stream(update_stream):
            for task_index, _ in enumerate(zip(params, handles, events)):
                AscendAttentionBackendImpl._update_graph_task(
                    update_stream,
                    forward_context,
                    num_tokens,
                    vllm_config,
                    task_index,
                    use_pa=use_pa,
                )

    @staticmethod
    def _update_graph_task(
        update_stream: Any,
        forward_context: Any,
        num_tokens: int,
        vllm_config: VllmConfig,
        task_index: int,
        *,
        use_pa: bool | None = None,
    ) -> bool:
        """Update one captured attention task; caller owns stream context."""
        graph_params = get_ascend_graph_params()
        if (
            graph_params is None
            or num_tokens not in graph_params.attention_params
            or num_tokens not in graph_params.handles
            or num_tokens not in graph_params.events
        ):
            return False
        params = graph_params.attention_params[num_tokens]
        handles = graph_params.handles[num_tokens]
        events = graph_params.events[num_tokens]
        if task_index >= min(len(params), len(handles), len(events)):
            return False

        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return False
        param = params[task_index]
        handle = handles[task_index]
        event = events[task_index]
        layer_name = param[0]
        metadata = attn_metadata.get(layer_name)
        if not isinstance(metadata, AscendMetadata):
            return False

        if use_pa is None:
            use_pa = using_paged_attention(num_tokens, vllm_config)
        if use_pa:
            (
                _,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_heads,
                scale,
                block_table,
                output,
            ) = param
            seq_lens = metadata.seq_lens
            workspace = torch_npu._npu_paged_attention_get_workspace(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                block_table=block_table,
                context_lens=seq_lens,
                out=output,
            )
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                block_table=block_table,
                context_lens=seq_lens,
                out=output,
                workspace=workspace,
            )
        else:
            (
                _,
                query,
                key_cache,
                value_cache,
                captured_block_table,
                attn_mask,
                block_size,
                num_kv_heads,
                num_heads,
                scale,
                output,
                softmax_lse,
                sparse_mode,
                pre_tokens,
                next_tokens,
            ) = param
            torch.npu.graph_task_update_begin(update_stream, handle)
            block_table = (
                captured_block_table
                if sparse_mode == 4
                else metadata.block_tables
            )
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key_cache,
                value=value_cache,
                block_table=block_table,
                atten_mask=attn_mask,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=metadata.seq_lens_list,
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                scale=scale,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                workspace=graph_params.workspaces.get(num_tokens),
                out=[output, softmax_lse],
            )
        torch.npu.graph_task_update_end(update_stream)
        event.record(update_stream)
        return True

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
            if slots.dtype != torch.int32:
                slots = slots.to(torch.int32)

            num_actual = attn_metadata.num_actual_tokens
            torch_npu._npu_reshape_and_cache(
                key=key[:num_actual],
                value=value[:num_actual],
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                slot_indices=slots[:num_actual]
            )
        return key, value

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """Forward pass using fused_infer_attention_score."""
        if is_ascend_graph_capturing():
            return self.full_graph_fused_infer_attention(
                query,
                key,
                value,
                attn_metadata,
                output,
                layer_name,
            )

        key, value, block_size, block_table, actual_seq_lengths_kv = \
            self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            key = key[:num_tokens]
            value = value[:num_tokens]

        # sparse_mode: 3 = causal with mask, 0 = no mask
        sparse_mode = 3 if attn_metadata.attn_mask is not None else 0

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
            sparse_mode=sparse_mode,
        )

        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def full_graph_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """Capture FIA and retain the task handle for host parameter update."""
        key, value, block_size, block_table, actual_seq_lengths_kv = (
            self._get_fia_params(key, value, attn_metadata)
        )
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        graph_params = get_ascend_graph_params()
        assert graph_params is not None

        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(
            1,
            dtype=query.dtype,
            device=query.device,
        )
        sparse_mode = (
            4
            if self.sliding_window is not None
            else 3 if attn_metadata.causal else 0
        )
        pre_tokens = self.sliding_window or SWA_INT_MAX
        next_tokens = 0 if self.sliding_window is not None else SWA_INT_MAX

        if workspace is None:
            workspace = (
                torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    sparse_mode=sparse_mode,
                    pre_tokens=pre_tokens,
                    next_tokens=next_tokens,
                    scale=self.scale,
                )
            )
            update_ascend_graph_params_workspace(num_tokens, workspace)

        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attention_params[num_tokens].append(
            (
                layer_name,
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask)
                if attn_metadata.attn_mask is not None
                else None,
                block_size,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
                sparse_mode,
                pre_tokens,
                next_tokens,
            )
        )

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
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
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        record_ascend_graph_task(
            num_tokens,
            "attention",
            len(graph_params.handles[num_tokens]) - 1,
            layer_name,
        )
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        layer_name: str = "",
    ) -> torch.Tensor:
        """Forward pass using paged attention for decode."""
        if is_ascend_graph_capturing():
            graph_params = get_ascend_graph_params()
            assert graph_params is not None
            num_tokens = query.shape[0]
            workspace = graph_params.workspaces[num_tokens]
            if workspace is None:
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
                graph_params.workspaces[num_tokens] = workspace

            stream = torch.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attention_params[num_tokens].append(
                (
                    layer_name,
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    weak_ref_tensors(attn_metadata.block_tables),
                    weak_ref_tensors(output),
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
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            record_ascend_graph_task(
                num_tokens,
                "attention",
                len(graph_params.handles[num_tokens]) - 1,
                layer_name,
            )
            return output

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
        layer_name: str,
    ):
        """Forward implementation dispatching to appropriate attention method."""
        num_tokens = query.shape[0]

        # Use paged attention for decode-only state
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
        ):
            output = self.forward_paged_attention(
                query, attn_metadata, output, layer_name
            )
        else:
            output = self.forward_fused_infer_attention(
                query,
                key,
                value,
                attn_metadata,
                output,
                layer_name,
            )

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
        # Note: kv_cache[0]/[1] may be non-contiguous views of a
        # [2, num_blocks, ...] tensor.  _npu_reshape_and_cache handles
        # them directly via slot_indices — no contiguous copy needed.
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
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            layer.layer_name,
        )
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

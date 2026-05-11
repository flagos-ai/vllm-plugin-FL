# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 2026 - Modified by Kunlunxin, Inc. All Rights Reserved.

"""
Kunlunxin attention backend for vllm-plugin-FL.

This module provides Kunlunxin attention using xtorch_ops:
  - xtorch_ops.prefill_attention(...)          -- prefill (flash attention)
  - xtorch_ops.decode_paged_attention(...)     -- decode (paged attention)
  - xtorch_ops.reshape_and_cache(...)          -- KV cache update (HND layout)
  - xtorch_ops.reshape_and_cache_flash(...)    -- KV cache update (hybrid attention, e.g. Qwen3-Next)

KV cache layout HND by default: (2, num_blocks, num_kv_heads, block_size, head_size)
"""

from __future__ import annotations

import os
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType
)

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    split_decodes_and_prefills,
    AttentionCGSupport
)

from vllm.v1.attention.backends.utils import AttentionMetadataBuilder

from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = logging.getLogger(__name__)

# Check torch_xmlir / xtorch_ops availability
_KUNLUNXIN_OPS_AVAILABLE = False
try:
    import torch_xmlir  # noqa: F401
    import xtorch_ops
    _KUNLUNXIN_OPS_AVAILABLE = True
except ImportError:
    logger.debug(
        "xtorch_ops or torch_xmlir not available. "
        "Kunlunxin attention ops will not work."
    )


def is_kunlunxin_ops_available() -> bool:
    """Check if Kunlunxin ops are available."""
    return _KUNLUNXIN_OPS_AVAILABLE


@dataclass
class KunlunxinMetadata:
    """Metadata for Kunlunxin attention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: torch.Tensor | None
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: torch.Tensor | None

    num_prefills: int

    num_prefill_tokens: int

    num_decode_tokens: int

    slot_mapping: torch.Tensor

    enable_kv_scales_calculation: bool

    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]


    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    num_actual_tokens: int
    # Whether or not if cuda graph is enabled.
    use_cuda_graph: bool

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None
    seq_lens_tensor_host: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Max number of key/value length in the batch, especially for prefix cache
    max_kv_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    query_start_loc_host: Optional[torch.Tensor] = None
    # serve only for prefix cache
    kv_prefix_start_loc_host: Optional[torch.Tensor] = None
    kv_prefix_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["KunlunxinMetadata"] = None
    _cached_decode_metadata: Optional["KunlunxinMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: Optional[torch.Tensor] = None

    # for spec decode api
    is_spec_decode: Optional[bool] = False

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None
        self.encoder_attn_bias: Optional[List[AttentionBias]] = None
        self.cross_attn_bias: Optional[List[AttentionBias]] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return ((self.encoder_seq_lens is not None)
                and (self.encoder_seq_lens_tensor is not None)
                and (self.max_encoder_seq_len is not None))

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return (self.is_all_encoder_attn_metadata_set
                and (self.cross_slot_mapping is not None)
                and (self.cross_block_tables is not None))

    @property
    def prefill_metadata(self) -> Optional["KunlunxinMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[-(self.num_prefills + 1):] - self.query_start_loc[-(self.num_prefills + 1)])
        # flash attention needs both lod information on host and device
        query_start_loc_host = (None if self.query_start_loc_host is None else
                           self.query_start_loc_host[-(self.num_prefills + 1):] - self.query_start_loc_host[-(self.num_prefills + 1)])

        kv_prefix_start_loc = (None if self.kv_prefix_start_loc is None else
                    self.kv_prefix_start_loc[-(self.num_prefills + 1):] - self.kv_prefix_start_loc[-(self.num_prefills + 1)])
        kv_prefix_start_loc_host = (None if self.kv_prefix_start_loc_host is None else
            self.kv_prefix_start_loc_host[-(self.num_prefills + 1):] - self.kv_prefix_start_loc_host[-(self.num_prefills + 1)])

        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[-self.num_prefill_tokens:])

        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[-self.num_prefills:])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[-self.num_prefills:])

        block_tables = (None if self.block_tables is None else
                        self.block_tables[-self.num_prefills:])
        input_positions = (None if self.input_positions is None else
                    self.input_positions[-self.num_prefills:])

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = KunlunxinMetadata(
            num_actual_tokens=self.num_actual_tokens,
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=None,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_kv_len=self.max_kv_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            query_start_loc_host=query_start_loc_host,
            input_positions=input_positions,
            kv_prefix_start_loc=kv_prefix_start_loc,
            kv_prefix_start_loc_host=kv_prefix_start_loc_host,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["KunlunxinMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        if self.num_prefills != 0:
            # Compute some attn_metadata fields which default to None
            slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:-self.num_prefill_tokens])
            seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:-self.num_prefills])
            seq_lens_tensor_host = (None if self.seq_lens_tensor_host is None else
                           self.seq_lens_tensor_host[:-self.num_prefills])

            block_tables = (None if self.block_tables is None else
                        self.block_tables[:-self.num_prefills])
            query_start_loc = (None if self.query_start_loc is None else
                               self.query_start_loc[:-self.num_prefills])
            query_start_loc_host = (None if self.query_start_loc_host is None else
                                    self.query_start_loc_host[:-self.num_prefills])
        else:
            # Compute some attn_metadata fields which default to None
            slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping)
            seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor)
            seq_lens_tensor_host = (None if self.seq_lens_tensor_host is None else
                           self.seq_lens_tensor_host)
            block_tables = (None if self.block_tables is None else
                        self.block_tables)
            query_start_loc = (None if self.query_start_loc is None else
                               self.query_start_loc[:-self.num_prefills])
            query_start_loc_host = (None if self.query_start_loc_host is None else
                                    self.query_start_loc_host[:-self.num_prefills])

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = KunlunxinMetadata(
            num_actual_tokens=self.num_actual_tokens,
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens_tensor=seq_lens_tensor,
            seq_lens_tensor_host=seq_lens_tensor_host,
            query_start_loc=query_start_loc,
            query_start_loc_host=query_start_loc_host,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False,
            is_spec_decode=self.is_spec_decode)
        return self._cached_decode_metadata


class KunlunxinAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builder for Kunlunxin attention metadata."""

    reorder_batch_threshold: ClassVar[int] = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec: AttentionSpec,
                 layer_names: list[str],
                 vllm_config: VllmConfig,
                 device: torch.device):
        self.device = device
        self.is_spec_decode = vllm_config.speculative_config is not None
        if self.is_spec_decode:
            self.reorder_batch_threshold = 1 + vllm_config.speculative_config.num_speculative_tokens

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: how if a prefilled sequence has only one token
            if num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill],
                                        decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens
        return modified_batch


    def build(
        self, common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False
    ) -> KunlunxinMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        seq_lens = common_attn_metadata.seq_lens

        max_kv_len = max(seq_lens_cpu).item()
        query_start_loc_host = common_attn_metadata.query_start_loc_cpu
        query_start_loc = common_attn_metadata.query_start_loc

        seq_lens_loc_host = torch.zeros(num_reqs + 1, dtype=torch.int32)
        seq_lens_loc_host[1:] = torch.cumsum(seq_lens_cpu, dim=0)
        seq_lens_loc = seq_lens_loc_host.to(self.device, non_blocking=True)
        seq_lens_host = seq_lens_cpu

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.reorder_batch_threshold)
        num_scheduled_tokens = torch.diff(query_start_loc_host)
        tmp_decode_scheduled_tokens = num_scheduled_tokens[:num_decodes]
        if num_decode_tokens == 0:
            max_decode_seq_len = 0
        else:
            max_decode_seq_len = torch.max(tmp_decode_scheduled_tokens).item()
        tmp_prefill_scheduled_tokens = num_scheduled_tokens[num_decodes: num_reqs]
        if num_prefill_tokens == 0:
            max_prefill_seq_len = 0
        else:
            max_prefill_seq_len = torch.max(tmp_prefill_scheduled_tokens).item()


        attn_metadata = KunlunxinMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            enable_kv_scales_calculation=True,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens_tensor=seq_lens,
            seq_lens_tensor_host=seq_lens_host,
            max_query_len=max_prefill_seq_len,
            max_decode_query_len=(self.reorder_batch_threshold if self.is_spec_decode else None),
            max_kv_len=max_kv_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            query_start_loc_host=query_start_loc_host,
            kv_prefix_start_loc_host=seq_lens_loc_host,
            kv_prefix_start_loc=seq_lens_loc,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=block_table_tensor,
            use_cuda_graph=False,
            is_spec_decode=self.is_spec_decode,
        )

        return attn_metadata


    def build_for_cudagraph_capture(
            self, common_attn_metadata: CommonAttentionMetadata):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with MHA.
        """
        m = common_attn_metadata
        assert m.num_reqs <= (m.num_actual_tokens *
                              self.reorder_batch_threshold), \
            "MHA only supports decode-only full CUDAGraph capture. " \
            "Make sure all cudagraph capture sizes <= max_num_seq."

        assert m.max_query_len <= self.reorder_batch_threshold  # decode only

        return self.build(0, m)

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class KunlunxinAttentionBackend(AttentionBackend):
    """
    Kunlunxin attention backend.

    KV cache shape: (2, num_blocks, num_kv_heads, block_size, head_size), must be contiguous
    """
    # crucial to cuda graph
    accept_output_buffer: bool = True

    supports_quant_query_input: bool = False

    @staticmethod
    def get_name() -> str:
        return "KUNLUNXIN_FL"

    @staticmethod
    def get_impl_cls() -> Type["KunlunxinAttentionBackendImpl"]:
        return KunlunxinAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["KunlunxinMetadata"]:
        return KunlunxinMetadata

    @staticmethod
    def get_builder_cls() -> Type["KunlunxinAttentionMetadataBuilder"]:
        return KunlunxinAttentionMetadataBuilder

    @staticmethod
    def get_supported_head_size() -> List[int]:
        # Note: 128 is best for performance
        return [32, 64, 80, 96, 112, 120, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> Tuple[int, ...]:
        # Kunlunxin uses NHD layout
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        raise NotImplementedError


class KunlunxinPagedAttention(PagedAttention):
    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = kv_cache.shape[1]
        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, -1, head_size)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, -1, head_size)
        return key_cache, value_cache

    @staticmethod
    def split_write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        # Split fused kv_cache [2, num_blocks, ...] into key_cache / value_cache
        key_cache, value_cache = KunlunxinPagedAttention.split_kv_cache(
            kv_cache,
            num_kv_heads,
            head_size,
        )
        xtorch_ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
        )

    def reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_max: torch.Tensor | None = None,
        v_max: torch.Tensor | None = None,
        quant_mode: int = 0,
        force_sdnn: bool = False,
        BLHD_LAYOUT: bool = True) -> int:
        """
        reshape and store key and value in cache_key, cache_value, respectively.

        Args:
            key (torch.Tensor): Shape [num_tokens, num_heads, head_size], dtype bf16/fp16/fp32.
            value (torch.Tensor): Shape [num_tokens, num_heads, head_size], same dtype as key. 
            Can be empty; quantization is not supported when value is empty.
            slot_mapping (torch.Tensor): Maps tokens to position indices in the cache.
            k_max (torch.Tensor, optional): Shape [num_heads] (quant_mode=0)
                                    or [ctx->max_ptr_size()] (quant_mode=2). Defaults to None. Dtype is fp32.
            v_max (torch.Tensor optional): Shape [num_heads] (quant_mode=0)
                                    or [ctx->max_ptr_size()] (quant_mode=2). Defaults to None. Dtype is fp32.
            quant_mode (int, optional): Quantization type. quant_mode=0:
                                                quant_mode=2: Block-wise quantization by max_ptr_size().
                                                Defaults to 0.
            force_sdnn (bool, optional): Hardware type.
            BLHD_LAYOUT (bool, optional): Whether to use BLHD_LAYOUT. Defaults to True (optimal performance).
        Returns:
            key_cache (torch.Tensor): Output cache for k, shape [num_blocks, block_size,  num_heads, head_size],
                same dtype as key or int8.
                When key is bfloat16, key_cache can be float16.
            value_cache (torch.Tensor): Output cache for v, shape [num_blocks, block_size,  num_heads, head_size], 
                same dtype as key.
                Can be empty; quantization is not supported when value is empty.
        """
        if key_cache.dtype is torch.int8 and force_sdnn:
            raise ValueError("reshape and cache flash use sdnn do not support quant")

        if value is None or value_cache is None:
            if key_cache.dtype is torch.int8:
                raise ValueError("reshape and cache flash k only do not support quant")

        return xtorch_ops.reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    k_max=k_max,
                    v_max=v_max,
                    quant_mode=quant_mode,
                    force_sdnn=force_sdnn,
                    BLHD_LAYOUT=BLHD_LAYOUT)

    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_host: torch.Tensor,
        max_seq_len: int,
        num_decode_tokens: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        max_window_size: int = -1,
        output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(query)
        assert max_window_size == -1, "not support sliding window"
        xtorch_ops.decode_paged_attention(
            query[:num_decode_tokens],
            key_cache,
            value_cache,
            seq_lens_host[:num_decode_tokens],
            seq_lens[:num_decode_tokens],
            block_tables,
            output[:num_decode_tokens],
            alpha=scale,
            k_perchannel_scale=k_scale,
            v_perchannel_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sink=None,
        )
        return output

class KunlunxinAttentionBackendImpl(AttentionImpl[KunlunxinMetadata]):
    """
    Kunlunxin attention implementation using xtorch_ops.
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
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not _KUNLUNXIN_OPS_AVAILABLE:
            raise RuntimeError(
                "xtorch_ops is required for Kunlunxin attention backend. "
                "Please install xtorch_ops for Kunlunxin hardware support."
            )

        if logits_soft_cap is not None:
            raise ValueError(
                "kunlunxinAttention does not support attention logits soft capping.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        # specific atten_scale for attention kernel
        self.adjusted_scale = self.scale * math.sqrt(self.head_size)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # 0.13.0 get_supported_head_size is not avaiable any more
        suppored_head_sizes = KunlunxinAttentionBackend.get_supported_head_size()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by KunlunxinAttentionBackend. "
                f"Supported head sizes are: {suppored_head_sizes}.")

    def _get_window_size(
        self, attn_type: AttentionType
    ) -> Optional[Tuple[int, int]]:
        if self.sliding_window is None or self.sliding_window <= 0:
            return None
        if attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            return (self.sliding_window, self.sliding_window)
        return (self.sliding_window, 0)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: Optional[KunlunxinMetadata],
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = query.view(-1, self.num_heads, self.head_size)
        if output is None:
            output = torch.empty_like(query)
        if attn_metadata is None:
            # Profiling run.

            # make fake-meta-data, and let attn run during warmup period. In order to avoid cuda-graph fail risk
            fake_meta = KunlunxinMetadata(seq_lens_tensor=None,
                                       max_decode_seq_len=1,
                                       block_tables=None,
                                       num_prefills=1,
                                       num_prefill_tokens=1,
                                       num_decode_tokens=1,
                                       slot_mapping=None,
                                       enable_kv_scales_calculation=False,
                                       max_prefill_seq_len=1,
                                       num_actual_tokens=1,
                                       use_cuda_graph=False, # up params is useless to context_forward
                                       max_query_len=query.shape[0],
                                       max_kv_len=query.shape[0],
                                       query_start_loc=torch.tensor((0, query.shape[0]), dtype=torch.int32, device=query.device),
                                       query_start_loc_host=torch.tensor((0, query.shape[0]), dtype=torch.int32, device="cpu"),
                                       kv_prefix_start_loc=torch.tensor([0, query.shape[0]], dtype=torch.int32, device=query.device),
                                       kv_prefix_start_loc_host=torch.tensor([0, query.shape[0]], dtype=torch.int32, device="cpu"))
            self.context_forward(query, key, value, output, fake_meta, attn_type=attn_type)

            return output.view(-1, self.num_heads * self.head_size)

        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        # Self-attention vs. cross-attention will impact
        # which KV cache memory-mapping & which
        # seqlen datastructures we utilize
        # Encoder and encoder-only attention do not use KV cache
        if (attn_type != AttentionType.ENCODER and
            attn_type != AttentionType.ENCODER_ONLY and
            kv_cache.numel() > 0):
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention

            updated_slot_mapping = attn_metadata.slot_mapping

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory
            # profiling run.
            if os.environ.get("USE_RESHAPE_AND_CACHE_FLASH", "0") == "1":
                key_cache, value_cache = KunlunxinPagedAttention.split_kv_cache(
                    kv_cache, self.num_kv_heads, self.head_size
                )
                key = key.contiguous()
                value = value.contiguous()
                KunlunxinPagedAttention.reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    updated_slot_mapping,
                    BLHD_LAYOUT=False,
                )
            else:
                key = key.contiguous()
                value = value.contiguous()
                KunlunxinPagedAttention.split_write_to_paged_cache(
                                                key,
                                                value,
                                                kv_cache,
                                                self.num_kv_heads,
                                                self.head_size,
                                                updated_slot_mapping,
                                                self.kv_cache_dtype,
                                                k_scale, v_scale)
                key_cache, value_cache = KunlunxinPagedAttention.split_kv_cache(
                    kv_cache, self.num_kv_heads, self.head_size
                )

        # Handle encoder-only and encoder attention (e.g., BERT, BGE models)
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # Encoder attention: use bidirectional attention without KV cache
            num_actual_tokens = attn_metadata.num_actual_tokens
            self.context_forward(
                query[:num_actual_tokens],
                key[:num_actual_tokens] if key is not None else None,
                value[:num_actual_tokens] if value is not None else None,
                output[:num_actual_tokens],
                attn_metadata,
                attn_type=attn_type,
                key_cache=None,
                value_cache=None,
                is_causal=False  # Bidirectional attention for encoder
            )
            return output.view(-1, self.num_heads * self.head_size)

        # Decoder self-attention supports chunked prefill.
        assert attn_type == AttentionType.DECODER, \
            f"Unsupported attention type: {attn_type}"
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Only enforce this shape-constraint for decoder
        # self-attention
        if prefill_meta := attn_metadata.prefill_metadata:
            # prefill
            prefill_query = query[num_decode_tokens:attn_metadata.num_actual_tokens]
            prefill_key = key[num_decode_tokens:attn_metadata.num_actual_tokens]
            prefill_value = value[num_decode_tokens:attn_metadata.num_actual_tokens]
            assert prefill_query.shape[0] == num_prefill_tokens
            prefill_out = output[num_decode_tokens:attn_metadata.num_actual_tokens]
            self.context_forward(
                prefill_query,
                prefill_key,
                prefill_value,
                prefill_out,
                prefill_meta,
                attn_type=attn_type,
                key_cache=key_cache,
                value_cache=value_cache,
                is_causal=True
            )

        if num_decode_tokens != 0:
            decode_meta = attn_metadata.decode_metadata
            if os.environ.get("USE_RESHAPE_AND_CACHE_FLASH", "0") == "1":
                # For hybrid Attention (Qwen3-Next, Qwen3.5)
                tmp_block_tables = (
                    decode_meta.block_tables * 2
                )
            else:
                tmp_block_tables = decode_meta.block_tables

            if attn_metadata.decode_metadata.is_spec_decode:
                # query_start_loc_host = decode_meta.query_start_loc_host.to(torch.int32)
                # TODO: add MTP support
                assert False, "speculative_attention_variable not implemented"

            else:
                max_window_size = -1
                if self.sliding_window is not None and self.sliding_window > 0:
                    max_window_size = self.sliding_window
                KunlunxinPagedAttention.forward_decode(
                    query,
                    key_cache,
                    value_cache,
                    tmp_block_tables,
                    attn_metadata.seq_lens_tensor,
                    attn_metadata.seq_lens_tensor_host,
                    attn_metadata.max_decode_seq_len,
                    num_decode_tokens,
                    self.kv_cache_dtype,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                    k_scale,
                    v_scale,
                    max_window_size=max_window_size,
                    output=output
                )
        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def context_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        attn_metadata: KunlunxinMetadata,
        attn_type: AttentionType = AttentionType.DECODER,
        key_cache: torch.Tensor = None,
        value_cache: torch.Tensor = None,
        is_causal=True
    ):
        if query is not None and not query.is_contiguous():
            query = query.contiguous()
        if key is not None and not key.is_contiguous():
            key = key.contiguous()
        if value is not None and not value.is_contiguous():
            value = value.contiguous()

        actual_query_start_loc = attn_metadata.query_start_loc
        actual_query_start_loc_host = attn_metadata.query_start_loc_host
        kv_prefix_start_loc = attn_metadata.kv_prefix_start_loc
        kv_prefix_start_loc_host = attn_metadata.kv_prefix_start_loc_host
        window_size = self._get_window_size(attn_type)
        window_left = -1 if window_size is None else window_size[0]
        window_right = -1 if window_size is None else window_size[1]

        # prefix cache part
        if actual_query_start_loc_host[-1] != kv_prefix_start_loc_host[-1]:
            max_kv_len = attn_metadata.max_kv_len
            if os.environ.get("USE_RESHAPE_AND_CACHE_FLASH", "0") == "1":
                # For hybrid Attention (Qwen3-Next, Qwen3.5)
                tmp_block_tables = (attn_metadata.block_tables * 2)
            else:
                tmp_block_tables = attn_metadata.block_tables
            xtorch_ops.prefill_attention(
                query,
                key_cache,
                value_cache,
                out,
                is_causal=is_causal,
                is_prefix_cache=True,
                alpha=self.adjusted_scale,
                context_qlen_lod_cpu=actual_query_start_loc_host,
                context_qlen_lod_xpu=actual_query_start_loc,
                context_kvlen_lod_cpu=kv_prefix_start_loc_host,
                context_kvlen_lod_xpu=kv_prefix_start_loc,
                block_table=tmp_block_tables,
                alibi_slopes=self.alibi_slopes,
                swa_left=window_left,
                swa_right=window_right,
            )
        # no prefix cache part
        else:
            max_kv_len = attn_metadata.max_query_len
            xtorch_ops.prefill_attention(
                query,
                key,
                value,
                out,
                is_causal=is_causal,
                is_prefix_cache=False,
                alpha=self.adjusted_scale,
                context_qlen_lod_cpu=actual_query_start_loc_host,
                context_qlen_lod_xpu=actual_query_start_loc,
                alibi_slopes=self.alibi_slopes,
                swa_left=window_left,
                swa_right=window_right,
            )


__all__ = [
    "KunlunxinAttentionBackend",
    "KunlunxinAttentionBackendImpl",
    "KunlunxinAttentionMetadataBuilder",
    "KunlunxinMetadata",
    "is_kunlunxin_ops_available",
]

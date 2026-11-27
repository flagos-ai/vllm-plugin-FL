# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""V4.1 cache ownership and metadata, using the installed vLLM allocator."""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    KVCacheLayout,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.block_table import get_block_table_width

from .attention_metadata import pad_cache_metadata, prepare_cache_metadata


@dataclass
class DeepseekV41CacheMetadata:
    slot_mapping: torch.Tensor
    group_positions: torch.Tensor
    context_lens: torch.Tensor
    token_to_req: torch.Tensor
    block_table: torch.Tensor
    swa_indices: torch.Tensor | None
    token_block_tables: torch.Tensor | None
    compress_ratio: int
    storage_block_size: int
    padding_buffers: tuple | None = None

    def for_num_tokens(self, num_tokens):
        """Expose initialized padding when DP aligns a draft's eager token count."""
        if num_tokens == self.slot_mapping.numel():
            return self
        if self.padding_buffers is None or num_tokens > self.padding_buffers[0].numel():
            raise ValueError("V4.1 attention exceeds its initialized metadata capacity")
        slots, positions, lengths, requests, swa, tables = self.padding_buffers
        columns = (
            None
            if self.token_block_tables is None
            else self.token_block_tables.shape[1]
        )
        return DeepseekV41CacheMetadata(
            slots[:num_tokens],
            positions[:num_tokens],
            lengths[:num_tokens],
            requests[:num_tokens],
            self.block_table,
            None if swa is None else swa[:num_tokens],
            None if tables is None else tables[:num_tokens, :columns],
            self.compress_ratio,
            self.storage_block_size,
            self.padding_buffers,
        )


class DeepseekV41CacheBackend(AttentionBackend):
    @classmethod
    def supported_kv_cache_layouts(cls):
        # Different SWA/compressed/index page sizes use the native packed
        # allocator. Each layer retains its own pitched, contiguous NHC page.
        return (KVCacheLayout.BLNHC,)

    @staticmethod
    def get_name():
        return "DEEPSEEK_V41_TEXT_CACHE"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        # These are logical block sizes; allocator uses storage_block_size for
        # compressed pages, as installed attn_utils.reshape_kv_cache requires.
        return [MultipleOf(64)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size):
        return 64

    @classmethod
    def get_supported_head_sizes(cls):
        return [512, 132]

    @staticmethod
    def get_builder_cls():
        return DeepseekV41CacheMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        if (
            num_kv_heads != 1
            or block_size not in (32, 64)
            or head_size not in (512, 132)
        ):
            raise ValueError(
                "V4.1 cache requires one head, physical pages32/64 and head512/132"
            )
        return (num_blocks, block_size, 584 if head_size == 512 else 132)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (0, 1, 2, 3) if include_num_layers_dimension else (0, 1, 2)


class DeepseekV41Cache(nn.Module, AttentionLayerBase):
    """A single allocator-owned cache; consumers keep a reference to its owner."""

    def __init__(self, vllm_config, prefix, *, kind, compress_ratio=1):
        super().__init__()
        if kind not in ("swa", "compressed", "index"):
            raise ValueError("unknown V4.1 cache kind")
        if (
            isinstance(compress_ratio, bool)
            or not isinstance(compress_ratio, int)
            or compress_ratio not in (1, 2)
        ):
            raise ValueError("V4.1 compression ratio must be integer 1 or 2")
        if kind == "swa" and compress_ratio != 1:
            raise ValueError("SWA cache is not compressed")
        if vllm_config.cache_config.block_size != 64:
            raise ValueError("V4.1 FP8 paged MQA requires logical block_size=64")
        if vllm_config.cache_config.cache_dtype != "fp8_ds_mla":
            raise ValueError("V4.1 text cache requires fp8_ds_mla")
        if (
            getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1) != 1
            or getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
            != 1
        ):
            raise NotImplementedError(
                "V4.1 cache metadata supports TP/PP without context parallelism"
            )
        self.prefix, self.kind, self.compress_ratio = prefix, kind, compress_ratio
        self.block_size, self.head_size = 64, 132 if kind == "index" else 512
        self.kv_cache = torch.empty(0, dtype=torch.uint8)
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"duplicate V4.1 cache prefix: {prefix}")
        context[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        common = dict(
            block_size=64,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=torch.uint8,
            alignment=576,
        )
        if self.kind == "swa":
            return SlidingWindowMLASpec(
                **common,
                sliding_window=128,
                cache_dtype_str="fp8_ds_mla",
                state_content_bytes=584,
                model_version="deepseek_v4",
                kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
            )
        if self.kind == "index":
            return MLAAttentionSpec(**common, tokens_per_state=self.compress_ratio)
        return MLAAttentionSpec(
            **common,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str="fp8_ds_mla",
            state_content_bytes=584,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
        )

    def get_attn_backend(self):
        return DeepseekV41CacheBackend

    def bind_kv_cache(self, cache):
        if cache.ndim == 4 and cache.shape[1] == 1:
            cache = cache.squeeze(1)
        expected = 132 if self.kind == "index" else 584
        if (
            cache.ndim != 3
            or cache.shape[1:] != (64 // self.compress_ratio, expected)
            or cache.dtype != torch.uint8
            or cache.stride(-1) != 1
            or cache.stride(1) != expected
            or cache.stride(0) < cache.shape[1] * expected
        ):
            raise ValueError(
                f"allocated V4.1 cache incompatible: shape={tuple(cache.shape)}, "
                f"dtype={cache.dtype}, stride={cache.stride()}, "
                f"expected rows={64 // self.compress_ratio}, row_bytes={expected}"
            )
        if self.kind == "index" and cache.stride(0) % 4:
            raise ValueError("indexer FP32 scales require aligned physical pages")
        self.kv_cache = cache

    def as_page_bytes(self):
        """Content-only view; preserve allocator pitch without exposing other layers."""
        cache = self.kv_cache
        if cache.ndim != 3:
            raise ValueError("V4.1 cache is not bound")
        return cache.view(cache.shape[0], -1)

    def as_attention_cache(self):
        """Public attention APIs consume [pages,rows,one_head,row_bytes]."""
        if self.kv_cache.ndim != 3:
            raise ValueError("V4.1 cache is not bound")
        return self.kv_cache.unsqueeze(2)

    def forward(self):
        return None


class DeepseekV41CacheMetadataBuilder(
    AttentionMetadataBuilder[DeepseekV41CacheMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    reorder_batch_threshold = None

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if (
            not isinstance(kv_cache_spec, (MLAAttentionSpec, SlidingWindowMLASpec))
            or kv_cache_spec.block_size != 64
        ):
            raise ValueError("V4.1 metadata requires logical 64-token MLA specs")
        if kv_cache_spec.head_size not in (
            512,
            132,
        ) or kv_cache_spec.tokens_per_state not in (1, 2):
            raise ValueError("unsupported V4.1 cache geometry")
        self.causal = not vllm_config.attention_config.use_non_causal
        if not self.causal and not all(
            vllm_config.compilation_config.static_forward_context[name].kind == "swa"
            for name in layer_names
        ):
            raise ValueError("DSpark noncausal metadata supports SWA draft layers only")
        self.ratio = kv_cache_spec.tokens_per_state
        self.kind = (
            "swa"
            if isinstance(kv_cache_spec, SlidingWindowMLASpec)
            else ("index" if kv_cache_spec.head_size == 132 else "compressed")
        )
        self.max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_blocks = (vllm_config.model_config.max_model_len + 63) // 64
        self.table_width = get_block_table_width(max_blocks, 64)
        self.owner = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ]
        self.requests = torch.zeros(self.max_tokens, device=device, dtype=torch.int32)
        self.slots = torch.empty(self.max_tokens, device=device, dtype=torch.int64)
        self.group_positions = torch.empty_like(self.slots)
        self.lengths = torch.empty(self.max_tokens, device=device, dtype=torch.int32)
        self.swa = (
            torch.empty(
                (self.max_tokens, 1, 128 if self.causal else 192),
                device=device,
                dtype=torch.int32,
            )
            if self.kind == "swa"
            else None
        )
        self.tables = (
            torch.empty(
                (self.max_tokens, self.table_width), device=device, dtype=torch.int32
            )
            if self.kind == "index"
            else None
        )

    def build(
        self,
        common_prefix_len,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build=False,
    ):
        common = common_attn_metadata
        if common.causal != self.causal:
            raise ValueError("V4.1 target/draft attention causality mismatch")
        n = common.slot_mapping.numel()
        if n > self.max_tokens or common.block_table_tensor.shape[1] > self.table_width:
            raise ValueError("V4.1 metadata exceeds frozen token/table capacity")
        mapped_requests = common.token_to_req_indices(self.requests)
        # A different attention builder may already own the native cached map.
        # Retain this builder's stable buffer for metadata and graph replay.
        if mapped_requests.data_ptr() != self.requests.data_ptr():
            self.requests[: mapped_requests.numel()].copy_(mapped_requests)
        # Native eager drafting builds metadata for real query rows while DP
        # may run the model at a larger common shape. Never expose stale rows.
        pad_cache_metadata(
            self.slots,
            self.group_positions,
            self.lengths,
            self.requests,
            self.swa,
            self.tables,
            n,
        )
        # Native mapping may return only actual rows. Padding is guarded by
        # original_slots and zero context lengths, so keep the full stable view.
        requests = self.requests[:n]
        slots, groups, lengths = (
            self.slots[:n],
            self.group_positions[:n],
            self.lengths[:n],
        )
        swa = self.swa[:n] if self.swa is not None else None
        tables = (
            self.tables[:n, : common.block_table_tensor.shape[1]]
            if self.tables is not None
            else None
        )
        positions = common.positions[:n] if common.positions is not None else None
        cache = self.owner.kv_cache
        blocks = cache.shape[0] if cache.ndim == 3 else 0
        prepare_cache_metadata(
            positions,
            common.query_start_loc,
            common.seq_lens,
            requests,
            common.block_table_tensor,
            common.slot_mapping,
            common.num_actual_tokens,
            common.num_reqs,
            blocks,
            self.ratio,
            64,
            slots,
            groups,
            lengths,
            swa_indices_out=swa,
            token_tables_out=tables,
            causal=self.causal,
        )
        return DeepseekV41CacheMetadata(
            slots,
            groups,
            lengths,
            requests,
            common.block_table_tensor,
            swa,
            tables,
            self.ratio,
            64 // self.ratio,
            (
                self.slots,
                self.group_positions,
                self.lengths,
                self.requests,
                self.swa,
                self.tables,
            ),
        )

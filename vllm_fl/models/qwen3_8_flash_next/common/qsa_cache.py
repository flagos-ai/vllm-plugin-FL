# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged side-cache ownership and metadata for Qwen3.8-Flash-Next QSA.

Each QSA layer keeps an uncompressed raw index key and one compressed key.
MRoPE models pack exact three-axis positions into an integer-typed tail of
each raw-key page. Text models keep only the key and derive group positions
from logical positions. The compressed owner uses
``MLAAttentionSpec.compress_ratio`` so its block table and physical storage
follow the main KV-cache lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)


def canonical_qsa_rope_positions(positions: torch.Tensor) -> torch.Tensor:
    """Return exact per-token positions as ``[tokens, 1, 3]`` int64 rows."""

    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    elif positions.ndim != 2 or positions.shape[0] not in (1, 3):
        raise ValueError("QSA RoPE positions must be [tokens] or [1|3, tokens]")
    if positions.shape[0] == 1:
        positions = positions.expand(3, -1)
    return positions.transpose(0, 1).unsqueeze(1).to(torch.int64)


def _logical_positions(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    num_tokens: int,
    arange: torch.Tensor | None = None,
) -> torch.Tensor:
    if num_tokens == 0:
        return seq_lens.new_empty((0,), dtype=torch.int64)
    if arange is None:
        arange = torch.arange(num_tokens, device=query_start_loc.device)
    else:
        arange = arange[:num_tokens]
    requests = token_to_req[:num_tokens].long()
    query_lens = torch.diff(query_start_loc)
    within_query = arange - query_start_loc.index_select(0, requests)
    return (
        seq_lens.index_select(0, requests).long()
        - query_lens.index_select(0, requests).long()
        + within_query.long()
    )


def _populate_token_to_req(
    common: CommonAttentionMetadata,
    out: torch.Tensor,
) -> torch.Tensor:
    """Pure-Torch fallback for platforms that cannot launch QSA Triton.

    The production accelerator path writes this buffer from the fused metadata
    kernel.  Keeping the fallback here preserves vLLM 0.24 and non-Triton
    compatibility without making the graph-captured path depend on a private
    backend operator.
    """

    num_tokens = common.num_actual_tokens
    num_mapped_tokens = int(common.query_start_loc_cpu[-1])
    if num_mapped_tokens:
        query_lens = common.query_start_loc[1:] - common.query_start_loc[:-1]
        mapped = torch.repeat_interleave(
            torch.arange(
                query_lens.shape[0],
                dtype=torch.int32,
                device=out.device,
            ),
            query_lens,
            output_size=num_mapped_tokens,
        )
        out[:num_mapped_tokens].copy_(mapped)
    if num_mapped_tokens < num_tokens:
        out[num_mapped_tokens:num_tokens].zero_()
    return out[:num_tokens]


def _logical_to_physical_qsa_slots(
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("QSA cache block size must be positive")
    if block_table.ndim != 2:
        raise ValueError("QSA block table must be two-dimensional")
    if request_indices.shape != logical_positions.shape:
        request_indices = torch.broadcast_to(request_indices, logical_positions.shape)

    requests = request_indices.to(device=block_table.device, dtype=torch.long)
    positions = logical_positions.to(device=block_table.device, dtype=torch.long)
    valid = (requests >= 0) & (requests < block_table.shape[0]) & (positions >= 0)
    logical_blocks = torch.div(
        positions.clamp_min(0), block_size, rounding_mode="floor"
    )
    valid &= logical_blocks < block_table.shape[1]
    safe_requests = requests.clamp(0, max(block_table.shape[0] - 1, 0))
    safe_blocks = logical_blocks.clamp(0, max(block_table.shape[1] - 1, 0))
    if not all(block_table.shape):
        return torch.full_like(positions, -1)
    physical_blocks = block_table[safe_requests, safe_blocks].long()
    valid &= physical_blocks >= 0
    slots = physical_blocks * block_size + positions.remainder(block_size)
    return torch.where(valid, slots, torch.full_like(slots, -1))


def compressed_qsa_slot_mapping(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build boundary-only slots for an ``MLAAttentionSpec`` QSA cache."""

    if storage_block_size <= 0 or compress_ratio <= 0:
        raise ValueError("QSA block size and compression ratio must be positive")
    compressed_positions = torch.div(
        logical_positions.clamp_min(0), compress_ratio, rounding_mode="floor"
    )
    slots = _logical_to_physical_qsa_slots(
        block_table,
        token_to_req,
        compressed_positions,
        storage_block_size,
    )
    valid = (logical_positions >= 0) & (
        (logical_positions + 1).remainder(compress_ratio) == 0
    )
    slots = torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int64)
    if out is not None:
        out.fill_(-1)
        out[: slots.numel()].copy_(slots)
        return out[: slots.numel()]
    return slots


def _global_full_cudagraph_enabled(vllm_config: VllmConfig) -> bool:
    """Honor the resolved vLLM graph mode without enabling graphs locally."""

    cudagraph_mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
    has_full_cudagraphs = getattr(cudagraph_mode, "has_full_cudagraphs", None)
    return bool(has_full_cudagraphs is not None and has_full_cudagraphs())


@dataclass
class QSAForwardMetadata(AttentionMetadata):
    """Common per-forward metadata for one QSA side cache."""

    common: CommonAttentionMetadata
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    qsa_slot_mapping: torch.Tensor
    storage_block_size: int
    compress_ratio: int
    _prepared: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def block_table(self) -> torch.Tensor:
        return self.common.block_table_tensor

    @property
    def slot_mapping(self) -> torch.Tensor:
        """Backward-compatible name used by the existing QSA call sites."""

        return self.qsa_slot_mapping

    @property
    def seq_lens(self) -> torch.Tensor:
        return self.common.seq_lens

    @property
    def query_start_loc(self) -> torch.Tensor:
        return self.common.query_start_loc

    @property
    def num_actual_tokens(self) -> int:
        return self.common.num_actual_tokens

    def prepare(self) -> None:
        """Populate replay-sensitive outputs exactly once for this forward."""

        if self._prepared:
            return
        num_tokens = self.num_actual_tokens
        if self.compress_ratio == 1:
            # Ratio-one cache owners reuse common.slot_mapping and do not launch
            # the fused metadata kernel.  Their logical positions are not used
            # by the production QSA path; keep this reference implementation for
            # direct/backward-compatible callers.
            _populate_token_to_req(self.common, self.token_to_req)
            num_mapped_tokens = int(self.common.query_start_loc_cpu[-1])
            self.logical_positions[:num_mapped_tokens].copy_(
                _logical_positions(
                    self.query_start_loc,
                    self.seq_lens,
                    self.token_to_req[:num_mapped_tokens],
                    num_mapped_tokens,
                )
            )
            if num_mapped_tokens < num_tokens:
                self.logical_positions[num_mapped_tokens:].fill_(-1)
            self._prepared = True
            return

        from ..gpu.ops.qsa import build_qsa_forward_metadata

        if not build_qsa_forward_metadata(
            self.token_to_req,
            self.query_start_loc,
            self.seq_lens,
            self.block_table,
            self.common.slot_mapping[:num_tokens],
            self.logical_positions,
            self.qsa_slot_mapping,
            self.storage_block_size,
            self.compress_ratio,
        ):
            # Triton unavailable/unsupported: preserve the pre-existing pure
            # Torch behavior.  Builders only defer this call into model forward
            # when the Triton capability check succeeded, so this fallback does
            # not become an allocation-heavy graph-capture path.
            _populate_token_to_req(self.common, self.token_to_req)
            num_mapped_tokens = int(self.common.query_start_loc_cpu[-1])
            self.logical_positions[:num_mapped_tokens].copy_(
                _logical_positions(
                    self.query_start_loc,
                    self.seq_lens,
                    self.token_to_req[:num_mapped_tokens],
                    num_mapped_tokens,
                )
            )
            if num_mapped_tokens < num_tokens:
                self.logical_positions[num_mapped_tokens:].fill_(-1)
            compressed_qsa_slot_mapping(
                self.block_table,
                self.token_to_req,
                self.logical_positions,
                self.storage_block_size,
                self.compress_ratio,
                self.qsa_slot_mapping,
            )
            self.qsa_slot_mapping.masked_fill_(
                self.common.slot_mapping[:num_tokens] < 0, -1
            )
        self._prepared = True


class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
    """Build QSA metadata from vLLM's cache-group-specific common metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compress_ratio = (
            kv_cache_spec.compress_ratio
            if isinstance(kv_cache_spec, MLAAttentionSpec)
            else 1
        )
        self.storage_block_size = kv_cache_spec.storage_block_size
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        try:
            from ..gpu.ops.qsa import qsa_forward_metadata_triton_supported

            triton_metadata_supported = qsa_forward_metadata_triton_supported(device)
        except (ImportError, RuntimeError):
            triton_metadata_supported = False
        # Capability never enables graphs by itself.  We only defer the launch
        # into model forward when the already-resolved global config requests a
        # full graph; otherwise metadata is built eagerly once below.
        self.defer_metadata_to_forward = (
            _global_full_cudagraph_enabled(vllm_config)
            and triton_metadata_supported
        )
        self.token_to_req_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        # Graph capture setup may inspect the fixed-address metadata storage
        # before the first replay-sensitive producer runs.  Give padding and
        # ratio-one/raw-cache users the same deterministic zero sentinel as the
        # old repeat_interleave path without rebuilding it on every forward.
        self.token_to_req_buffer.zero_()
        self.slot_mapping_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        self.logical_positions_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        self.logical_positions_buffer.fill_(-1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QSAForwardMetadata:
        del common_prefix_len, fast_build
        num_tokens = common_attn_metadata.num_actual_tokens
        # The compressed cache's prepare() call is the sole producer.  On an
        # accelerator it writes token_to_req together with logical positions
        # and compressed slots from one graph-safe Triton launch.  The ratio-one
        # raw-cache metadata never consumes this buffer on the production GPU
        # path; its CPU/non-Triton prepare() retains the pure-Torch fallback.
        token_to_req = self.token_to_req_buffer[:num_tokens]
        logical_positions = self.logical_positions_buffer[:num_tokens]
        if self.compress_ratio == 1:
            slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
        else:
            slot_mapping = self.slot_mapping_buffer[:num_tokens]
        metadata = QSAForwardMetadata(
            common=common_attn_metadata,
            token_to_req=token_to_req,
            logical_positions=logical_positions,
            qsa_slot_mapping=slot_mapping,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )
        if self.compress_ratio > 1 and not self.defer_metadata_to_forward:
            metadata.prepare()
        return metadata

class QSAStateBackend(AttentionBackend):
    """Key-only dummy backend for out-of-band BF16 QSA side-cache operations."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_EXP_QSA_STATE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "QSA state caches run out-of-band and have no attention impl"
        )

    @staticmethod
    def get_builder_cls() -> type[QSAMetadataBuilder]:
        return QSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        if num_kv_heads != 1:
            raise ValueError("QSA side caches require exactly one KV head")
        return (num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        # This side cache uses the identity/layered layout below, not the
        # block-major layout required by vLLM's cross-layer packing path.
        return False

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)


class _QSAStateCache(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(
        self,
        *,
        head_size: int,
        dtype: torch.dtype,
        cache_config: CacheConfig,
        prefix: str,
        vllm_config: VllmConfig,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        if head_size <= 0:
            raise ValueError("QSA cache head size must be positive")
        if compress_ratio <= 0:
            raise ValueError("QSA compression ratio must be positive")
        if cache_config.block_size % compress_ratio:
            raise ValueError(
                "QSA cache block size must be divisible by the compression ratio"
            )
        self.head_size = head_size
        self.dtype = dtype
        self.cache_config = cache_config
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.kv_cache = torch.tensor([])

        static_context = vllm_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def forward(self) -> None: ...

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind storage on both legacy direct-assign and newer hook runtimes."""
        self.kv_cache = kv_cache

    def get_attn_backend(self) -> type[AttentionBackend]:
        return QSAStateBackend


class QSAKeyStateCache(_QSAStateCache):
    """Raw BF16 key, optionally followed by exact int64 MRoPE positions."""

    _BF16_PER_INT64 = 4
    _NUM_ROPE_AXES = 3

    def __init__(self, *, cache_rope_positions: bool = False, **kwargs) -> None:
        key_head_size = int(kwargs.pop("head_size"))
        self.key_head_size = key_head_size
        self.cache_rope_positions = bool(cache_rope_positions)
        self.rope_position_offset = (
            (key_head_size + self._BF16_PER_INT64 - 1) // self._BF16_PER_INT64
        ) * self._BF16_PER_INT64
        storage_head_size = key_head_size
        if self.cache_rope_positions:
            storage_head_size = self.rope_position_offset + (
                self._NUM_ROPE_AXES * self._BF16_PER_INT64
            )
        super().__init__(head_size=storage_head_size, **kwargs)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if kv_cache.ndim != 4 or kv_cache.shape[2] != 1:
            raise ValueError("QSA raw cache must be [blocks, block_size, 1, width]")
        if kv_cache.dtype != torch.bfloat16 or kv_cache.shape[3] != self.head_size:
            raise ValueError("QSA raw cache does not match its packed BF16 cache spec")
        super().bind_kv_cache(kv_cache)
        self.key_cache = kv_cache[..., : self.key_head_size]
        if self.cache_rope_positions:
            position_tail = kv_cache[..., self.rope_position_offset :]
            self.rope_position_cache = position_tail.view(torch.int64)
        else:
            self.rope_position_cache = None

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return FullAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self.dtype,
        )


class QSACompressedKeyCache(_QSAStateCache):
    """Normalized, group-first-RoPE BF16 key at one row per complete group."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
        )


__all__ = [
    "QSACompressedKeyCache",
    "QSAForwardMetadata",
    "QSAKeyStateCache",
    "QSAMetadataBuilder",
    "QSAStateBackend",
    "canonical_qsa_rope_positions",
    "compressed_qsa_slot_mapping",
]

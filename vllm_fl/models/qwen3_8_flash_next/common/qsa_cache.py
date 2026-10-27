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

from contextlib import contextmanager
from contextvars import ContextVar
from vllm.platforms import current_platform

from dataclasses import dataclass, field, fields as dataclass_fields
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


def _spec_supports_block_stride_flag() -> bool:
    """Whether this vLLM ABI carries the padded-page block-stride contract.

    vLLM 0.24 added ``AttentionSpec.indexes_kv_by_block_stride`` so a side
    cache can be padded to a unified page size while still being read through
    a strided view.  Older builds lack the field entirely; the exporter keeps
    the flag out of the spec there instead of raising at construction time.
    """

    try:
        return any(
            field.name == "indexes_kv_by_block_stride"
            for field in dataclass_fields(AttentionSpec)
        )
    except Exception:
        return False


_BLOCK_STRIDE_SPEC_FLAG = _spec_supports_block_stride_flag()


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

# Width is a graph geometry choice, independent of cache allocation/page size.
_QSA_CAPTURE_COLUMNS: ContextVar[int | None] = ContextVar(
    "qwen38_qsa_capture_columns", default=None
)


def qsa_column_buckets(capacity: int) -> tuple[int, ...]:
    if capacity <= 0:
        return (0,)
    return tuple(sorted({min(capacity, n) for n in (512, 2048, 8192, capacity)}))


def qsa_score_columns(max_seq_len: int, compress_ratio: int, capacity: int) -> int:
    """Cover all visible complete groups using a trusted CPU length bound."""
    if max_seq_len < 0 or compress_ratio <= 0 or capacity < 0:
        raise ValueError("invalid QSA score geometry")
    required = max_seq_len // compress_ratio
    if required > capacity:
        raise ValueError("QSA visible groups exceed the allocated cache capacity")
    return next(n for n in qsa_column_buckets(capacity) if n >= required)


@contextmanager
def qsa_capture_columns(columns: int):
    """Only dummy/capture runs may force a width; zero means full capacity."""
    token = _QSA_CAPTURE_COLUMNS.set(columns)
    try:
        yield
    finally:
        _QSA_CAPTURE_COLUMNS.reset(token)


def qsa_graph_columns(metadata) -> int:
    if not isinstance(metadata, dict):
        return 0
    columns = {
        m.score_columns for m in metadata.values()
        if isinstance(m, PreparedQSAForwardMetadata) and m.compress_ratio > 1
        and m.score_columns is not None
    }
    if len(columns) > 1:
        raise ValueError("QSA layers disagree on the score-width graph key")
    return next(iter(columns), 0)


def qsa_cudagraph_key(batch_descriptor, forward_context):
    columns = qsa_graph_columns(forward_context.attn_metadata)
    return (batch_descriptor, columns) if columns else batch_descriptor


@dataclass
class PreparedQSAForwardMetadata(AttentionMetadata):
    """Common per-forward metadata for one QSA side cache."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    num_actual_tokens: int
    storage_block_size: int
    compress_ratio: int
    score_columns: int | None = None

class PreparedQSAMetadataBuilder(AttentionMetadataBuilder[PreparedQSAForwardMetadata]):
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
        self.length_buckets = bool(
            vllm_config.additional_config.get("qwen38_qsa_length_buckets", False)
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self.arange_buffer = torch.arange(max_tokens, dtype=torch.int64, device=device)
        self.slot_mapping_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        self.logical_positions_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> PreparedQSAForwardMetadata:
        del common_prefix_len, fast_build
        num_tokens = common_attn_metadata.num_actual_tokens
        token_to_req_fn = getattr(common_attn_metadata, "token_to_req_indices", None)
        if token_to_req_fn is not None:
            token_to_req = token_to_req_fn(self.token_to_req_buffer)[:num_tokens]
        else:
            # vLLM 0.24 predates the cached convenience method. Reproduce the
            # same device-side mapping without changing CommonAttentionMetadata.
            num_mapped = int(common_attn_metadata.query_start_loc_cpu[-1])
            query_lens = (
                common_attn_metadata.query_start_loc[1:]
                - common_attn_metadata.query_start_loc[:-1]
            )
            mapped = torch.repeat_interleave(
                torch.arange(
                    query_lens.shape[0],
                    dtype=torch.int32,
                    device=self.token_to_req_buffer.device,
                ),
                query_lens,
                output_size=num_mapped,
            )
            self.token_to_req_buffer[:num_mapped].copy_(mapped)
            if num_mapped < num_tokens:
                self.token_to_req_buffer[num_mapped:num_tokens].zero_()
            token_to_req = self.token_to_req_buffer[:num_tokens]
        num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        logical_positions = self.logical_positions_buffer[:num_tokens]
        logical_positions[:num_mapped_tokens].copy_(
            _logical_positions(
                common_attn_metadata.query_start_loc,
                common_attn_metadata.seq_lens,
                token_to_req[:num_mapped_tokens],
                num_mapped_tokens,
                self.arange_buffer,
            )
        )
        if num_mapped_tokens < num_tokens:
            logical_positions[num_mapped_tokens:].fill_(-1)
        if self.compress_ratio == 1:
            slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
        else:
            slot_mapping = compressed_qsa_slot_mapping(
                common_attn_metadata.block_table_tensor,
                token_to_req,
                logical_positions,
                self.storage_block_size,
                self.compress_ratio,
                self.slot_mapping_buffer,
            )
            slot_mapping.masked_fill_(
                common_attn_metadata.slot_mapping[:num_tokens] < 0, -1
            )
        columns = None
        if self.length_buckets and self.compress_ratio > 1:
            capacity = common_attn_metadata.block_table_tensor.shape[1] * self.storage_block_size
            forced = _QSA_CAPTURE_COLUMNS.get()
            if forced is not None:
                columns = capacity if forced == 0 else min(forced, capacity)
            else:
                columns = qsa_score_columns(
                    int(common_attn_metadata.max_seq_len), self.compress_ratio, capacity
                )
        return PreparedQSAForwardMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=slot_mapping,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc,
            token_to_req=token_to_req,
            logical_positions=logical_positions,
            num_actual_tokens=num_tokens,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            score_columns=columns,
        )

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
        return PreparedQSAMetadataBuilder if (getattr(current_platform, "vendor_name", None) == "metax") else QSAMetadataBuilder

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
        # vLLM 0.24 may pad this side cache while unifying per-layer page sizes;
        # the flattened block/row layout below preserves that padded addressing.
        return True

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

    def forward(self) -> None:
        return None

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

    def validate_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Reject an unsupported dtype/layout before any binding mutation.

        The typed key view and the int64 position view both require the last
        dimension to be contiguous; the position view additionally requires the
        BF16 offset to be a whole number of int64 elements *and* every outer
        stride to be view-compatible.  Offset alignment alone is not enough:
        a non-last-dim stride (e.g. shape ``[2, 2, 1, 140]`` with stride
        ``(281, 140, 140, 1)``) still fails ``view(torch.int64)`` later.  The
        preflight therefore constructs the exact position view so a bad layout
        is rejected before the raw cache is assigned.
        """
        if kv_cache.ndim != 4 or kv_cache.shape[2] != 1:
            raise ValueError("QSA raw cache must be [blocks, block_size, 1, width]")
        if kv_cache.dtype != torch.bfloat16 or kv_cache.shape[3] != self.head_size:
            raise ValueError("QSA raw cache does not match its packed BF16 cache spec")
        if kv_cache.stride(-1) != 1:
            raise ValueError(
                "QSA raw cache requires a contiguous last dimension to build "
                f"its typed views (stride(-1)={kv_cache.stride(-1)})"
            )
        if self.cache_rope_positions:
            base = kv_cache.storage_offset() + self.rope_position_offset
            if base % self._BF16_PER_INT64:
                raise ValueError(
                    "QSA position view requires the pack offset to align to "
                    f"int64 ({base} bf16 elements is not a multiple of "
                    f"{self._BF16_PER_INT64})"
                )
            try:
                kv_cache[..., self.rope_position_offset:].view(torch.int64)
            except RuntimeError as exc:
                raise ValueError(
                    "QSA position cache cannot be viewed as int64 from the raw "
                    f"cache (shape={tuple(kv_cache.shape)}, "
                    f"stride={tuple(kv_cache.stride())}): {exc}"
                ) from exc

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.validate_kv_cache(kv_cache)
        super().bind_kv_cache(kv_cache)

    @property
    def key_cache(self) -> torch.Tensor:
        # Legacy direct assignment and the explicit binding hook share one owner.
        return self.kv_cache[..., : self.key_head_size]

    @property
    def rope_position_cache(self) -> torch.Tensor | None:
        if not self.cache_rope_positions:
            return None
        return self.kv_cache[..., self.rope_position_offset :].view(torch.int64)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        if (getattr(current_platform, "vendor_name", None) == "metax"):
            return FullAttentionSpec(
                block_size=self.cache_config.block_size, num_kv_heads=1,
                head_size=self.head_size, head_size_v=0, dtype=self.dtype,
            )
        spec_kwargs = dict(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self.dtype,
        )
        if _BLOCK_STRIDE_SPEC_FLAG:
            # Opt into vLLM 0.24's padded-page contract so this side cache can
            # share a unified page size.  The stride-aware QSA stores/loads and
            # the ``bind_kv_cache`` slice below read it through the strided view.
            spec_kwargs["indexes_kv_by_block_stride"] = True
        return FullAttentionSpec(**spec_kwargs)


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


# Register QSA side caches as KV-cache owners so the common model runner can
# invoke their bind hook without matching on this module's name.
from vllm_fl.compat.kv_cache import register_kv_cache_owner

register_kv_cache_owner(_QSAStateCache)


__all__ = [
    "QSACompressedKeyCache",
    "QSAForwardMetadata",
    "QSAKeyStateCache",
    "QSAMetadataBuilder",
    "QSAStateBackend",
    "canonical_qsa_rope_positions",
    "compressed_qsa_slot_mapping",
]

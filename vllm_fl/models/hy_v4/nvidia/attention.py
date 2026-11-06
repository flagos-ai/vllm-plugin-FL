# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 MLA layers and sink-capable FlashMLA backend for NVIDIA/vLLM 0.28.

The backend keeps the native FLASHMLA_SPARSE name and KV layout. Qualified
prefill uses the native BF16 reader and its constructor-owned buffers; request
lengths use MAX for indexer score capacity and SUM for the flattened KV gather.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd, flash_mla_with_kvcache
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.worker.workspace import current_workspace_manager

from ..config import HYV4Config
from ..fusions import maybe_mrope_direct_cache
from ..indexer_policy import compute_skip_topk_layers
from ..metadata import CPUSequenceLengthsProof, get_current_cpu_seq_lens_proof
from .fusions import GateOverlap
from vllm_fl.models.hy_v4.gated_mla import hpc_gated_mla_gemm, hpc_gated_mla_supported
from vllm_fl.quantization.w8a8.gate_quant_input import maybe_gate_quant_input
from vllm_fl.quantization.w8a8.q_a_rms_quant import (
    configure_q_a_rms_quant,
    maybe_q_a_rms_quant,
)
from vllm_fl.utils import use_flaggems_op

logger = init_logger(__name__)


class HYV4FlashMLASparseImpl(FlashMLASparseImpl):
    """FlashMLA sparse impl that applies HY V4's per-head learnable sink.

    The sink enters as the ``sinks`` impl kwarg of
    `vllm.model_executor.layers.attention.MLAAttention` and is consumed by the
    FlashMLA kernels, which fold it into the softmax denominator:
    ``out *= exp(lse) / (exp(lse) + exp(sink))``.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Indexer | None = None,
        **mla_args,
    ) -> None:
        # ``SparseMLACommonImpl`` takes explicit keyword arguments only, so the
        # sink has to be removed before the base classes see``mla_args``.
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **mla_args,
        )
        self._validate_sinks(sinks, num_heads)
        self.sinks = sinks

    @staticmethod
    def _validate_sinks(sinks: torch.Tensor | None, num_heads: int) -> None:
        """Reject sink tensors the FlashMLA kernels cannot consume.

        Args:
            sinks: Candidate sink tensor, or None when the layer has no sink.
            num_heads: Local (TP-sharded) query head count of this layer.

        Raises:
            ValueError: If the dtype is not float32 or the shape is not
                ``(num_heads,)``.

        """
        if sinks is None:
            return
        if sinks.dtype != torch.float32:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have dtype "
                f"torch.float32, but got {sinks.dtype}."
            )
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have shape "
                f"({num_heads},), but got {tuple(sinks.shape)}."
            )

    def _sinks_for_query(
        self,
        q: torch.Tensor,
        head_dim: int,
        kernel_heads: int,
    ) -> torch.Tensor | None:
        """Return the sink laid out for the kernel's query head count.

        Args:
            q: Query tensor, before any head padding.
            head_dim: Axis of ``q`` holding the query heads.
            kernel_heads: Head count the kernel is invoked with, which may
                exceed the query head count because of padding.

        Returns:
            The sink tensor padded to ``kernel_heads`` with ``-inf`` (a no-op
            sink) for the padded lanes, or None when the layer has no sink.

        Raises:
            ValueError: If the sink and query head layouts disagree, or if they
                live on different devices.

        """
        sinks = self.sinks
        if sinks is None:
            return None

        query_heads = q.shape[head_dim]
        if sinks.shape[0] != query_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sink head count must match the "
                f"runtime query layout: sinks={sinks.shape[0]}, "
                f"query_heads={query_heads}. The sink must use the same "
                "unpadded head layout as the query."
            )
        if sinks.device != q.device:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks and query must be on the "
                f"same device, but got sinks={sinks.device}, query={q.device}."
            )
        if kernel_heads < query_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse kernel head count cannot be smaller than "
                f"the runtime query layout: query_heads={query_heads}, "
                f"kernel_heads={kernel_heads}."
            )
        if kernel_heads == query_heads:
            return sinks

        # Mirror the query padding the kernels require. Reading ``sinks`` here
        # (rather than caching a padded copy at construction time) keeps the
        # values correct for weights loaded after the module is built, and the
        # allocation plus copy are captured in the CUDA graph.
        padded_sinks = sinks.new_full((kernel_heads,), float("-inf"))
        padded_sinks[:query_heads] = sinks
        return padded_sinks

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        logical_query_len = q.size(1)
        padded_num_heads = self.fp8_decode_padded_heads
        # Keep the normal reader's physical query shape consistent with q4
        # verification. Dummy query rows never contribute to returned rows.
        align_single_query = (
            self.kv_cache_dtype == "fp8_ds_mla"
            and q.shape == (1, 1, 4, 576)
            and q.dtype == torch.bfloat16
            and topk_indices.shape == (1, 1, 2048)
            and topk_indices.dtype == torch.int32
            and self.softmax_scale == 1 / 16
            and current_platform.is_device_capability(90)
        )
        physical_query_len = 4 if align_single_query else logical_query_len
        attn_sink = self._sinks_for_query(q, head_dim=2, kernel_heads=padded_num_heads)

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros(
                (q.size(0), physical_query_len, padded_num_heads, q.size(3))
            )
            q_padded[:, :logical_query_len, :actual_num_heads, :] = q
            q = q_padded
        if align_single_query:
            # The three zero-query rows use the same legal keys as q0. No
            # future query or KV entry is needed to align the physical shape.
            topk_indices = topk_indices.expand(-1, physical_query_len, -1).contiguous()

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
            attn_sink=attn_sink,
        )

        if align_single_query:
            out = out[:, :logical_query_len, :, :]
            lse = lse[:, :, :logical_query_len]

        # Slice output and lse back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]
            lse = lse[:, :actual_num_heads, :]

        return out, lse

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
        actual_num_heads: int | None = None,
        out: torch.Tensor | None = None,
        *,
        _prepared_q_sink: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of 64.
        # Pad from q's head count, not self.num_heads: under DCP the heads are
        # all-gathered before this.
        if actual_num_heads is None:
            actual_num_heads = q.shape[1]
        if (
            use_flaggems_op("pad_mla_q_sink_slot_bound", default=False)
            and self.kv_cache_dtype == "fp8_ds_mla"
            and self.num_heads == 4
            and self.prefill_padding == 64
            and current_platform.is_device_capability(90)
            and actual_num_heads == 4
            and 1 <= num_tokens <= 2048
            and q.shape == (num_tokens, 4, 576)
            and q.dtype == torch.bfloat16
            and q.stride() == (2304, 576, 1)
            and self.softmax_scale == 1 / 16
            and num_tokens
            <= kv_c_and_k_pe_cache.shape[0]
            <= self.prefill_bf16_workspace.shape[0]
            and kv_c_and_k_pe_cache.shape[1:] == (1, 576)
            and kv_c_and_k_pe_cache.dtype == torch.bfloat16
            and kv_c_and_k_pe_cache.stride() == (576, 576, 1)
            and topk_indices.shape == (num_tokens, 2048)
            and topk_indices.dtype == torch.int32
            and topk_indices.stride() == (2048, 1)
            and q.device == kv_c_and_k_pe_cache.device == topk_indices.device
            and self.sinks is not None
            and self.sinks.dtype == torch.float32
            and self.sinks.shape == (4,)
            and self.sinks.stride() == (1,)
            and self.sinks.device == q.device
            and (
                topk_length is None
                or (
                    topk_length.shape == (num_tokens,)
                    and topk_length.dtype == torch.int32
                    and topk_length.stride() == (1,)
                    and topk_length.device == q.device
                )
            )
            and out is None
            and _prepared_q_sink is None
        ):
            q_padded = torch.empty(
                (num_tokens, 64, 576), dtype=q.dtype, device=q.device
            )
            sink_padded = torch.empty((64,), dtype=self.sinks.dtype, device=q.device)
            slot_bound = torch.empty((num_tokens,), dtype=torch.int32, device=q.device)
            from flaggems_vllm import pad_mla_q_sink_slot_bound

            pad_mla_q_sink_slot_bound(
                q, self.sinks, topk_indices, q_padded, sink_padded, slot_bound
            )
            topk_length = slot_bound
            _prepared_q_sink = (q_padded, sink_padded)
        padded_num_heads = (
            (actual_num_heads + self.prefill_padding - 1)
            // self.prefill_padding
            * self.prefill_padding
        )
        # q_concat_buffer may already include the kernel-required padding.
        # The sink remains defined only for the real query heads.
        prepared_q_sink = _prepared_q_sink
        if prepared_q_sink is None:
            attn_sink = self._sinks_for_query(
                q[:, :actual_num_heads], head_dim=1, kernel_heads=padded_num_heads
            )

            if q.shape[1] < padded_num_heads:
                logger.warning_once(
                    f"Padding num_heads from {actual_num_heads} to "
                    f"{padded_num_heads} for BF16 sparse prefill kernel"
                )
                # Zero (not new_empty) the padded lanes: topk_indices is shared by
                # all heads, so the kernel reduces across the head group and NaNs
                # from uninitialized memory would leak into the real heads.
                if out is None:
                    q_padded = q.new_zeros((q.shape[0], padded_num_heads, q.shape[2]))
                else:
                    *_, q_padded, _, _ = current_workspace_manager().get_simultaneous(
                        *self.workspace_specs
                    )
                    q_padded = q_padded[:num_tokens]
                    q_padded[:, actual_num_heads:].zero_()
                q_padded[:, :actual_num_heads, :] = q
                q = q_padded

        else:
            q, attn_sink = prepared_q_sink

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output, _, lse = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
        )

        output = output[:, :actual_num_heads, :]
        lse = lse[:, :actual_num_heads]
        # vLLM 0.28 consumes only the BF16 attention output here.
        return output


class HYV4FlashMLASparseMetadataBuilder(FlashMLASparseMetadataBuilder):
    """Compose native metadata for qualified Hy4-preview sparse prefill."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        parallel = vllm_config.parallel_config
        dims = self.mla_dims
        self._fp8_bf16_prefill_supported = (
            self.use_fp8_kv_cache
            and self.num_heads == 4
            and self.topk_tokens == 2048
            and self.model_config.dtype == torch.bfloat16
            and kv_cache_spec.block_size == 64
            and (
                dims.kv_lora_rank,
                dims.qk_nope_head_dim,
                dims.qk_rope_head_dim,
                dims.v_head_dim,
            )
            == (512, 192, 64, 256)
            and current_platform.is_device_capability(90)
            and not self.use_pcp
            and self.dcp_world_size == 1
            and not parallel.use_ubatching
            and vllm_config.lora_config is None
            and vllm_config.attention_config.sparse_mla_force_mqa
            and bool(layer_names)
            and all(
                self._supports_fp8_bf16_prefill_layer(
                    vllm_config.compilation_config.static_forward_context[name]
                )
                for name in layer_names
            )
        )
        # Native constructor-owned buffers: no model-context identity check.
        self._fp8_bf16_prefill_query_capacity = (
            min(
                vllm_config.compilation_config.static_forward_context[
                    name
                ].impl.q_concat_buffer.shape[0]
                for name in layer_names
            )
            if self._fp8_bf16_prefill_supported
            else 0
        )
        self._fp8_bf16_prefill_workspace_capacity = (
            min(
                vllm_config.compilation_config.static_forward_context[
                    name
                ].impl.prefill_bf16_workspace.shape[0]
                for name in layer_names
            )
            if self._fp8_bf16_prefill_supported
            else 0
        )
        self._fp8_bf16_prefill_max_context = (
            min(
                self.model_config.max_model_len,
                self._fp8_bf16_prefill_workspace_capacity,
            )
            if self._fp8_bf16_prefill_supported
            else 0
        )

    @staticmethod
    def _supports_fp8_bf16_prefill_layer(layer):
        impl = layer.impl
        return (
            isinstance(impl, HYV4FlashMLASparseImpl)
            and impl.num_heads == 4
            and impl.kv_cache_dtype == "fp8_ds_mla"
            and impl.softmax_scale == 1 / 16
            and impl.prefill_padding == 64
            and impl.sinks is not None
            and impl.sinks.dtype == torch.float32
            and impl.sinks.shape == (4,)
            and impl.sinks.stride() == (1,)
            and impl.prefill_bf16_workspace.dtype == torch.bfloat16
            and impl.prefill_bf16_workspace.ndim == 2
            and impl.prefill_bf16_workspace.shape[0] >= 1024
            and impl.prefill_bf16_workspace.shape[1] == 576
            and impl.prefill_bf16_workspace.stride() == (576, 1)
            and impl.q_concat_buffer.dtype == torch.bfloat16
            and impl.q_concat_buffer.shape[0] >= 1024
            and impl.q_concat_buffer.shape[1:] == (4, 576)
            and impl.q_concat_buffer.stride() == (2304, 576, 1)
            and impl.sinks.device == impl.q_concat_buffer.device
            and impl.prefill_bf16_workspace.device == impl.q_concat_buffer.device
        )

    def _use_fp8_bf16_mixed_prefill(self, common_prefix_len, common, metadata):
        """Use the native contiguous prefill reader when its real buffers fit.

        The native q4 decode prefix remains unchanged. Only prefill lengths
        proven from this input step's immutable packet may plan the gather.
        Request-local score capacity and concatenated gather capacity differ.
        """
        num_decodes = metadata.num_decodes
        num_prefills = metadata.num_prefills
        if not (
            self._fp8_bf16_prefill_supported
            and use_flaggems_op("pad_mla_q_sink_slot_bound", default=False)
            and common_prefix_len == 0
            and num_decodes > 0
            and num_prefills > 0
            and common.num_reqs == num_decodes + num_prefills
            and metadata.num_decode_tokens == 4 * num_decodes
            and metadata.num_decode_tokens
            < common.num_actual_tokens
            <= self._fp8_bf16_prefill_query_capacity
            and common.causal is True
            and common.mm_req_doc_ranges is None
            and metadata.fp8_use_mixed_batch
            and isinstance(
                metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
            )
            and (metadata.prefill is None or not metadata.prefill.use_dense_mha)
        ):
            return False
        proof = common.cpu_seq_lens_proof
        if (
            not isinstance(proof, CPUSequenceLengthsProof)
            or proof is not get_current_cpu_seq_lens_proof()
        ):
            return False
        starts, lengths = common.query_start_loc_cpu, common.seq_lens_cpu_upper_bound
        if (
            lengths is None
            or common.is_prefilling is None
            or common.is_prefilling.device.type != "cpu"
            or starts.device.type != "cpu"
            or lengths.device.type != "cpu"
            or tuple(starts.shape) != (common.num_reqs + 1,)
            or tuple(lengths.shape) != (common.num_reqs,)
        ):
            return False
        query_starts = tuple(int(x) for x in starts)
        if (
            query_starts != proof.query_starts
            or query_starts[: num_decodes + 1]
            != tuple(range(0, 4 * num_decodes + 1, 4))
            or query_starts[-1] != common.num_actual_tokens
        ):
            return False
        prefill_lengths = []
        for i in range(num_decodes, common.num_reqs):
            q_len = query_starts[i + 1] - query_starts[i]
            prev, draft = proof.previous_positions[i], proof.previous_drafts[i]
            bound = proof.computed[i] + q_len
            if (
                q_len <= 0
                or not bool(common.is_prefilling[i])
                or proof.computed[i] < 0
                or not proof.computed[i] < proof.prompt_lengths[i]
                or not ((prev == -1 and draft == -1) or (prev >= 0 and draft == 0))
                or int(lengths[i]) != bound
                or not q_len <= bound <= self.model_config.max_model_len
            ):
                return False
            prefill_lengths.append(bound)
        # This is the concatenated native BF16 gather, so it needs SUM, not MAX.
        # Both capacities were read from the constructor-owned layer buffers,
        # before capture. No new storage is allocated or rebound in this selector.
        return sum(prefill_lengths) <= self._fp8_bf16_prefill_workspace_capacity

    def _use_fp8_bf16_prefill(self, common_prefix_len, common, metadata):
        if self._use_fp8_bf16_mixed_prefill(common_prefix_len, common, metadata):
            return True
        if not (
            self._fp8_bf16_prefill_supported
            and use_flaggems_op("pad_mla_q_sink_slot_bound", default=False)
            and common_prefix_len == 0
            and common.num_reqs == 1
            and common.num_actual_tokens == 1024
            and common.max_query_len == 1024
            and 1024 <= common.max_seq_len <= self._fp8_bf16_prefill_max_context
            and common.causal is True
            and common.mm_req_doc_ranges is None
            and metadata.num_prefills == 1
            and metadata.num_decodes == 0
            and metadata.num_decode_tokens == 0
            and metadata.fp8_use_mixed_batch
            and isinstance(
                metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
            )
            and (metadata.prefill is None or not metadata.prefill.use_dense_mha)
        ):
            return False
        query_starts = common.query_start_loc_cpu
        seq_lens = common.seq_lens_cpu_upper_bound
        if (
            seq_lens is None
            or query_starts.device.type != "cpu"
            or seq_lens.device.type != "cpu"
            or query_starts.shape != (2,)
            or seq_lens.shape != (1,)
        ):
            return False
        # Prefill CPU upper bounds are exact. No device-value read or sync.
        return query_starts.tolist() == [0, 1024] and seq_lens.tolist() == [
            common.max_seq_len
        ]

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        from ..runtime import bind_common_runtime_metadata

        bind_common_runtime_metadata(common_attn_metadata)
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if self._use_fp8_bf16_prefill(
            common_prefix_len, common_attn_metadata, metadata
        ):
            extra_metadata = self._build_fp8_separate_prefill_decode(
                common_attn_metadata, metadata
            )
            metadata.fp8_extra_metadata = extra_metadata
            metadata.fp8_use_mixed_batch = False
        return metadata


class HYV4FlashMLASparseBackend(FlashMLASparseBackend):
    """``FLASHMLA_SPARSE`` with attention-sink support for HY V4.

    Keeps the parent's name and composes existing native metadata APIs.
    """

    @staticmethod
    def get_builder_cls() -> type[HYV4FlashMLASparseMetadataBuilder]:
        return HYV4FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[HYV4FlashMLASparseImpl]:
        return HYV4FlashMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True


_SPARSE_LAYER_TYPES = (
    "sparse_attention",
    "sparse",
    "indexed_attention",
    "deepseek_sparse_attention",
)


class Indexer(nn.Module):
    """Lightning indexer selecting the top-k tokens for sparse MLA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: HYV4Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # No tensor parallelism, just replicated.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcast to BF16 while loading to keep the fusion.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6).to(dtype=torch.float32)
        self.softmax_scale = self.head_dim**-0.5

        from ..indexer_policy import use_int8_indexer

        self.use_int8 = use_int8_indexer()
        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        # FP8 naive cache: values in fp8 plus one fp32 scale per
        # ``quant_block_size`` elements.
        assert cache_config is not None, "HYV4 indexer requires cache_config"
        cache_cls = DeepseekV32IndexerCache
        if self.use_int8:
            from .indexer import HYV4Int8IndexerCache

            cache_cls = HYV4Int8IndexerCache
        self.k_cache = cache_cls(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        if self.use_int8:
            from .indexer import HYV4Int8Indexer

            self.indexer_op = HYV4Int8Indexer(
                self.k_cache,
                self.topk_tokens,
                self.max_model_len,
                self.topk_indices_buffer,
                vllm_config.parallel_config,
            )
        else:
            self.indexer_op = SparseAttnIndexer(
                self.k_cache,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        hidden_states, q_quant, k, weights = self.prepare_inputs(
            hidden_states, qr, positions, rotary_emb
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the quantized query, key and per-head weights of the indexer."""
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        # Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims.
        q_nope, q_pe = torch.split(
            q, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_nope, k_pe = torch.split(
            k, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        if positions.ndim == 2:
            direct_rope = maybe_mrope_direct_cache(
                rotary_emb, positions, q_pe, k_pe.unsqueeze(1)
            )
            if direct_rope is None:
                q_pe, k_pe = rotary_emb(positions, q_pe.reshape(q.shape[0], -1), k_pe)
            else:
                q_pe, k_pe = direct_rope
        else:
            q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # Reassemble with the original physical layout: no_pe first, pe last.
        q = torch.cat([q_nope, q_pe], dim=-1)
        # ``k_pe`` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_nope, k_pe.squeeze(-2)], dim=-1)

        # Only q is quantized here; k quantization is fused with cache insertion.
        q = q.view(-1, self.head_dim)
        if self.use_int8:
            from flaggems_vllm import int8_indexer_quantize

            q_fp8, q_scale = int8_indexer_quantize(q)
        else:
            q_fp8, q_scale = per_token_group_quant_fp8(
                q,
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=self.scale_fmt is not None,
            )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        return hidden_states, q_fp8, k, weights


class HYV4MLAAttention(nn.Module):
    """Multi-head latent attention with optional sparse lightning indexer.

    Main reference: the DeepSeek-V2 paper and the FlashInfer implementation
    (https://arxiv.org/abs/2405.04434). HY V4 additionally supports an output
    gate (``gated_mla``) and a per-head learnable attention sink.

    The sink is applied by binding the sink-capable backend from
    this module; if no backend on this platform can consume sinks, the
    initialization fails to preserve the trained architecture.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: HYV4Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.layer_id = int(prefix.split(".")[-2])
        layer_types = config.layer_types
        requested_sparse = (
            layer_types is not None
            and self.layer_id < len(layer_types)
            and layer_types[self.layer_id] in _SPARSE_LAYER_TYPES
        )
        # Only actual sparse layers may share another layer's top-k indices.
        self.skip_topk = requested_sparse and self.layer_id in compute_skip_topk_layers(
            config
        )
        # The skip pattern only governs backbone layers. MTP/nextn layers
        # (layer_id >= num_hidden_layers) always build a full indexer: they
        # compute indices at draft step 0 and toggle at runtime.
        is_mtp_layer = self.layer_id >= config.num_hidden_layers
        self.create_indexer = requested_sparse and (not self.skip_topk or is_mtp_layer)
        self.is_sparse = requested_sparse

        # Do not silently degrade sparse layers into dense attention. Probe the
        # sparse MLA backend directly and fail fast with the real error.
        kv_cache_dtype = cache_config.cache_dtype if cache_config else "auto"
        if self.is_sparse:
            get_attn_backend(
                head_size=self.kv_lora_rank + self.qk_rope_head_dim,
                dtype=torch.get_default_dtype(),
                kv_cache_dtype=kv_cache_dtype,
                use_mla=True,
                has_sink=False,
                use_sparse=True,
                num_heads=self.num_local_heads,
            )

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.fused_qkv_a_proj = None
        self.kv_a_proj_with_mqa = None
        if self.q_lora_rank is not None:
            # ``q_a_proj`` and ``kv_a_proj_with_mqa`` read the same
            # ``hidden_states`` and are both TP-replicated, so they run as one
            # GEMM. The checkpoint keeps them separate; `load_weights` merges
            # them through ``stacked_params_mapping``.
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        self.q_a_layernorm = None
        self.q_b_proj = None
        self.q_proj = None
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        # create_weights has selected the native q_b scheme; its original
        # loading/postprocess still runs after this one-time optional selection.
        self._hy4_q_a_rms_quant_policy = configure_q_a_rms_quant(
            self.q_a_layernorm, self.q_b_proj
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        self.indexer_rope_emb: nn.Module | None
        self.indexer: Indexer | None
        if self.create_indexer:
            # The checkpoint stores indexer q_pe/k_pe in interleaved
            # (Megatron/PTM) layout, so the indexer must use interleaved RoPE
            # (is_neox_style=False) like the main attention path. Using NeoX
            # here loses the relative-position dependence and corrupts the DSA
            # top-k selection.
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )
            # The indexer projects its queries from the MLA q_lora activations,
            # so a sparse layer requires a query down-projection.
            assert q_lora_rank is not None, (
                "HYV4 sparse attention requires q_lora_rank to be set"
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        self.gated_mla = config.gated_mla
        self.linear_gate: ColumnParallelLinear | None
        if self.gated_mla:
            if config.gating_type == "headwise":
                self.gate_projection_size_per_head = 1
            elif config.gating_type == "elementwise":
                self.gate_projection_size_per_head = self.v_head_dim
            else:
                raise ValueError(f"Unknown gating type: {config.gating_type}")
            self.linear_gate = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.gate_projection_size_per_head,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_gate",
            )
            self.use_hpc_gated_mla = hpc_gated_mla_supported(
                config.gating_type, self.linear_gate
            )
        else:
            self.linear_gate = None
            self.use_hpc_gated_mla = False
        self._hy4_gate_overlap = GateOverlap() if self.gated_mla else None
        self.prefix = prefix

        # Per-head learnable attention sink. Created BEFORE ``MLAAttention`` so
        # it can be forwarded as the ``sinks`` impl kwarg. The parameter always
        # holds the local TP shard.
        self.learnable_sink = config.learnable_sink
        sinks = None
        sink_backend: type[AttentionBackend] | None = None
        if self.learnable_sink:
            sink_backend = self._resolve_sink_backend(kv_cache_dtype)
            self.learnable_sink_param = nn.Parameter(
                torch.empty(
                    self.num_local_heads,
                    dtype=torch.float32,
                )
            )
            sinks = self.learnable_sink_param
            self._force_sparse_mqa()

        extra_impl_args = {} if sinks is None else {"sinks": sinks}
        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
            attn_backend=sink_backend,
            **extra_impl_args,
        )

    @property
    def topk_indices_buffer(self) -> torch.Tensor | None:
        """The sparse top-k index buffer this layer reads from.

        The real consumer is ``mla_attn.impl``, an ``AttentionImpl`` rather than
        an ``nn.Module``, so a plain ``named_modules()`` walk cannot reach it.
        The MTP proposer rebinds the draft layers onto the target model's buffer
        through such a walk, so expose it here: without this the draft attention
        would keep reading its own buffer while the indices it needs are written
        to the target's.
        """
        return self.mla_attn.impl.topk_indices_buffer if self.is_sparse else None

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, buffer: torch.Tensor) -> None:
        self.mla_attn.impl.topk_indices_buffer = buffer  # type: ignore[attr-defined]

    def _resolve_sink_backend(self, kv_cache_dtype: str) -> type[AttentionBackend]:
        """Return an MLA backend that can apply this layer's learnable sink.

        The sink is part of the architecture, so a backend that cannot apply it
        changes the model's output. Resolution order:

        1. If the backend the selector would pick already advertises
           `supports_sink`, keep it — this also honours an explicit
           ``--attention-backend`` choice.
        2. Otherwise fall back to the sink-capable ``FLASHMLA_SPARSE`` subclass
           in this module, whose kernels accept ``attn_sink``, provided
           it validates against the current runtime configuration.
        3. Otherwise fail initialization; silently ignoring sinks changes the model.

        Args:
            kv_cache_dtype: The layer's KV cache dtype string.

        Returns:
            The sink-capable backend class to bind.

        """
        head_size = self.kv_lora_rank + self.qk_rope_head_dim
        dtype = torch.get_default_dtype()
        selected_cls = get_attn_backend(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            use_mla=True,
            use_sparse=self.is_sparse,
            num_heads=self.num_local_heads,
        )

        if selected_cls.supports_sink():
            return selected_cls

        # Mirror how the selector derives the configuration-dependent inputs so
        # this check accepts exactly what the backend would accept at runtime.
        capability = current_platform.get_device_capability()
        if capability is None:
            raise RuntimeError(
                "HYV4 cannot validate attention sinks without device capability"
            )
        cache_config = get_current_vllm_config().cache_config
        block_size = (
            cache_config.block_size
            if cache_config is not None and cache_config.user_specified_block_size
            else None
        )
        invalid_reasons = HYV4FlashMLASparseBackend.validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=cast(CacheDType, kv_cache_dtype),
            block_size=block_size,
            use_mla=True,
            has_sink=True,
            use_sparse=self.is_sparse,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=capability,
            attn_type=AttentionType.DECODER,
        )
        if invalid_reasons:
            raise RuntimeError(
                "HYV4 sink-capable sparse attention is unavailable: "
                + ", ".join(invalid_reasons)
            )

        logger.info_once(
            "HYV4 learnable sink enabled: using the sink-capable "
            "FLASHMLA_SPARSE impl instead of %s, which cannot apply sinks.",
            selected_cls.get_name(),
        )
        return HYV4FlashMLASparseBackend

    def _force_sparse_mqa(self) -> None:
        """Keep every token on the sink-capable sparse MQA path.

        ``_resolve_sink_backend`` only binds the backend that serves decode.
        ``MLAAttention`` additionally routes short prefills to a separate dense
        MLA prefill backend, and none of those accept ``attn_sink``, so prefill
        would silently drop the sink while decode applies it. Such a partially
        applied sink is not the trained architecture and corrupts the output, so
        opt out of the dense split instead.

        Prefills up to ``index_topk`` keep every token inside the sparse top-k,
        making the sparse path numerically equivalent to the dense one apart from
        also applying the sink.
        """
        attention_config = get_current_vllm_config().attention_config
        if attention_config.sparse_mla_force_mqa:
            return
        attention_config.sparse_mla_force_mqa = True
        logger.info_once(
            "HYV4 learnable sink enabled: forcing sparse MQA for prefill too, "
            "as the dense MLA prefill backends cannot apply sinks."
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            fused_q_a = maybe_q_a_rms_quant(
                self._hy4_q_a_rms_quant_policy,
                self.q_a_layernorm,
                self.q_b_proj,
                q_c,
            )
            if fused_q_a is None:
                q_c = self.q_a_layernorm(q_c)
                q_b_input = q_c
            else:
                # Keep the original BF16 q_c for indexer.wq_b. Only q_b sees
                # the standard QuantizedActivation, consumed immediately.
                q_c, q_b_input = fused_q_a
            q = self.q_b_proj(q_b_input)[0]
            # Match native temporary lifetime before the eager attention break.
            del q_b_input, fused_q_a
        else:
            assert self.q_proj is not None
            assert self.kv_a_proj_with_mqa is not None
            q = self.q_proj(hidden_states)[0]
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        if positions.ndim == 2:
            direct_rope = maybe_mrope_direct_cache(
                self.rotary_emb,
                positions,
                q[..., self.qk_nope_head_dim :],
                k_pe.unsqueeze(1),
            )
            if direct_rope is None:
                q_pe, k_pe = self.rotary_emb(
                    positions,
                    q[..., self.qk_nope_head_dim :].reshape(q.shape[0], -1),
                    k_pe,
                )
            else:
                q_pe, k_pe = direct_rope
            q[..., self.qk_nope_head_dim :] = q_pe.reshape(
                q.shape[0], self.num_local_heads, self.qk_rope_head_dim
            )
            k_pe = k_pe.reshape(-1, 1, self.qk_rope_head_dim)
        else:
            k_pe = k_pe.unsqueeze(1)
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        output_shape = (
            hidden_states.shape[0],
            self.num_local_heads * self.v_head_dim,
        )
        # Single coarse eager break covering the indexer and MLA attention, as
        # the breakable cudagraph contract requires: everything that reads
        # per-batch metadata runs in one eager segment, so no tensor has to stay
        # alive across a capture-segment boundary.
        attn_out = torch.empty(
            output_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        gate_ticket = (
            None
            if self._hy4_gate_overlap is None
            else self._hy4_gate_overlap.begin(self, hidden_states)
        )
        gate_factor = None
        try:
            self._indexer_and_attn(
                hidden_states, q_c, positions, q, kv_c_normed, k_pe, attn_out
            )
        finally:
            if gate_ticket is not None:
                gate_factor = gate_ticket.finish()

        if self.gated_mla and self.linear_gate is not None:
            if gate_factor is not None:
                quantized = maybe_gate_quant_input(self.o_proj, attn_out, gate_factor)
                attn_out = attn_out * gate_factor if quantized is None else quantized
            elif self.use_hpc_gated_mla:
                # Projection, sigmoid and the product in one launch. The gate
                # is column-parallel and unbiased, so its local weight shard
                # maps straight onto the local attn_out columns.
                assert hidden_states.is_contiguous() and attn_out.is_contiguous()
                attn_out = hpc_gated_mla_gemm(
                    hidden_states,
                    self.linear_gate.weight,
                    attn_out,
                )
            else:
                gate_score = self.linear_gate(hidden_states)[0]
                if self.config.gating_type == "headwise":
                    gate_score = gate_score.unsqueeze(-1)
                    attn_out = attn_out.reshape(
                        *attn_out.shape[:-1], -1, self.v_head_dim
                    )
                    attn_out = attn_out * torch.sigmoid(gate_score)
                    attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)
                else:
                    gate_factor = torch.sigmoid(gate_score)
                    quantized = maybe_gate_quant_input(
                        self.o_proj, attn_out, gate_factor
                    )
                    attn_out = (
                        attn_out * gate_factor if quantized is None else quantized
                    )

        out, _ = self.o_proj(attn_out)
        return out

    @eager_break_during_capture
    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | None,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, heads * v_head_dim], written in place
    ) -> None:
        """Run the lightning indexer and MLA attention in one eager segment.

        Both read per-batch attention metadata, so under the breakable cudagraph
        they must not be captured. Keeping them in a single break (instead of one
        break each) also means the attention inputs never have to survive a
        capture-segment boundary. The nested ``sparse_attn_indexer`` and
        ``unified_mla_attention_with_output`` breaks short-circuit here, since
        the capture is no longer active inside an eager segment.
        """
        if self.indexer is not None and self.is_sparse and not self.skip_topk:
            from vllm.forward_context import get_forward_context

            from .indexer import score_width_attention_scope

            with score_width_attention_scope(get_forward_context()):
                self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        out.copy_(self.mla_attn(q, kv_c_normed, k_pe, output_shape=out.shape))

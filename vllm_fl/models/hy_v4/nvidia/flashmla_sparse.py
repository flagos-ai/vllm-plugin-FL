# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink-capable FlashMLA sparse backend for HY V4 (NVIDIA).

HY V4 adds a per-head learnable attention sink on top of sparse MLA. The
vendored FlashMLA kernels already accept an ``attn_sink`` argument, but vLLM's
shared ``FLASHMLA_SPARSE`` backend neither advertises sink support nor forwards
the tensor, so the bias would be silently dropped.

This module supplies the missing wiring inside the model package, mirroring how
`vllm.models.deepseek_v4.nvidia.flashinfer_sparse` hands ``sinks`` to the
FlashInfer sparse MLA kernels: subclass the platform backend, declare
`supports_sink`, and thread the sink into every kernel call.

The subclass intentionally keeps the inherited ``get_name()``
(``"FLASHMLA_SPARSE"``). Several shared code paths key off that exact string —
`_canonicalize_sparse_mla_kv_cache_dtype` promotes a quantized KV cache to
``fp8_ds_mla`` for it, and `FlashMLASparseImpl` asserts that layout — so a new
name would silently change KV cache behaviour. Sink support and kernel wrappers
remain unchanged; qualified prefill composes the native builder metadata.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_fl.utils import use_flaggems_op

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


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
        indexer: "Indexer | None" = None,
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
                *(
                    vllm_config.compilation_config.static_forward_context[
                        name
                    ].impl.prefill_bf16_workspace.shape[0]
                    for name in layer_names
                ),
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
        from vllm.v1.attention import backend

        proof_type = getattr(backend, "CPUSequenceLengthsProof", None)
        current_proof = getattr(backend, "get_current_cpu_seq_lens_proof", None)
        proof = getattr(common, "cpu_seq_lens_proof", None)
        if (
            not isinstance(proof_type, type)
            or not callable(current_proof)
            or not isinstance(proof, proof_type)
            or proof is not current_proof()
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
        seq_lens = getattr(common, "seq_lens_cpu_upper_bound", None)
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

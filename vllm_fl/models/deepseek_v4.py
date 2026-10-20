# SPDX-License-Identifier: Apache-2.0
"""ModelRegistry thin model adding W8A8 DSV4 output projection support."""

from __future__ import annotations

from itertools import islice
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CompilationMode, VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import PPMissingLayer, make_layers
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV4MoE,
    _select_dsv4_attn_cls,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.multi_stream_utils import execute_in_parallel, maybe_execute_in_parallel
from vllm.utils.torch_utils import direct_register_custom_op, vllm_lib
from vllm.v1.worker.workspace import WorkspaceManager, current_workspace_manager

from vllm_fl.ops import mhc_int8_tilelang as _mhc_int8_tilelang  # noqa: F401
from vllm_fl.ops.deepseek_v4 import (
    combine_topk_swa_indices,
    compress_int8_indexer_k_cache,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    fused_indexer_q_rope_quant,
    fused_indexer_q_rope_quant_int8,
    fused_q_kv_rmsnorm,
    hc_head,
    int8_scaled_mm,
    inv_rope_quant_fp8,
    mhc_fused_post_pre,
    mhc_fused_post_pre_int8,
    mhc_post,
    mhc_pre,
    mhc_pre_int8,
    qnorm_rope_kv_bf16_insert,
    qnorm_rope_kv_fp8_insert,
    qnorm_rope_kv_quant_insert,
)
from vllm_fl.ops.deepseek_v4_int8_woa import fused_inv_rope_quant_int8
from vllm_fl.ops.sparse_attn_indexer import SparseAttnIndexerFL


def _patch_hopper_fp8_inv_rope_kernel() -> None:
    """Bridge the vLLM/Triton PDL argument mismatch on pre-SM100 GPUs."""
    capability = current_platform.get_device_capability()
    if capability is None or capability.major >= 10:
        return
    vllm_lib.impl(
        "fused_inv_rope_fp8_quant_kernel",
        inv_rope_quant_fp8,
        dispatch_key=current_platform.dispatch_key,
        allow_override=True,
    )


_patch_hopper_fp8_inv_rope_kernel()


_WORKSPACE_ALIGNMENT = 256


def _workspace_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    size = prod(shape) * dtype.itemsize
    return (
        (size + _WORKSPACE_ALIGNMENT - 1) // _WORKSPACE_ALIGNMENT * _WORKSPACE_ALIGNMENT
    )


def _workspace_views(
    workspace: torch.Tensor,
    *specs: tuple[tuple[int, ...], torch.dtype],
) -> list[torch.Tensor]:
    views = []
    offset = 0
    for shape, dtype in specs:
        actual_bytes = prod(shape) * dtype.itemsize
        views.append(
            workspace[offset : offset + actual_bytes].view(dtype).reshape(shape)
        )
        offset += _workspace_nbytes(shape, dtype)
    assert offset <= workspace.numel()
    return views


class _DeepseekV4AttentionIOWorkspace:
    """Graph-stable decode storage and lifecycle-packed prefill arena.

    Large prefill Q stays live while the indexer runs. Once the indexer has
    completed, its scratch is dead and the same tail is split into O and the
    FlashMLA gather buffer. This keeps one contiguous allocation for every
    prefill attention temporary and avoids allocator fragmentation.
    """

    def __init__(
        self,
        *,
        decode_capacity: int,
        padded_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        num_ubatches: int,
    ) -> None:
        self.decode_capacity = decode_capacity
        self.padded_heads = padded_heads
        self.head_dim = head_dim
        self.dtype = dtype
        # ParallelConfig uses 0 to represent disabled u-batching, whereas
        # WorkspaceManager always needs at least the main-thread slot.
        workspace_slots = max(1, num_ubatches)
        self.decode = WorkspaceManager(device, workspace_slots)
        self.prefill = WorkspaceManager(device, workspace_slots)

    def get(
        self,
        num_tokens: int,
        prefill_tail_bytes: int = 0,
        prefill_gather_bytes: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        is_decode_shape = num_tokens <= self.decode_capacity
        if is_decode_shape:
            shape = (self.decode_capacity, self.padded_heads, self.head_dim)
            q_padded, o_padded = self.decode.get_simultaneous(
                (shape, self.dtype),
                (shape, self.dtype),
            )
            empty = q_padded.new_empty((0,), dtype=torch.uint8)
            return q_padded[:num_tokens], o_padded[:num_tokens], empty, empty

        shape = (num_tokens, self.padded_heads, self.head_dim)
        output_bytes = _workspace_nbytes(shape, self.dtype)
        assert prefill_tail_bytes >= output_bytes
        q_padded, tail = self.prefill.get_simultaneous(
            (shape, self.dtype),
            ((prefill_tail_bytes,), torch.uint8),
        )
        o_padded, gather_workspace = _workspace_views(
            tail,
            (shape, self.dtype),
            ((prefill_gather_bytes,), torch.uint8),
        )
        return q_padded, o_padded, tail, gather_workspace


def _deepseek_v4_fl_attention_workspace(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
    qr_q: torch.Tensor | None,
    qr_scale: torch.Tensor | None,
    positions: torch.Tensor,
    padded_heads: int,
    n_local_heads: int,
    head_dim: int,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    num_tokens = hidden_states.shape[0]
    tail_bytes, gather_bytes = layer._fl_prefill_workspace_bytes(num_tokens)
    q_padded, out, prefill_workspace, gather_workspace = (
        layer._fl_attention_io_workspace.get(num_tokens, tail_bytes, gather_bytes)
    )
    layer.attention_impl(
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        qr_q,
        qr_scale,
        positions,
        out,
        q_padded,
        prefill_workspace,
        gather_workspace,
    )
    return out[:, :n_local_heads, :]


def _deepseek_v4_fl_attention_workspace_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
    qr_q: torch.Tensor | None,
    qr_scale: torch.Tensor | None,
    positions: torch.Tensor,
    padded_heads: int,
    n_local_heads: int,
    head_dim: int,
    layer_name: str,
) -> torch.Tensor:
    result = hidden_states.new_empty((hidden_states.shape[0], n_local_heads, head_dim))
    del (
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        qr_q,
        qr_scale,
        positions,
        padded_heads,
        n_local_heads,
        head_dim,
        layer_name,
    )
    return result


direct_register_custom_op(
    op_name="deepseek_v4_fl_attention_workspace",
    op_func=_deepseek_v4_fl_attention_workspace,
    mutates_args=[],
    fake_impl=_deepseek_v4_fl_attention_workspace_fake,
)


def _deepseek_v4_fl_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
    qr_q: torch.Tensor | None,
    qr_scale: torch.Tensor | None,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    """Compile-safe attention entrypoint with an explicit output buffer.

    The workspace-backed eager op returns a view into storage shared by all
    layers.  Returning that hidden alias from a custom op is not representable
    to Dynamo/Inductor.  In compiled graphs the caller owns ``out`` and the
    mutation is declared in the operator schema, restoring a correct lifetime.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    num_tokens = hidden_states.shape[0]
    tail_bytes, gather_bytes = layer._fl_prefill_workspace_bytes(num_tokens)
    q_padded, _, prefill_workspace, gather_workspace = (
        layer._fl_attention_io_workspace.get(num_tokens, tail_bytes, gather_bytes)
    )
    layer.attention_impl(
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        qr_q,
        qr_scale,
        positions,
        out,
        q_padded,
        prefill_workspace,
        gather_workspace,
    )


def _deepseek_v4_fl_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
    qr_q: torch.Tensor | None,
    qr_scale: torch.Tensor | None,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    del (
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        qr_q,
        qr_scale,
        positions,
        out,
        layer_name,
    )


direct_register_custom_op(
    op_name="deepseek_v4_fl_attention",
    op_func=_deepseek_v4_fl_attention,
    mutates_args=["out"],
    fake_impl=_deepseek_v4_fl_attention_fake,
)


def _deepseek_v4_fl_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    hidden_size: int,
    layer_name: str,
) -> torch.Tensor:
    del hidden_size
    layer = get_forward_context().no_compile_layers[layer_name]
    return DeepseekV4FlashMLAAttention._o_proj(layer, o, positions)


def _deepseek_v4_fl_fp8_o_proj_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    hidden_size: int,
    layer_name: str,
) -> torch.Tensor:
    del positions, layer_name
    return torch.empty(
        (o.shape[0], hidden_size),
        dtype=o.dtype,
        device=o.device,
    )


direct_register_custom_op(
    op_name="deepseek_v4_fl_fp8_o_proj",
    op_func=_deepseek_v4_fl_fp8_o_proj,
    mutates_args=[],
    fake_impl=_deepseek_v4_fl_fp8_o_proj_fake,
)


class DeepseekV4FLFlashMLAAttention(DeepseekV4FlashMLAAttention):
    """FlashMLA attention with a W8A8-only wo_a branch."""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = args[0] if args else kwargs["vllm_config"]
        self._fl_compile_mode = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
        )
        self._fl_qr_use_pass = self._fl_compile_mode
        super().__init__(*args, **kwargs)
        indexer = self.indexer
        if indexer is None:
            return
        quantization_config = getattr(indexer.config, "quantization_config", None)
        use_int8_kv = (
            isinstance(quantization_config, dict)
            and quantization_config.get("format") == "int-quantized"
        )
        indexer.use_int8_kv = use_int8_kv
        if not use_int8_kv:
            return

        # The 512-wide Sparse FlashMLA cache remains FP8. Only the independent
        # 128-wide lightning-indexer cache switches to INT8 plus one fp32 scale.
        indexer.use_fp4_kv = False
        indexer.compressor.use_fp4_cache = False
        native_op = indexer.indexer_op
        indexer.indexer_op = SparseAttnIndexerFL(
            native_op.k_cache,
            native_op.quant_block_size,
            native_op.scale_fmt,
            native_op.topk_tokens,
            native_op.head_dim,
            native_op.max_model_len,
            native_op.max_total_seq_len,
            native_op.topk_indices_buffer,
            skip_k_cache_insert=native_op.skip_k_cache_insert,
            use_fp4_cache=False,
        )

    def attn_gemm_parallel_execute(
        self,
        hidden_states: torch.Tensor,
        hidden_states_q: torch.Tensor | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ):
        if hidden_states_q is None or hidden_states_scale is None:
            return super().attn_gemm_parallel_execute(hidden_states)

        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]
        aux_fns = [None, None, None]

        if self.compressor is not None:
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        def fused_wqa_wkv() -> torch.Tensor:
            layer = self.fused_wqa_wkv
            return int8_scaled_mm(
                hidden_states_q,
                layer.weight,
                hidden_states_scale,
                layer.weight_scale,
                torch.bfloat16,
                layer.bias,
            )

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    @staticmethod
    def _linear_from_int8(layer, x_q, x_scale):
        return int8_scaled_mm(
            x_q,
            layer.weight,
            x_scale,
            layer.weight_scale,
            torch.bfloat16,
            layer.bias,
        )

    def _indexer_forward(
        self,
        indexer,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        qr_q: torch.Tensor | None = None,
        qr_scale: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        compressor = indexer.compressor

        def wq_b_and_q_quant():
            if qr_q is not None and qr_scale is not None:
                q = self._linear_from_int8(indexer.wq_b, qr_q, qr_scale)
            else:
                q, _ = indexer.wq_b(qr)
            q = q.view(-1, indexer.n_head, indexer.head_dim)
            if getattr(indexer, "use_int8_kv", False):
                return fused_indexer_q_rope_quant_int8(
                    positions,
                    q,
                    self.indexer_rotary_emb.cos_sin_cache,
                    indexer_weights,
                    indexer.softmax_scale,
                    indexer.n_head**-0.5,
                )
            return fused_indexer_q_rope_quant(
                positions,
                q,
                self.indexer_rotary_emb.cos_sin_cache,
                indexer_weights,
                indexer.softmax_scale,
                indexer.n_head**-0.5,
                use_fp4=indexer.use_fp4_kv,
            )

        (q_quant, weights), k = maybe_execute_in_parallel(
            wq_b_and_q_quant,
            lambda: (
                compress_int8_indexer_k_cache(
                    compressor,
                    compressed_kv_score,
                    positions,
                    self.indexer_rotary_emb,
                )
                if getattr(indexer, "use_int8_kv", False)
                else compressor(
                    compressed_kv_score,
                    positions,
                    self.indexer_rotary_emb,
                )
            ),
            indexer.ln_events[0],
            indexer.ln_events[1],
            indexer.aux_stream,
        )
        if getattr(indexer, "use_int8_kv", False):
            if workspace is None:
                workspace = hidden_states.new_empty((0,), dtype=torch.uint8)
            return indexer.indexer_op(hidden_states, q_quant, k, weights, workspace)
        return indexer.indexer_op(hidden_states, q_quant, k, weights)

    def _fl_prefill_workspace_bytes(self, num_tokens: int) -> tuple[int, int]:
        """Return packed tail bytes and its post-indexer gather subregion."""
        if num_tokens <= self._fl_attention_io_workspace.decode_capacity:
            return 0, 0

        output_shape = (num_tokens, self.padded_heads, self.head_dim)
        output_bytes = _workspace_nbytes(output_shape, torch.bfloat16)
        attn_metadata = get_forward_context().attn_metadata
        gather_bytes = 0
        self._fl_prefill_chunk_plan = None
        if isinstance(attn_metadata, dict):
            swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
            if swa_metadata is not None and swa_metadata.num_prefill_tokens > 0:
                chunk_plan = swa_metadata.get_prefill_chunk_plan(
                    compress_ratio=self.compress_ratio,
                    prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
                )
                self._fl_prefill_chunk_plan = chunk_plan
                gather_bytes = max(
                    _workspace_nbytes(
                        (chunk_end - chunk_start, chunk_m, self.head_dim),
                        torch.bfloat16,
                    )
                    for chunk_start, chunk_end, _chunk_n, chunk_m in chunk_plan
                )
        # O and gather are simultaneously live during sparse attention, while
        # indexer scratch is dead by then. The indexer consumes this same tail
        # when it fits; unusually large indexer-only shapes keep using vLLM's
        # existing global workspace rather than growing the persistent arena.
        return output_bytes + gather_bytes, gather_bytes

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        qr_q: torch.Tensor | None,
        qr_scale: torch.Tensor | None,
        positions: torch.Tensor,
        out: torch.Tensor,
        q_padded: torch.Tensor,
        prefill_workspace: torch.Tensor,
        gather_workspace: torch.Tensor,
    ) -> None:
        """Upstream orchestration with every direct CUDA helper dispatched."""
        attn_metadata = get_forward_context().attn_metadata
        if self.indexer is not None:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            assert self.compressor is not None
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                if qr_q is not None and qr_scale is not None:
                    q = self._linear_from_int8(self.wq_b, qr_q, qr_scale)
                else:
                    q = self.wq_b(qr)
                q = q.view(-1, self.n_local_heads, self.head_dim)
                return self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata, q_padded
                )

            q, _ = execute_in_parallel(
                wq_b_kv_insert,
                [
                    lambda: self._indexer_forward(
                        indexer,
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        qr_q,
                        qr_scale,
                        prefill_workspace,
                    ),
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None,
            )
        elif self.compressor is not None:
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                if qr_q is not None and qr_scale is not None:
                    q = self._linear_from_int8(self.wq_b, qr_q, qr_scale)
                else:
                    q = self.wq_b(qr)
                q = q.view(-1, self.n_local_heads, self.head_dim)
                return self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata, q_padded
                )

            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            if qr_q is not None and qr_scale is not None:
                q = self._linear_from_int8(self.wq_b, qr_q, qr_scale)
            else:
                q = self.wq_b(qr)
            q = q.view(-1, self.n_local_heads, self.head_dim)
            q = self._fused_qnorm_rope_kv_insert(
                q, kv, positions, attn_metadata, q_padded
            )
        # With compilation disabled, the profiling runner can retain
        # max_num_batched_tokens-sized backing buffers while the attention
        # metadata narrows Q to the actual token count. FlashMLA requires all
        # token views to have that same leading dimension.
        out = out[: q.shape[0]]
        kv = kv[: q.shape[0]]
        positions = positions[: q.shape[0]]
        self._fl_prefill_gather_workspace = gather_workspace
        try:
            self.forward_mqa(q, kv, positions, out)
        finally:
            self._fl_prefill_gather_workspace = None
            self._fl_prefill_chunk_plan = None

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata,
        q_padded: torch.Tensor,
    ) -> torch.Tensor:
        """Route each cache-layout-specific fused insert through OpManager."""
        if not isinstance(attn_metadata, dict):
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None
        swa_kv_cache = self.swa_cache_layer.kv_cache
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        kv = kv[: q.shape[0]]
        positions = positions[: q.shape[0]]
        slot_mapping = swa_metadata.slot_mapping[: q.shape[0]]

        if swa_kv_cache.dtype == torch.uint8:
            return qnorm_rope_kv_quant_insert(
                q,
                kv,
                swa_kv_cache.view(swa_kv_cache.shape[0], -1),
                slot_mapping,
                positions,
                cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
                q_padded[: q.shape[0]],
            )

        block_size = swa_metadata.block_size
        swa_kv_cache_3d = swa_kv_cache.view(-1, block_size, self.head_dim)
        if swa_kv_cache.dtype == torch.bfloat16:
            qnorm_rope_kv_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
            )
            return q

        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        qnorm_rope_kv_fp8_insert(
            q,
            kv,
            q_fp8,
            swa_kv_cache_3d,
            slot_mapping,
            positions,
            cos_sin_cache,
            self._flashinfer_fp8_kv_scale,
            self._flashinfer_fp8_q_scale_inv,
            self.eps,
            block_size,
        )
        return q_fp8

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata,
        attn_metadata,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        """Decode path with all DSV4 CUDA helpers behind OpManager."""
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        q = q.unsqueeze(1)
        if topk_indices is not None:
            topk_indices = topk_indices[: q.shape[0]]
        if topk_lens is not None:
            topk_lens = topk_lens[: q.shape[0]]
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None
        flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_metadata.decode_swa_indices[: q.shape[0]],
            topk_length=swa_metadata.decode_swa_lens[: q.shape[0]],
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata,
        swa_metadata,
    ) -> None:
        """Prefill path with gather/index construction/attention dispatched."""
        del positions
        swa_only = attn_metadata is None
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0

        chunk_plan = getattr(self, "_fl_prefill_chunk_plan", None)
        if chunk_plan is None:
            chunk_plan = swa_metadata.get_prefill_chunk_plan(
                compress_ratio=self.compress_ratio,
                prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            )
        assert chunk_plan
        gather_workspace = getattr(self, "_fl_prefill_gather_workspace", None)
        assert gather_workspace is not None
        for chunk_start, chunk_end, chunk_n, chunk_m in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv_spec = ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16)
            if gather_workspace.numel() >= _workspace_nbytes(*kv_spec):
                kv = _workspace_views(gather_workspace, kv_spec)[0]
            else:
                kv = current_workspace_manager().get_simultaneous(kv_spec)[0]
            if not swa_only:
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_n,
            )
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            metadata_query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            if query_start >= q.shape[0]:
                break
            actual_query_end = min(metadata_query_end, q.shape[0])
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:metadata_query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_m,
                chunk_n,
            )
            q_chunk = q[query_start:actual_query_end]
            actual_query_tokens = q_chunk.shape[0]
            flash_mla_sparse_fwd(
                q=q_chunk,
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices[:actual_query_tokens].unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens[:actual_query_tokens],
                out=output[query_start:actual_query_end],
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        hidden_states_q: torch.Tensor | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        num_tokens = hidden_states.shape[0]
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(
                hidden_states, hidden_states_q, hidden_states_scale
            )
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        if self.wq_b.weight.dtype == torch.int8 and self._fl_qr_use_pass:
            # Keep a stable semantic target in compiled graphs so the fusion
            # pass can replace it with the dedicated fused op. Eager/CUDA
            # graph mode retains the faster independent normalization path.
            qr, kv, qr_q, qr_scale = torch.ops.vllm.deepseek_v4_qr_int8.default(
                qr,
                kv,
                self.q_norm.weight.data,
                self.kv_norm.weight.data,
                self.eps,
            )
        else:
            qr, kv = fused_q_kv_rmsnorm(
                qr,
                kv,
                self.q_norm.weight.data,
                self.kv_norm.weight.data,
                self.eps,
            )
            qr_q = qr_scale = None

        if self._fl_compile_mode:
            o_padded = torch.empty(
                (num_tokens, self.padded_heads, self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            torch.ops.vllm.deepseek_v4_fl_attention(
                hidden_states,
                qr,
                kv,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                qr_q,
                qr_scale,
                positions,
                o_padded,
                self.prefix,
            )
            o = o_padded[:, : self.n_local_heads, :]
        else:
            o = torch.ops.vllm.deepseek_v4_fl_attention_workspace(
                hidden_states,
                qr,
                kv,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                qr_q,
                qr_scale,
                positions,
                self.padded_heads,
                self.n_local_heads,
                self.head_dim,
                self.prefix,
            )
        return self._o_proj(o, positions)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        grouped_weight = getattr(self.wo_a, "_fl_w8a8_grouped_weight", None)
        grouped_scale = getattr(self.wo_a, "_fl_w8a8_grouped_weight_scale", None)
        if grouped_weight is None or grouped_scale is None:
            weight = getattr(self.wo_a, "weight", None)
            weight_scale = getattr(self.wo_a, "weight_scale", None)
            if weight is None or weight.dtype != torch.int8 or weight_scale is None:
                return torch.ops.vllm.deepseek_v4_fl_fp8_o_proj(
                    o,
                    positions,
                    self.hidden_size,
                    self.prefix,
                )
            output_per_group = weight.shape[1] // self.n_local_groups
            grouped_scale = weight_scale.reshape(self.n_local_groups, output_per_group)

        o_q, o_scale = fused_inv_rope_quant_int8(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.n_local_groups,
            self.n_local_heads // self.n_local_groups,
            self.nope_head_dim,
            self.rope_head_dim,
        )
        outputs = []
        for group_idx in range(self.n_local_groups):
            if grouped_weight is None:
                start = group_idx * output_per_group
                group_weight = weight[:, start : start + output_per_group]
            else:
                group_weight = grouped_weight[group_idx].transpose(0, 1)
            outputs.append(
                int8_scaled_mm(
                    o_q[group_idx],
                    group_weight,
                    o_scale[group_idx],
                    grouped_scale[group_idx],
                    o.dtype,
                )
            )
        return self.wo_b(torch.stack(outputs, dim=1).flatten(1))


class DeepseekV4FLDecoderLayer(DeepseekV4DecoderLayer):
    """Upstream decoder layer with a thin attention-class substitution."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        attn_cls = _select_dsv4_attn_cls(vllm_config)
        if attn_cls is DeepseekV4FlashMLAAttention:
            attn_cls = DeepseekV4FLFlashMLAAttention
        self.attn = attn_cls(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep NVIDIA MHC kernels opaque to Dynamo via registered ops."""
        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        use_mhc_int8 = self.attn.fused_wqa_wkv.weight.dtype == torch.int8
        if residual is None:
            residual = x
            mhc_result = (mhc_pre_int8 if use_mhc_int8 else mhc_pre)(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                1,
                attn_norm_weight,
                attn_norm_eps,
            )
            post_mix, res_mix, x = mhc_result[:3]
            x_q, x_scale = mhc_result[3:] if use_mhc_int8 else (None, None)
        else:
            assert post_mix is not None and res_mix is not None
            mhc_result = (
                mhc_fused_post_pre_int8 if use_mhc_int8 else mhc_fused_post_pre
            )(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                1,
                1,
                attn_norm_weight,
                attn_norm_eps,
            )
            residual, post_mix, res_mix, x = mhc_result[:4]
            x_q, x_scale = mhc_result[4:] if use_mhc_int8 else (None, None)

        x = self.attn(positions, x, None, x_q, x_scale)

        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        residual, post_mix, res_mix, x = mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            1,
            1,
            ffn_norm_weight,
            ffn_norm_eps,
        )
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class DeepseekV4FLModel(DeepseekV4Model):
    """Upstream DSV4 model whose layers use the FL decoder subclass."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError("DeepSeek V4 MegaMoE requires expert parallel")
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4FLDecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )
        local_attn_layers = [
            layer.attn
            for layer in islice(self.layers, self.start_layer, self.end_layer)
            if isinstance(layer, DeepseekV4FLDecoderLayer)
        ]
        assert local_attn_layers
        first_attn = local_attn_layers[0]
        max_graph_tokens = (
            vllm_config.compilation_config.max_cudagraph_capture_size or 0
        )
        decode_capacity = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            max(vllm_config.scheduler_config.max_num_seqs, max_graph_tokens),
        )
        attention_io_workspace = _DeepseekV4AttentionIOWorkspace(
            decode_capacity=decode_capacity,
            padded_heads=first_attn.padded_heads,
            head_dim=first_attn.head_dim,
            dtype=vllm_config.model_config.dtype,
            device=torch.device("cuda"),
            num_ubatches=vllm_config.parallel_config.num_ubatches,
        )
        for attn in local_attn_layers:
            attn._fl_attention_io_workspace = attention_io_workspace
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Run the upstream model flow with registered MHC custom ops."""
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        residual, post_mix, res_mix = None, None, None
        layer = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        if layer is not None:
            hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
        hidden_states = hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return self.norm(hidden_states)


class DeepseekV4FLForCausalLM(DeepseekV4ForCausalLM):
    """Registry entry retaining upstream behavior outside INT8 wo_a."""

    # The generic vLLM memory profile uses many short requests and skips
    # attention metadata. DSV4 needs its real single-request prefill path so
    # indexer/cache/sparse-attention workspaces are included in KV sizing.
    _fl_real_prefill_memory_profile = True

    model_cls = DeepseekV4FLModel


__all__ = ["DeepseekV4FLForCausalLM"]

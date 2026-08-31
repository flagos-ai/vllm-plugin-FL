# SPDX-License-Identifier: Apache-2.0
"""ModelRegistry thin model adding W8A8 DSV4 output projection support."""

from __future__ import annotations

from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
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
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_fl.ops.deepseek_v4 import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    fused_indexer_q_rope_quant,
    fused_q_kv_rmsnorm,
    hc_head,
    int8_scaled_mm,
    inv_rope_quant_fp8,
    mhc_fused_post_pre,
    mhc_post,
    mhc_pre,
    qnorm_rope_kv_bf16_insert,
    qnorm_rope_kv_fp8_insert,
    qnorm_rope_kv_quant_insert,
)
from vllm_fl.ops.deepseek_v4_int8_woa import fused_inv_rope_quant_int8


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


def _deepseek_v4_fl_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.attention_impl(
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        out,
    )


def _deepseek_v4_fl_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
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


class DeepseekV4FLFlashMLAAttention(DeepseekV4FlashMLAAttention):
    """FlashMLA attention with a W8A8-only wo_a branch."""

    def _indexer_forward(
        self,
        indexer,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        compressor = indexer.compressor

        def wq_b_and_q_quant():
            q, _ = indexer.wq_b(qr)
            q = q.view(-1, indexer.n_head, indexer.head_dim)
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
            lambda: compressor(compressed_kv_score, positions, self.indexer_rotary_emb),
            indexer.ln_events[0],
            indexer.ln_events[1],
            indexer.aux_stream,
        )
        return indexer.indexer_op(hidden_states, q_quant, k, weights)

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Upstream orchestration with every direct CUDA helper dispatched."""
        attn_metadata = get_forward_context().attn_metadata
        if self.indexer is not None:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            assert self.compressor is not None
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                return self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

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
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                return self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
            q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        self.forward_mqa(q, kv, positions, out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata,
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

        if swa_kv_cache.dtype == torch.uint8:
            return qnorm_rope_kv_quant_insert(
                q,
                kv,
                swa_kv_cache.view(swa_kv_cache.shape[0], -1),
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
            )

        block_size = swa_metadata.block_size
        swa_kv_cache_3d = swa_kv_cache.view(-1, block_size, self.head_dim)
        if swa_kv_cache.dtype == torch.bfloat16:
            qnorm_rope_kv_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
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
            swa_metadata.slot_mapping,
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
            indices=swa_metadata.decode_swa_indices,
            topk_length=swa_metadata.decode_swa_lens,
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

        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
        )
        assert chunk_plan
        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_n, chunk_m in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16),
            )[0]
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
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
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
            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )

        torch.ops.vllm.deepseek_v4_fl_attention(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
            self.prefix,
        )
        o = o_padded[:, : self.n_local_heads, :]
        return self._o_proj(o, positions)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        grouped_weight = getattr(self.wo_a, "_fl_w8a8_grouped_weight", None)
        grouped_scale = getattr(self.wo_a, "_fl_w8a8_grouped_weight_scale", None)
        if grouped_weight is None or grouped_scale is None:
            weight = getattr(self.wo_a, "weight", None)
            weight_scale = getattr(self.wo_a, "weight_scale", None)
            if weight is None or weight.dtype != torch.int8 or weight_scale is None:
                return super()._o_proj(o, positions)
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
        if residual is None:
            residual = x
            post_mix, res_mix, x = mhc_pre(
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
        else:
            assert post_mix is not None and res_mix is not None
            residual, post_mix, res_mix, x = mhc_fused_post_pre(
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

        x = self.attn(positions, x, None)

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

    model_cls = DeepseekV4FLModel


__all__ = ["DeepseekV4FLForCausalLM"]

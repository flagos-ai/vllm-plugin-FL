# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""V4.1 text attention topology and linear/cache orchestration.

Adapted from vLLM e77daef89e18. Numerical computation resolves through the
existing Plugin dispatcher to public FGV operators. No framework kernel is
copied; logical/physical index mapping remains framework metadata.
"""

import re

import torch
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import extract_layer_index

from .attention_metadata import logical_to_physical_indices
from .cache import DeepseekV41Cache
from .compressor import DeepseekV41Compressor
from .config import indexer_workspace_tokens
from .projections import V41FP32Projection, V41OutputProjection
from .rope import build_deepseek_v4_rope
from vllm_fl.dispatch import resolve_op


class V41RotaryTables(nn.Module):
    """One shared forward/inverse table pair per main/compressed RoPE family."""

    def __init__(self, config, compressed):
        super().__init__()
        self.rotary = build_deepseek_v4_rope(
            config,
            head_dim=512,
            rope_head_dim=64,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=int(compressed),
        )
        if self.rotary.cos_sin_cache.dtype != torch.float32:
            raise TypeError("V4 native RoPE constructor must retain FP32 caches")
        self.register_buffer(
            "inverse_sin", -self.rotary.cos_sin_cache[:, 32:], persistent=False
        )

    @property
    def cos_sin_cache(self):
        return self.rotary.cos_sin_cache


def _source_prefix(prefix, source):
    replacement, count = re.subn(r"\.layers\.\d+\.", f".layers.{source}.", prefix)
    if count != 1:
        raise ValueError("attention prefix must identify exactly one decoder layer")
    return replacement


def _norm_parameters(width):
    module = nn.Module()
    module.weight = nn.Parameter(
        torch.empty(width, dtype=torch.bfloat16), requires_grad=False
    )
    return module


class DeepseekV41Indexer(nn.Module):
    def __init__(
        self,
        vllm_config,
        *,
        prefix,
        ratio,
        cache,
        owns_k,
        indices,
        candidate_blocks,
        candidate_write,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.heads, self.topk, self.ratio = (
            config.index_n_heads,
            config.index_topk,
            ratio,
        )
        if (self.heads, config.index_head_dim, self.topk) != (32, 128, 512):
            raise NotImplementedError(
                "V4.1 indexer requires heads32, dim128 and topk512"
            )
        self.max_length = (indexer_workspace_tokens(vllm_config) + ratio - 1) // ratio
        self.k_cache, self.owns_k = cache, owns_k
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.heads * 128,
            bias=False,
            quant_config=vllm_config.quant_config,
            params_dtype=torch.bfloat16,
            prefix=prefix + ".wq_b",
        )
        self.weights_proj = V41FP32Projection(
            config.hidden_size, self.heads, prefix=prefix + ".weights_proj"
        )
        if owns_k:
            self.wk = ReplicatedLinear(
                512,
                128,
                bias=False,
                quant_config=vllm_config.quant_config,
                params_dtype=torch.bfloat16,
                prefix=prefix + ".wk",
            )
            self.k_norm = _norm_parameters(128)
        self.eps = config.rms_norm_eps
        self.indices = indices
        self.candidate_blocks = candidate_blocks
        self.candidate_write = candidate_write
        self.block_size = config.candidate_block_size
        # Existing MetaX TopK uses caller-owned scratch. Capacity covers every
        # row's 4096-element chunk histograms plus state in the declared domain.
        rows = vllm_config.scheduler_config.max_num_batched_tokens
        scratch_bytes = max(
            8 * 1024 * 1024, rows * (((self.max_length + 4095) // 4096) * 256 + 6) * 4
        )
        self.register_buffer(
            "workspace", torch.empty(scratch_bytes, dtype=torch.uint8), persistent=False
        )
        self.ops = {
            name: resolve_op(name)
            for name in (
                "deepseek_v41_indexer_k_store",
                "fused_indexer_q_rope_quant",
                "paged_mqa_logits",
                "deepseek_v41_select_candidate_blocks",
                "deepseek_v41_apply_candidate_mask",
                "persistent_topk",
            )
        }

    def forward(self, hidden, qr, latent, positions, rope, metadata):
        if self.owns_k:
            if latent is None:
                raise ValueError("index cache owner requires the compressor latent")
            k, _ = self.wk(latent)
            self.ops["deepseek_v41_indexer_k_store"](
                k,
                metadata.group_positions,
                rope.cos_sin_cache,
                self.k_norm.weight,
                self.eps,
                self.k_cache.kv_cache,
                metadata.slot_mapping,
            )
        q, _ = self.wq_b(qr)
        q = q.view(q.shape[0], self.heads, 128)
        weights = self.weights_proj(hidden)
        query, folded_weights = self.ops["fused_indexer_q_rope_quant"](
            positions,
            q,
            rope.cos_sin_cache,
            weights,
            128**-0.5,
            self.heads**-0.5,
            use_fp4=False,
        )
        n = hidden.shape[0]
        if metadata.token_block_tables is None:
            raise ValueError("indexer requires per-token block tables")
        logits = self.ops["paged_mqa_logits"](
            (query, None),
            self.k_cache.as_attention_cache(),
            folded_weights,
            metadata.context_lens.view(n, 1),
            metadata.token_block_tables,
            None,
            self.max_length,
            clean_logits=True,
        )
        if self.candidate_blocks is not None:
            candidates = self.candidate_blocks[:n]
            if self.candidate_write:
                self.ops["deepseek_v41_select_candidate_blocks"](
                    logits,
                    None,
                    metadata.context_lens,
                    candidates.shape[1],
                    self.block_size,
                    candidates,
                )
            else:
                self.ops["deepseek_v41_apply_candidate_mask"](
                    logits, None, metadata.context_lens, candidates, self.block_size
                )
        output = self.indices[:n]
        self.ops["persistent_topk"](
            logits,
            metadata.context_lens,
            output,
            self.workspace,
            self.topk,
            self.max_length,
            stable=True,
        )
        return output


class DeepseekV41Attention(nn.Module):
    def __init__(
        self, vllm_config, *, prefix, indices, candidate_blocks, rotary_tables
    ):
        super().__init__()
        c = vllm_config.model_config.hf_config
        tp = get_tensor_model_parallel_world_size()
        if (
            tp not in (1, 2, 4, 8, 16)
            or c.num_attention_heads != 64
            or c.o_groups != 8
            or c.head_dim != 512
            or c.qk_rope_head_dim != 64
        ):
            raise NotImplementedError(
                "V4.1 grouped output requires TP1/2/4/8/16, heads64, groups8 and dim512/64"
            )
        self.layer_id = extract_layer_index(prefix)
        self.prefix, self.heads, self.eps = prefix, 64 // tp, c.rms_norm_eps
        self.output_group_rank = get_tensor_model_parallel_rank() // max(
            1, tp // c.o_groups
        )
        self.gather_output_group = tp > c.o_groups
        self.ratio = c.compress_ratios[self.layer_id]
        if self.ratio not in (0, 1, 2):
            raise ValueError("V4.1 attention supports ratios0/1/2")
        self.is_kv_source = self.layer_id in c.kv_source_layer_ids
        self.is_index_source = self.layer_id in c.index_source_layer_ids
        self.kv_source = (
            max((i for i in c.kv_source_layer_ids if i <= self.layer_id), default=None)
            if self.ratio
            else None
        )
        self.index_source = (
            max(
                (i for i in c.index_source_layer_ids if i <= self.layer_id),
                default=None,
            )
            if self.ratio
            else None
        )
        quant = vllm_config.quant_config
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            c.hidden_size,
            [c.q_lora_rank, 512],
            bias=False,
            disable_tp=True,
            quant_config=quant,
            params_dtype=torch.bfloat16,
            return_bias=False,
            prefix=prefix + ".fused_wqa_wkv",
        )
        self.q_norm, self.kv_norm = (
            _norm_parameters(c.q_lora_rank),
            _norm_parameters(512),
        )
        self.wq_b = ColumnParallelLinear(
            c.q_lora_rank,
            64 * 512,
            bias=False,
            quant_config=quant,
            params_dtype=torch.bfloat16,
            return_bias=False,
            prefix=prefix + ".wq_b",
        )
        self.wo_a = V41OutputProjection(
            8 * 512, c.o_groups * c.o_lora_rank, prefix=prefix + ".wo_a"
        )
        self.wo_b = RowParallelLinear(
            c.o_groups * c.o_lora_rank,
            c.hidden_size,
            bias=False,
            quant_config=quant,
            params_dtype=torch.bfloat16,
            return_bias=False,
            prefix=prefix + ".wo_b",
        )
        self.attn_sink = nn.Parameter(
            torch.full((64,), -float("inf"), dtype=torch.float32), requires_grad=False
        )
        key = "compress" if self.ratio else "main"
        if key not in rotary_tables:
            rotary_tables[key] = V41RotaryTables(c, bool(self.ratio))
        self.rotary_tables = rotary_tables[key]
        self.swa_cache_layer = DeepseekV41Cache(
            vllm_config, prefix + ".swa_cache", kind="swa"
        )
        context = vllm_config.compilation_config.static_forward_context
        self.main_cache = self.index_cache = self.compressor = self.indexer = None
        if self.ratio:
            if self.kv_source is None or self.index_source is None:
                raise ValueError(
                    "compressed attention requires a preceding KV and index source"
                )
            owner = _source_prefix(prefix, self.kv_source)
            if self.is_kv_source:
                self.main_cache = DeepseekV41Cache(
                    vllm_config,
                    owner + ".compressed_cache",
                    kind="compressed",
                    compress_ratio=self.ratio,
                )
                self.index_cache = DeepseekV41Cache(
                    vllm_config,
                    owner + ".indexer.k_cache",
                    kind="index",
                    compress_ratio=self.ratio,
                )
                self.compressor = DeepseekV41Compressor(
                    vllm_config, self.ratio, prefix=prefix + ".compressor"
                )
            else:
                self.main_cache = context.get(owner + ".compressed_cache")
                self.index_cache = context.get(owner + ".indexer.k_cache")
                if self.main_cache is None or self.index_cache is None:
                    raise NotImplementedError(
                        "PP split crosses a V4.1 KV sharing group"
                    )
            if self.is_index_source:
                self.indexer = DeepseekV41Indexer(
                    vllm_config,
                    prefix=prefix + ".indexer",
                    ratio=self.ratio,
                    cache=self.index_cache,
                    owns_k=self.is_kv_source,
                    indices=indices,
                    candidate_blocks=(
                        candidate_blocks
                        if self.layer_id >= c.candidate_source_layer_id
                        else None
                    ),
                    candidate_write=self.layer_id == c.candidate_source_layer_id,
                )
        self.indices = indices
        self.ops = {
            name: resolve_op(name)
            for name in (
                "fused_q_kv_rmsnorm",
                "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
                "v41_sparse_attention",
                "apply_rotary_pos_emb",
            )
        }

    def forward(
        self,
        positions,
        hidden,
        *,
        compressor_state_slots=None,
        compressor_token_to_req=None,
        compressor_query_start_loc=None,
    ):
        n = hidden.shape[0]
        if not n:
            return torch.empty_like(hidden)
        qkv = self.fused_wqa_wkv(hidden)
        qr, kv = qkv.split([self.q_norm.weight.numel(), 512], -1)
        qr, kv = self.ops["fused_q_kv_rmsnorm"](
            qr, kv, self.q_norm.weight, self.kv_norm.weight, self.eps
        )
        q = self.wq_b(qr).view(n, self.heads, 512)
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            raise NotImplementedError(
                "V4.1 initial text path excludes DBO microbatch lists"
            )
        if not isinstance(metadata, dict):
            # Profile run: no allocator-owned cache exists yet. Keep linear
            # output allocation/GEMMs in the profile without touching cache.
            o = torch.zeros_like(q)
        else:
            swa = metadata[self.swa_cache_layer.prefix].for_num_tokens(n)
            main = (
                metadata[self.main_cache.prefix].for_num_tokens(n)
                if self.main_cache is not None
                else None
            )
            if (
                self.ratio == 2
                and self.compressor is not None
                and any(
                    t is None
                    for t in (
                        compressor_state_slots,
                        compressor_token_to_req,
                        compressor_query_start_loc,
                    )
                )
            ):
                raise ValueError("V4.1 attention requires V2 compressor state inputs")
            latent = None
            if self.compressor is not None:
                latent = self.compressor(
                    hidden,
                    positions,
                    main,
                    state_slots=compressor_state_slots,
                    token_to_req=compressor_token_to_req,
                    query_start_loc=compressor_query_start_loc,
                )
                empty_q = torch.empty(
                    (n, 0, 512), dtype=torch.bfloat16, device=hidden.device
                )
                self.ops["fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"](
                    empty_q,
                    latent,
                    self.main_cache.as_page_bytes(),
                    main.slot_mapping,
                    main.group_positions,
                    self.rotary_tables.cos_sin_cache,
                    self.eps,
                    main.storage_block_size,
                    apply_q_norm=False,
                )
            self.ops["fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"](
                q,
                kv,
                self.swa_cache_layer.as_page_bytes(),
                swa.slot_mapping,
                positions,
                self.rotary_tables.cos_sin_cache,
                self.eps,
                swa.storage_block_size,
                apply_q_norm=False,
            )
            if self.indexer is not None:
                self.indexer(
                    hidden,
                    qr,
                    latent,
                    positions,
                    self.rotary_tables,
                    metadata[self.index_cache.prefix].for_num_tokens(n),
                )
            extra_indices = None
            if self.main_cache is not None:
                physical = torch.empty_like(self.indices[:n])
                logical_to_physical_indices(
                    self.indices[:n],
                    main.token_to_req,
                    main.context_lens,
                    main.block_table,
                    self.main_cache.kv_cache.shape[0],
                    main.storage_block_size,
                    physical,
                )
                extra_indices = physical[:, None, :]
            q_padded = torch.zeros((n, 64, 512), device=q.device, dtype=q.dtype)
            q_padded[:, : self.heads].copy_(q)
            o = self.ops["v41_sparse_attention"](
                q_padded[:, None],
                self.swa_cache_layer.as_attention_cache(),
                swa.swa_indices,
                self.attn_sink,
                extra_cache=(
                    self.main_cache.as_attention_cache()
                    if self.main_cache is not None
                    else None
                ),
                extra_indices=extra_indices,
            )[:, 0, : self.heads]
            # The public rotary operator requires a K head. Use a separate
            # meaningful copy; aliasing Q here would rotate that head twice.
            tail = o[:, :, 448:]
            dummy = tail[:, :1].clone(memory_format=torch.contiguous_format)
            self.ops["apply_rotary_pos_emb"](
                tail,
                dummy,
                self.rotary_tables.cos_sin_cache[:, :32],
                self.rotary_tables.inverse_sin,
                positions,
                rotary_interleaved=True,
                inplace=True,
            )
        grouped_output = o.reshape(n, self.heads * 512)
        if self.gather_output_group:
            # TP16 splits each eight-head output group across two ranks.
            # Restore its complete input before the ordinary column-sharded
            # WO_A; WO_B consumes the corresponding output/scale column slice.
            gathered = tensor_model_parallel_all_gather(grouped_output, dim=-1)
            grouped_output = gathered.narrow(-1, self.output_group_rank * 4096, 4096)
        return self.wo_b(self.wo_a(grouped_output))

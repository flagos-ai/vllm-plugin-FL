# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 model-owned composition with dispatch-selected math providers.

These building blocks retain checkpoint names and explicit norm placement.
Cache allocation and scheduler metadata remain outside the model.
"""

import torch
from torch import nn

from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
)
from vllm.sequence import IntermediateTensors

from vllm_fl.dispatch import CachedOp
from vllm_fl.kv_cache.deepseek_v4 import create_deepseek_v4_cache_layers
from vllm_fl.ops.activation import SiluAndMulFL, SiluAndMulWithClampFL
from vllm_fl.ops.fused_moe.layer import FusedMoEFL
from vllm_fl.ops.layernorm import RMSNormFL as RMSNorm

_compressor = CachedOp("deepseek_v4_compressor")
_hc_pre = CachedOp("deepseek_v4_hc_pre")
_hc_post = CachedOp("deepseek_v4_hc_post")
_hc_head = CachedOp("deepseek_v4_hc_head")
_rope = CachedOp("deepseek_v4_partial_rope")
_hadamard = CachedOp("deepseek_v4_hadamard")
_index_quant = CachedOp("deepseek_v4_index_quant")
_grouped_project = CachedOp("deepseek_v4_grouped_project")
_index = CachedOp("deepseek_v4_indexer")
_sparse_attention = CachedOp("deepseek_v4_sparse_attention")


class DeepseekV4Compressor(nn.Module):
    def __init__(
        self,
        hidden_size,
        head_dim,
        compress_ratio,
        *,
        rms_eps=1e-6,
        params_dtype=torch.bfloat16,
        prefix="",
    ):
        super().__init__()
        if compress_ratio not in (4, 128):
            raise ValueError("Compression ratio must be 4 or 128")
        self.compress_ratio = compress_ratio
        self.norm_eps = rms_eps
        coff = 2 if compress_ratio == 4 else 1
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32),
            requires_grad=False,
        )
        for name in ("wkv", "wgate"):
            layer = ReplicatedLinear(
                hidden_size,
                coff * head_dim,
                bias=False,
                params_dtype=params_dtype,
                quant_config=None,
                prefix=f"{prefix}.{name}" if prefix else name,
                return_bias=False,
                disable_tp=True,
            )
            setattr(self, name, layer)
        self.norm = RMSNorm(head_dim, rms_eps, dtype=params_dtype)

    def forward(
        self,
        x,
        positions,
        slot_mapping,
        state_cache,
        state_block_table,
        output_request_ids,
        output_positions,
        cos_sin_cache,
    ):
        return _compressor(
            x,
            self.wkv.weight,
            self.wgate.weight,
            self.ape,
            state_cache,
            positions,
            slot_mapping,
            state_block_table,
            output_request_ids,
            output_positions,
            self.norm.weight,
            cos_sin_cache,
            compress_ratio=self.compress_ratio,
            rms_eps=self.norm_eps,
        )

    def forward_from_cache(
        self,
        x,
        state_leaf,
        compact_leaf,
        state_metadata,
        compact_metadata,
        cos_sin_cache,
    ):
        if (
            state_leaf.spec.compress_ratio != 1
            or compact_leaf.spec.compress_ratio != self.compress_ratio
        ):
            raise ValueError("Compressor cache ratios do not match the model")
        output = self(
            x,
            state_metadata.positions,
            state_metadata.slot_mapping,
            state_leaf.kv_cache,
            state_metadata.block_table,
            compact_metadata.output_request_ids,
            compact_metadata.output_positions,
            cos_sin_cache,
        )
        compact_leaf.store(output, compact_metadata)
        return output


class DeepseekV4Indexer(nn.Module):
    """C4 indexer with checkpoint-facing projections and injected cache leaves."""

    def __init__(self, config, quant_config, cache_layers, *, prefix, params_dtype):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
            params_dtype=params_dtype,
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
            params_dtype=params_dtype,
        )
        self.compressor = DeepseekV4Compressor(
            config.hidden_size,
            self.head_dim,
            4,
            rms_eps=config.rms_norm_eps,
            params_dtype=params_dtype,
            prefix=f"{prefix}.compressor",
        )
        self.compressor.state_cache = cache_layers["index_state"]
        self.k_cache = cache_layers["index_key"]
        self.scale_cache = cache_layers["index_scale"]
        matrix = torch.ones(1, 1, dtype=params_dtype)
        while matrix.shape[0] < self.head_dim:
            matrix = torch.cat(
                [
                    torch.cat([matrix, matrix], 1),
                    torch.cat([matrix, -matrix], 1),
                ],
                0,
            )
        self.register_buffer("hadamard", matrix, persistent=False)

    def forward(self, x, qr, positions, cos_sin_cache, metadata, query_metadata):
        q = self.wq_b(qr).reshape(x.shape[0], self.n_heads, self.head_dim)
        q = _hadamard(_rope(q, positions, cos_sin_cache), self.hadamard)
        smeta = metadata[self.compressor.state_cache.prefix]
        kmeta = metadata[self.k_cache.prefix]
        # Keys rotate after compression/RMSNorm/RoPE, not before state storage.
        keys = self.compressor(
            x,
            smeta.positions,
            smeta.slot_mapping,
            self.compressor.state_cache.kv_cache,
            smeta.block_table,
            kmeta.output_request_ids,
            kmeta.output_positions,
            cos_sin_cache,
        )
        keys = _hadamard(keys, self.hadamard)
        q, q_scale = _index_quant(q)
        keys, key_scale = _index_quant(keys)
        self.k_cache.store(keys, kmeta)
        self.scale_cache.store(key_scale, metadata[self.scale_cache.prefix])
        weights = self.weights_proj(x) * (self.head_dim**-0.5 * self.n_heads**-0.5)
        return _index(
            q,
            self.k_cache.kv_cache,
            weights.half(),
            kmeta.block_table,
            query_metadata.request_ids,
            query_metadata.positions,
            kmeta.key_lengths,
            self.topk,
            query_scale=q_scale,
            key_scale_cache=self.scale_cache.kv_cache,
        )


class DeepseekV4Attention(nn.Module):
    """Compose V4 attention without binding model control to an operator backend.

    The model factory supplies scheduler-registered cache leaves and an FP32
    RoPE cache, shared where appropriate. This module does not own allocation,
    page-size policy or global model registration.
    """

    def __init__(
        self,
        vllm_config,
        config,
        cache_layers,
        cos_sin_cache,
        *,
        prefix,
        params_dtype=torch.bfloat16,
    ):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        ratio = config.compress_ratios[extract_layer_index(prefix)]
        parallel = vllm_config.parallel_config
        if vllm_config.speculative_config is not None or (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise NotImplementedError(
                "Reference V4 attention requires non-speculative CP1"
            )
        if ratio not in (0, 4, 128) or params_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Reference V4 attention requires BF16 and C0/C4/C128"
            )
        if min(config.num_attention_heads, config.o_groups, config.sliding_window) < 1:
            raise ValueError("Attention heads, groups and window must be positive")
        if (
            config.num_attention_heads % tp
            or config.o_groups % tp
            or config.num_attention_heads % config.o_groups
        ):
            raise ValueError("Attention heads/groups must divide the active TP size")
        if (
            cos_sin_cache.ndim != 2
            or cos_sin_cache.shape[0] == 0
            or cos_sin_cache.shape[-1] != config.qk_rope_head_dim
            or cos_sin_cache.dtype != torch.float32
        ):
            raise ValueError("Expected FP32 [positions, rope_dim] cache")
        if (
            config.qk_rope_head_dim <= 0
            or config.qk_rope_head_dim % 2
            or config.qk_rope_head_dim > config.head_dim
        ):
            raise ValueError("Invalid partial RoPE dimension")
        required = {"swa"}
        if ratio:
            required |= {"state", "compact"}
        if ratio == 4:
            required |= {"index_state", "index_key", "index_scale"}
            if (
                config.index_head_dim < config.qk_rope_head_dim
                or min(config.index_n_heads, config.index_topk) < 1
            ):
                raise ValueError("Invalid indexer dimensions")
        if not required <= cache_layers.keys():
            raise ValueError(
                f"Missing scheduler cache leaves: {required - cache_layers.keys()}"
            )
        expected = {"swa": (config.head_dim, torch.bfloat16, 1)}
        if ratio:
            expected.update(
                {
                    "state": (
                        2 * (2 if ratio == 4 else 1) * config.head_dim,
                        torch.float32,
                        1,
                    ),
                    "compact": (config.head_dim, torch.bfloat16, ratio),
                }
            )
        if ratio == 4:
            expected.update(
                {
                    "index_state": (4 * config.index_head_dim, torch.float32, 1),
                    "index_key": (config.index_head_dim, torch.int8, 4),
                    "index_scale": (1, torch.float16, 4),
                }
            )
        for name, contract in expected.items():
            spec = cache_layers[name].spec
            if (spec.head_size, spec.dtype, spec.compress_ratio) != contract:
                raise ValueError(
                    f"Cache leaf does not match attention contract: {name}"
                )
        if (
            ratio == 4
            and cache_layers["index_key"].spec.block_size
            != cache_layers["index_scale"].spec.block_size
        ):
            raise ValueError(
                "Indexer K/scale leaves require the same logical block size"
            )
        self.hidden_size = config.hidden_size
        self.n_local_heads = config.num_attention_heads // tp
        self.n_local_groups = config.o_groups // tp
        self.head_dim = config.head_dim
        self.o_lora_rank = config.o_lora_rank
        self.window_size = config.sliding_window
        self.compress_ratio = ratio
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)
        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32), requires_grad=False
        )
        quant = vllm_config.quant_config
        for name, cls, inputs, outputs in (
            ("wq_a", ReplicatedLinear, config.hidden_size, config.q_lora_rank),
            (
                "wq_b",
                ColumnParallelLinear,
                config.q_lora_rank,
                config.num_attention_heads * self.head_dim,
            ),
            ("wkv", ReplicatedLinear, config.hidden_size, self.head_dim),
            (
                "wo_a",
                ColumnParallelLinear,
                config.num_attention_heads * self.head_dim // config.o_groups,
                config.o_groups * self.o_lora_rank,
            ),
            (
                "wo_b",
                RowParallelLinear,
                config.o_groups * self.o_lora_rank,
                config.hidden_size,
            ),
        ):
            setattr(
                self,
                name,
                cls(
                    inputs,
                    outputs,
                    bias=False,
                    quant_config=quant,
                    prefix=f"{prefix}.{name}",
                    return_bias=False,
                    params_dtype=params_dtype,
                ),
            )
        # The grouped operation consumes logical loader weights. Quantized
        # grouped layouts require a separate provider rather than a dense call.
        if self.wo_a.weight.dtype != params_dtype:
            raise NotImplementedError(
                "Grouped output projection requires floating weights"
            )
        self.q_norm = RMSNorm(
            config.q_lora_rank, config.rms_norm_eps, dtype=params_dtype
        )
        self.q_norm_without_weight = RMSNorm(
            self.head_dim, config.rms_norm_eps, has_weight=False, dtype=params_dtype
        )
        # The selected norm provider requires a gamma tensor. It is a constant,
        # not a checkpoint parameter, but must follow module device migration.
        gamma = self.q_norm_without_weight.weight
        del self.q_norm_without_weight.weight
        self.q_norm_without_weight.register_buffer("weight", gamma, persistent=False)
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps, dtype=params_dtype)
        self.swa_cache_layer = cache_layers["swa"]
        self.compressor = None
        self.indexer = None
        if ratio:
            self.compressor = DeepseekV4Compressor(
                config.hidden_size,
                self.head_dim,
                ratio,
                rms_eps=config.rms_norm_eps,
                params_dtype=params_dtype,
                prefix=f"{prefix}.compressor",
            )
            self.compressor.state_cache = cache_layers["state"]
            self.compact_cache = cache_layers["compact"]
        if ratio == 4:
            self.indexer = DeepseekV4Indexer(
                config,
                quant,
                cache_layers,
                prefix=f"{prefix}.indexer",
                params_dtype=params_dtype,
            )

    def forward(self, positions, hidden_states, llama_4_scaling=None, *, metadata=None):
        if llama_4_scaling is not None:
            raise NotImplementedError(
                "Llama4 scaling is not part of the V4 reference path"
            )
        if metadata is None:
            metadata = get_forward_context().attn_metadata
        if metadata is None:
            # Memory profiling precedes scheduler cache allocation. Keep that
            # path cache-free, while the surrounding HC/MoE still executes.
            return hidden_states.new_zeros((hidden_states.shape[0], self.hidden_size))
        swa = metadata[self.swa_cache_layer.prefix]
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != self.hidden_size
            or positions.shape != hidden_states.shape[:1]
            or swa.positions.shape != positions.shape
        ):
            raise ValueError("Attention inputs must match the scheduled token plan")
        active = swa.positions >= 0
        x = torch.where(active[:, None], hidden_states, 0)
        qr = self.q_norm(self.wq_a(x))
        query = self.wq_b(qr).reshape(x.shape[0], self.n_local_heads, self.head_dim)
        query = _rope(self.q_norm_without_weight(query), positions, self.cos_sin_cache)
        kv = _rope(self.kv_norm(self.wkv(x)), positions, self.cos_sin_cache)
        self.swa_cache_layer.store(kv, swa)
        compact = compact_meta = indices = None
        if self.compressor is not None:
            compact = self.compact_cache.kv_cache
            compact_meta = metadata[self.compact_cache.prefix]
            self.compressor.forward_from_cache(
                x,
                self.compressor.state_cache,
                self.compact_cache,
                metadata[self.compressor.state_cache.prefix],
                compact_meta,
                self.cos_sin_cache,
            )
            if self.indexer is not None:
                indices = self.indexer(
                    x, qr, positions, self.cos_sin_cache, metadata, swa
                )
            else:
                capacity = compact_meta.block_table.shape[1] * compact.shape[1]
                indices = torch.arange(capacity, device=x.device)[None].expand(
                    x.shape[0], -1
                )
        output = _sparse_attention(
            query,
            self.swa_cache_layer.kv_cache,
            compact,
            swa.block_table,
            compact_meta.block_table if compact_meta is not None else None,
            swa.request_ids,
            swa.positions,
            swa.seq_lens,
            compact_meta.key_lengths if compact_meta is not None else None,
            indices,
            self.window_size,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
        )
        output = _rope(output, positions, self.cos_sin_cache, inverse=True)
        output = _grouped_project(output, self.wo_a.weight, self.n_local_groups)
        output = self.wo_b(output)
        return torch.where(active[:, None], output, 0)


class DeepseekV4DecoderLayer(nn.Module):
    """Compose injected attention/MoE modules without binding their math backend."""

    def __init__(
        self,
        hidden_size,
        hc_mult,
        hc_sinkhorn_iters,
        hc_eps,
        rms_eps,
        self_attn,
        mlp,
        *,
        params_dtype=torch.bfloat16,
    ):
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise NotImplementedError("The current HC decoder requires BF16 residuals")
        self.self_attn = self_attn
        self.mlp = mlp
        self.norm_eps = rms_eps
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.input_layernorm = RMSNorm(hidden_size, rms_eps, dtype=params_dtype)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, rms_eps, dtype=params_dtype
        )
        mix = (2 + hc_mult) * hc_mult
        for block in ("attn", "ffn"):
            for name, shape in (
                ("fn", (mix, hc_mult * hidden_size)),
                ("base", (mix,)),
                ("scale", (3,)),
            ):
                self.register_parameter(
                    f"hc_{block}_{name}",
                    nn.Parameter(
                        torch.empty(shape, dtype=torch.float32), requires_grad=False
                    ),
                )

    def hc_pre(self, x, fn, scale, base):
        return _hc_pre(
            x, fn, scale, base, self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters
        )

    def hc_post(self, x, residual, post, comb):
        return _hc_post(x, residual, post, comb)

    def forward(self, positions, hidden_states, input_ids=None, llama_4_scaling=None):
        residual = hidden_states
        x, post, comb = self.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        x = self.self_attn(
            positions=positions, hidden_states=x, llama_4_scaling=llama_4_scaling
        )
        hidden_states = self.hc_post(x, residual, post, comb)
        residual = hidden_states
        x, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, input_ids=input_ids)
        return self.hc_post(x, residual, post, comb)


def _make_rope_cache(config, max_model_len, *, compressed):
    # YaRN correction uses the original context, not the expanded runtime limit.
    params = config.rope_parameters
    factor = params.get("factor", 1.0)
    rotary = DeepseekScalingRotaryEmbedding(
        head_size=config.qk_rope_head_dim,
        rotary_dim=config.qk_rope_head_dim,
        max_position_embeddings=params.get(
            "original_max_position_embeddings", config.max_position_embeddings
        ),
        base=config.compress_rope_theta if compressed else config.rope_theta,
        is_neox_style=False,
        scaling_factor=factor,
        dtype=torch.float32,
        beta_fast=params.get("beta_fast", 32),
        beta_slow=params.get("beta_slow", 1),
        mscale=0,
        mscale_all_dim=0,
        init_cache=False,
    )
    inv_freq = rotary._compute_inv_freq(factor)
    phases = torch.outer(
        torch.arange(max_model_len, device=inv_freq.device, dtype=torch.float32),
        inv_freq,
    )
    return torch.cat((phases.cos(), phases.sin()), dim=-1)


class DeepseekV4MLP(nn.Module):
    def __init__(self, config, intermediate_size, quant_config, *, prefix):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("DeepSeek V4 requires SiLU activation")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            return_bias=False,
        )
        limit = getattr(config, "swiglu_limit", None)
        self.act_fn = SiluAndMulWithClampFL(limit) if limit else SiluAndMulFL()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            return_bias=False,
            reduce_results=False,
        )

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class DeepseekV4MoE(nn.Module):
    def __init__(self, config, quant_config, *, prefix):
        super().__init__()
        self.is_hash = extract_layer_index(prefix) < config.num_hash_layers
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        if self.is_hash:
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(
                    config.vocab_size, config.num_experts_per_tok, dtype=torch.int32
                ),
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        shared_count = config.n_shared_experts or 0
        shared = (
            DeepseekV4MLP(
                config,
                shared_count * config.moe_intermediate_size,
                quant_config,
                prefix=f"{prefix}.shared_experts",
            )
            if shared_count
            else None
        )
        self.shared_experts = shared
        # The generic runner owns shared-expert addition, reduction and routing
        # scale. Do not repeat any of those operations around its output.
        self.experts = FusedMoEFL(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            params_dtype=torch.bfloat16,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=config.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=getattr(config, "swiglu_limit", None),
            shared_experts=shared,
            gate=self.gate,
            router_logits_dtype=torch.float32,
        )

    def forward(self, x, input_ids=None):
        if self.is_hash and input_ids is None:
            raise ValueError("DeepSeek V4 hash routing requires input IDs")
        logits = x if self.experts.is_internal_router else self.gate(x)[0]
        return self.experts(x, router_logits=logits, input_ids=input_ids)


class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__()
        config = self.config = vllm_config.model_config.hf_config
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("DeepSeek V4 reference model requires BF16")
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.bfloat16,
                prefix=f"{prefix}.embed_tokens",
            )
            if get_pp_group().is_first_rank
            else PPMissingLayer()
        )
        rope_caches = {}

        def make_decoder(prefix):
            compressed = config.compress_ratios[extract_layer_index(prefix)] != 0
            if compressed not in rope_caches:
                rope_caches[compressed] = _make_rope_cache(
                    config,
                    vllm_config.model_config.max_model_len,
                    compressed=compressed,
                )
            attn_prefix = f"{prefix}.self_attn"
            attention = DeepseekV4Attention(
                vllm_config,
                config,
                create_deepseek_v4_cache_layers(
                    config, vllm_config.cache_config, attn_prefix
                ),
                rope_caches[compressed],
                prefix=attn_prefix,
            )
            return DeepseekV4DecoderLayer(
                config.hidden_size,
                config.hc_mult,
                config.hc_sinkhorn_iters,
                config.hc_eps,
                config.rms_norm_eps,
                attention,
                DeepseekV4MoE(config, vllm_config.quant_config, prefix=f"{prefix}.mlp"),
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, make_decoder, prefix=f"{prefix}.layers"
        )
        self.norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=torch.bfloat16)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        # Checkpoints contain these root parameters on every PP rank. Keep
        # their ownership uniform; only the final rank executes the HC head.
        for name, shape in (
            ("fn", (config.hc_mult, config.hc_mult * config.hidden_size)),
            ("base", (config.hc_mult,)),
            ("scale", (1,)),
        ):
            self.register_parameter(
                f"hc_head_{name}",
                nn.Parameter(
                    torch.empty(shape, dtype=torch.float32), requires_grad=False
                ),
            )

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.empty(
                    batch_size,
                    self.hc_mult,
                    self.config.hidden_size,
                    dtype=dtype,
                    device=device,
                )
            }
        )

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if get_pp_group().is_first_rank:
            hidden = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            hidden = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            hidden = intermediate_tensors["hidden_states"]
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden = layer(positions, hidden, input_ids=input_ids)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden})
        hidden = _hc_head(
            hidden,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm_eps,
            self.hc_eps,
        )
        return self.norm(hidden)


class DeepseekV4ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    """Checkpoint-facing envelope; operator implementations remain dispatched."""

    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed_tokens.",
            "head.": "lm_head.",
        },
        orig_to_new_substr={
            ".attn.": ".self_attn.",
            ".w1.": ".gate_proj.",
            ".w2.": ".down_proj.",
            ".w3.": ".up_proj.",
            ".ffn.": ".mlp.",
            ".ffn_norm.": ".post_attention_layernorm.",
            ".attn_norm.": ".input_layernorm.",
        },
    )

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DeepseekV4Model(
            vllm_config=vllm_config, prefix=f"{prefix}.model" if prefix else "model"
        )
        self.lm_head = (
            ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                params_dtype=torch.bfloat16,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.lm_head" if prefix else "lm_head",
            )
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.expert_weights = []
        self.moe_layers = [
            layer.mlp.experts
            for layer in self.model.layers
            if isinstance(layer, DeepseekV4DecoderLayer)
        ]
        self.num_moe_layers = self.config.num_hidden_layers if self.moe_layers else 0
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_logical_experts = self.config.n_routed_experts
        self.num_physical_experts = self.num_logical_experts
        self.num_local_physical_experts = (
            self.moe_layers[0].routed_experts.local_num_experts
            if self.moe_layers
            else 0
        )
        self.num_routed_experts = self.config.n_routed_experts
        self.num_shared_experts = self.config.n_shared_experts or 0
        self.num_redundant_experts = 0

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    @staticmethod
    def _checkpoint_name(name):
        if name.startswith("model."):
            name = name[len("model.") :]
        for source, target in (
            ("layers.", "model.layers."),
            ("embed.", "model.embed_tokens."),
            ("embed_tokens.", "model.embed_tokens."),
            ("head.", "lm_head."),
            ("norm.", "model.norm."),
            ("hc_head_", "model.hc_head_"),
        ):
            if name.startswith(source):
                name = target + name[len(source) :]
                break
        for source, target in (
            (".attn.", ".self_attn."),
            (".ffn.", ".mlp."),
            (".attn_norm.", ".input_layernorm."),
            (".ffn_norm.", ".post_attention_layernorm."),
            (".w1.", ".gate_proj."),
            (".w2.", ".down_proj."),
            (".w3.", ".up_proj."),
            (".gate.bias", ".gate.e_score_correction_bias"),
        ):
            name = name.replace(source, target)
        if name.endswith(".scale"):
            name = name[: -len(".scale")] + ".weight_scale"
        return name

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        expert_mapping = fused_moe_make_expert_params_mapping(
            self, "gate_proj", "down_proj", "up_proj", self.config.n_routed_experts
        )
        for raw_name, tensor in weights:
            clean = raw_name.removeprefix("model.")
            if clean.startswith("mtp.") or clean.endswith("rotary_emb.inv_freq"):
                continue
            if (
                clean.startswith("layers.")
                and int(clean.split(".")[1]) >= self.config.num_hidden_layers
            ):
                continue
            name = self._checkpoint_name(raw_name)
            if is_pp_missing_parameter(name, self):
                continue
            for target, source, expert_id, shard_id in expert_mapping:
                if source not in name:
                    continue
                mapped = name.replace(source, target)
                param = params[mapped]
                success = param.weight_loader(
                    param,
                    tensor,
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded.add(mapped)
                break
            else:
                shard = None
                for source, shard_id in ((".gate_proj.", 0), (".up_proj.", 1)):
                    if source in name:
                        name = name.replace(source, ".gate_up_proj.")
                        shard = shard_id
                        break
                param = params[name]
                if name.endswith(".attn_sink"):
                    tensor = tensor.chunk(
                        get_tensor_model_parallel_world_size(), dim=0
                    )[get_tensor_model_parallel_rank()]
                loader = getattr(param, "weight_loader", default_weight_loader)
                if shard is None:
                    loader(param, tensor)
                else:
                    loader(param, tensor, shard)
                loaded.add(name)
        return loaded

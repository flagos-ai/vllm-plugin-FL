# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 attention composition over dispatch-selected operators.

Cache ownership and scheduling stay outside these model attention modules.
"""

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import extract_layer_index

from vllm_fl.dispatch import CachedOp
from vllm_fl.ops.layernorm import RMSNormFL as RMSNorm

_compressor = CachedOp("deepseek_v4_compressor")
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
                "DeepSeek V4 attention requires non-speculative CP1"
            )
        if ratio not in (0, 4, 128) or params_dtype != torch.bfloat16:
            raise NotImplementedError(
                "DeepSeek V4 attention requires BF16 and C0/C4/C128"
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
        if self.wo_a.weight.dtype != params_dtype:
            configure_grouped = getattr(
                self.wo_a.quant_method, "configure_grouped", None
            )
            if configure_grouped is None:
                from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
                    CompressedTensorsLinearMethod,
                )
                from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
                    CompressedTensorsW8A8Int8,
                )

                if isinstance(
                    self.wo_a.quant_method, CompressedTensorsLinearMethod
                ) and isinstance(self.wo_a.scheme, CompressedTensorsW8A8Int8):
                    from vllm_fl.quantization.w8a8.compressed_tensors import (
                        AscendGroupedW8A8LinearMethod,
                    )

                    self.wo_a.quant_method = AscendGroupedW8A8LinearMethod(
                        self.wo_a.quant_method.quantization_config
                    )
                    configure_grouped = self.wo_a.quant_method.configure_grouped
            if configure_grouped is None:
                raise NotImplementedError(
                    "Grouped output projection requires a grouped quantization method"
                )
            configure_grouped(self.n_local_groups)
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
                "DeepSeek V4 attention does not support Llama4 scaling"
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
        # wo_a applies independent head groups rather than one shared Linear.
        # The INT8 adapter also owns each group's prepared weight and scales.
        if self.wo_a.weight.dtype == torch.int8:
            output = self.wo_a.quant_method.apply_grouped(self.wo_a, output)
        else:
            output = _grouped_project(output, self.wo_a.weight, self.n_local_groups)
        output = self.wo_b(output)
        return torch.where(active[:, None], output, 0)

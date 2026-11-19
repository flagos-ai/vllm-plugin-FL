# Projection/normalization order follows vLLM's Qwen3NextAttention.
"""Prefill attention CP with native TP weights, persistent cache and decode.

Only the prefill attention kernel uses sequence shards. QKV/O, GDN, dense/MoE
FFNs and all persistent states keep the upstream TP contract. Full-head KV
pages are a temporary workspace reconstructed from the active native pages.
"""

from dataclasses import replace

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import unified_kv_cache_update
from vllm.model_executor.layers.attention.kv_transfer_utils import maybe_transfer_kv_layer
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)

from vllm_fl.cp.layout import PagedKVPlan, ZigZagPlan

logger = init_logger(__name__)


def flash_attn_varlen_func(**kwargs):
    # Delay the CUDA extension import so layout/config tests work on CPU.
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func as kernel
    return kernel(**kwargs)


def validate_config(vllm_config):
    """Reject execution modes that change the full-row / native-cache contract."""
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    config = model.hf_text_config
    if config.model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        raise ValueError("FL Qwen CP supports Qwen3.5/3.6 dense and MoE only")
    tp = parallel.tensor_parallel_size
    if tp < 2:
        raise ValueError("FL Qwen CP requires tensor_parallel_size >= 2 (CP = TP)")
    for name in ("num_attention_heads", "linear_num_key_heads", "linear_num_value_heads"):
        if getattr(config, name) % tp:
            raise ValueError(f"{name} must be divisible by tensor_parallel_size")
    kv_heads = config.num_key_value_heads
    if max(tp, kv_heads) % min(tp, kv_heads):
        raise ValueError("KV heads and TP must support native head sharding/replication")
    unsupported = {
        "use enforce_eager=True": not model.enforce_eager,
        "torch.compile": vllm_config.compilation_config.mode not in (None, 0),
        "CUDA Graph": bool(vllm_config.compilation_config.cudagraph_mode),
        "vLLM sequence parallel padding":
            bool(vllm_config.compilation_config.pass_config.enable_sp),
        "sequence parallel MoE": getattr(parallel, "use_sequence_parallel_moe", False),
        "expert parallel": getattr(parallel, "enable_expert_parallel", False),
        "pipeline parallel": parallel.pipeline_parallel_size != 1,
        "data parallel": parallel.data_parallel_size != 1,
        "native prefill CP (leave prefill_context_parallel_size=1)":
            parallel.prefill_context_parallel_size != 1,
        "native decode CP (leave decode_context_parallel_size=1)":
            parallel.decode_context_parallel_size != 1,
        "microbatching / DBO": parallel.use_ubatching,
        "speculative decoding": vllm_config.speculative_config is not None,
        "LoRA": vllm_config.lora_config is not None,
        "quantization": vllm_config.quant_config is not None,
        "quantized KV cache": vllm_config.cache_config.cache_dtype != "auto",
        "prefix caching": vllm_config.cache_config.enable_prefix_caching,
        "sliding window": getattr(config, "sliding_window", None) is not None,
        "dual chunk attention": getattr(config, "dual_chunk_attention_config", None)
            is not None,
    }
    for feature, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"FL Qwen CP v1 does not support {feature}")


class QwenCollocatedCPAttention(Qwen3NextAttention):
    """Native TP layer with a sequence-parallel prefill attention kernel."""

    def __init__(self, config, model_config=None, cache_config=None,
                 quant_config=None, prefix=""):
        vllm_config = get_current_vllm_config()
        validate_config(vllm_config)
        # Preserve upstream parameter names, head loaders, Attention instance,
        # layer name and KVCacheSpec. D can use an entirely unmodified model.
        super().__init__(config, model_config, cache_config, quant_config, prefix)
        if self.attn.get_attn_backend() is not FlashAttentionBackend:
            raise ValueError("FL Qwen CP requires attention_config.backend=FLASH_ATTN")
        if self.attn.impl.vllm_flash_attn_version != 2:
            raise ValueError(
                "FL Qwen CP correctness configuration requires flash_attn_version=2; "
                "FA3 has not passed the long-prompt logprob comparison"
            )
        if (config.model_type == "qwen3_5_moe_text"
                and vllm_config.kernel_config.moe_backend != "triton"):
            raise ValueError("FL Qwen MoE CP requires kernel_config.moe_backend=triton")
        self.cp_group = get_tp_group()
        self.cp_prefill_calls = 0
        self.tp_decode_calls = 0
        self.tp_mixed_calls = 0
        self.tp_short_prefill_calls = 0
        # Keep at least 16 query rows per zigzag block on every rank. Tiny
        # shards add collectives and select different low-precision kernels.
        self.cp_min_query_len = 2 * self.cp_group.world_size * 16
        # The hybrid manager can enlarge attention pages after construction.
        # Read the final block size during profiling, not here.
        self._profile_cache_config = vllm_config.cache_config
        self._profile_max_model_len = vllm_config.model_config.max_model_len
        self._profile_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        logger.info_once(
            "FL Qwen CP: prefill attention CP=%d; QKV/O, GDN, FFN, cache and "
            "decode use native TP=%d; mixed/short-query batches use TP; eager mode",
            self.cp_group.world_size, self.cp_group.world_size,
        )

    def _get_plan(self, context, metadata, num_tokens):
        key = ("fl_qwen_cp", id(metadata), self.cp_group.rank_in_group)
        if key not in context.additional_kwargs:
            starts = [0, num_tokens] if metadata is None else metadata.query_start_loc.cpu().tolist()
            lengths = [num_tokens] if metadata is None else metadata.seq_lens.cpu().tolist()
            if metadata is not None and (starts[-1] != metadata.num_actual_tokens
                                         or num_tokens != metadata.num_actual_tokens):
                raise ValueError("FL Qwen CP requires unpadded eager query metadata")
            # A one-token row may be decode. The whole mixed batch uses native
            # TP so decode never silently switches to redundant/CP attention.
            mixed = any(b - a <= 1 for a, b in zip(starts, starts[1:]))
            plan = ZigZagPlan.build(starts, lengths, self.cp_group.world_size,
                                   self.cp_group.rank_in_group)
            context.additional_kwargs[key] = (plan, lengths, mixed)
        return context.additional_kwargs[key]

    @maybe_transfer_kv_layer
    def _cp_attention(self, q, k, v, plan, lengths, metadata, layer_name):
        # The connector sees the original native cache and original metadata.
        # Its wait runs before the cache write and save after CP computation.
        if metadata is not None:
            unified_kv_cache_update(k, v, layer_name)
        local_q = plan.select(self.cp_group.all_gather(q.contiguous(), dim=1))
        local_output = torch.zeros_like(local_q)

        if metadata is None:
            # Account for the worst active-history workspace in the engine's
            # memory profile. Otherwise a tiny chunk with a long prefix could
            # exceed the free memory left after native KV cache allocation.
            # Includes selected local pages, gathered replicas, unique KV.
            workspace_heads = (self.num_kv_heads * (1 + 2 * self.cp_group.world_size)
                               + self.total_num_kv_heads)
            block_size = self._profile_cache_config.block_size or 16
            num_blocks = ((self._profile_max_model_len + block_size - 1)
                          // block_size * self._profile_max_num_seqs)
            _workspace_reserve = q.new_empty((
                2, num_blocks, block_size,
                workspace_heads, self.head_dim,
            ))
            return plan.gather(local_output, self.cp_group.all_gather)

        cache = self.attn.kv_cache
        block_size = cache.shape[2]  # Logical FA shape, independent of strides.
        context = get_forward_context()
        key = ("fl_qwen_cp_pages", id(metadata), block_size)
        if key not in context.additional_kwargs:
            context.additional_kwargs[key] = PagedKVPlan.build(
                metadata.block_table.cpu().tolist(), lengths, block_size)
        pages = context.additional_kwargs[key]
        # Every rank participates, including ranks with zero query rows.
        full_cache = pages.gather(cache, self.total_num_kv_heads,
                                  self.cp_group.world_size, self.cp_group.all_gather)
        if plan.rows:
            local_meta = plan.metadata(replace(
                metadata, block_table=pages.table(cache.device)))
            flash_attn_varlen_func(
                q=local_q, k=full_cache[0], v=full_cache[1], out=local_output,
                cu_seqlens_q=local_meta.query_start_loc,
                max_seqlen_q=local_meta.max_query_len,
                seqused_k=local_meta.seq_lens, max_seqlen_k=local_meta.max_seq_len,
                block_table=local_meta.block_table, softmax_scale=self.scaling,
                causal=True, fa_version=self.attn.impl.vllm_flash_attn_version,
                scheduler_metadata=None,
                num_splits=metadata.max_num_splits,
            )
        return plan.gather(local_output, self.cp_group.all_gather)

    def forward(self, positions, output, hidden_states):
        context = get_forward_context()
        metadata_map = context.attn_metadata
        if metadata_map is not None and not isinstance(metadata_map, dict):
            raise ValueError("FL Qwen CP requires a single eager forward batch")
        metadata = None if metadata_map is None else metadata_map[self.attn.layer_name]
        if metadata is not None:
            if not isinstance(metadata, FlashAttentionMetadata) or not metadata.causal:
                raise ValueError("FL Qwen CP requires causal FlashAttention metadata")
            if metadata.max_query_len <= 1:
                self.tp_decode_calls += 1
                return super().forward(positions, output, hidden_states)

        plan, lengths, mixed = self._get_plan(context, metadata, hidden_states.shape[0])
        if metadata is not None and mixed:
            self.tp_mixed_calls += 1
            return super().forward(positions, output, hidden_states)
        if metadata is not None:
            starts = metadata.query_start_loc.cpu().tolist()
            if any(b - a < self.cp_min_query_len for a, b in zip(starts, starts[1:])):
                self.tp_short_prefill_calls += 1
                return super().forward(positions, output, hidden_states)
        if metadata is not None:
            self.cp_prefill_calls += 1

        # Native TP QKV projection over the full ordered sequence. Gather Q
        # heads only after normalization and RoPE, then distribute query rows.
        qkv, _ = self.qkv_proj(hidden_states)
        gate = None
        if self.attn_output_gate:
            q_gate, k, v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], -1)
            q, gate = q_gate.reshape(-1, self.num_heads, 2 * self.head_dim).chunk(2, -1)
            gate = gate.reshape(-1, self.q_size)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary_emb(positions, q.reshape(-1, self.q_size),
                               k.reshape(-1, self.kv_size))
        full_output = self._cp_attention(
            q.reshape(-1, self.num_heads, self.head_dim),
            k.reshape(-1, self.num_kv_heads, self.head_dim),
            v.reshape(-1, self.num_kv_heads, self.head_dim),
            plan, lengths, metadata, self.attn.layer_name,
        )
        start = self.cp_group.rank_in_group * self.num_heads
        tp_output = full_output[:, start:start + self.num_heads].reshape(-1, self.q_size)
        if gate is not None:
            tp_output = tp_output * torch.sigmoid(gate)
        output[:], _ = self.o_proj(tp_output)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""Only the changed MiniMax methods; unchanged model plumbing stays in vLLM."""

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.minimax_m3.nvidia import model as native

from vllm_fl.minimax_m3_config import load
from vllm_fl.ops.minimax_m3.expert_layer import FusedMoEFL as FusedMoE
from vllm_fl.ops.minimax_m3.layers import qknorm_rope_insert as _m3_qkv
from vllm_fl.ops.minimax_m3.router import M3GateLinear as GateLinear
from vllm_fl.ops.minimax_m3.routing import FusedTopKBiasRouterFL

MiniMaxM3MLP = native.MiniMaxM3MLP


class MiniMAXGemmaRMSNorm(native.MiniMAXGemmaRMSNorm):
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if load().framework.flagos_norm:
            from vllm_fl.ops.minimax_m3.layers import gemma

            return gemma(x, self.weight, self.variance_epsilon, residual)
        from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

        if residual is None:
            return gemma_rmsnorm(x, self.weight, self.variance_epsilon)

        # gemma_fused_add_rmsnorm mutates x and residual in place.
        gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
        return x, residual


class MiniMaxM3MoE(native.MiniMaxM3MoE):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)

        # Sigmoid routing uses a per-expert score-correction bias for selection.
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                MiniMaxM3MoE.ebias_weight_loader
            )
        else:
            self.e_score_correction_bias = None

        # Router weights are stored in fp32; GateLinear upcasts the bf16
        # activations and computes the gate in fp32 (fp32 router logits).
        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts: MiniMaxM3MLP | None = None
        if self.n_shared_experts:
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * self.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        # The accepted model has correction bias and output-side routed scaling.
        assert self.e_score_correction_bias is not None
        router = FusedTopKBiasRouterFL(
            top_k=config.num_experts_per_tok,
            global_num_experts=config.num_local_experts,
            e_score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
            routed_scaling_factor=1.0,
            scoring_func=config.scoring_func,
        )
        self.experts = FusedMoE(
            router=router,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
            # w13 (gate_up_proj) is loaded packed via MergedColumnParallelLinear
            # ([all gates; all ups]), so use the uninterleaved SwiGLU-OAI variant
            # rather than the interleaved gpt-oss layout.
            activation="swigluoai_uninterleave",
            swiglu_limit=config.swiglu_limit,
            swiglu_alpha=config.swiglu_alpha,
            swiglu_beta=config.swiglu_beta,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            router_logits_dtype=self.gate.out_dtype,
            shared_experts=self.shared_experts,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )


class MiniMaxM3Attention(native.MiniMaxM3Attention):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        # Fused per-head Gemma QK-norm + partial NeoX RoPE on q/k, in place.

        _m3_qkv(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            kv_cache_dtype="auto",
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3SparseAttention(native.MiniMaxM3SparseAttention):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Single fused projection emitting [q | k | v | index_q | index_k].
        qkv, _ = self.qkv_proj(hidden_states)

        # Horizontally-fused per-head Gemma QK-norm + partial NeoX RoPE on the
        # main (q/k) and index (index_q/index_k) branches, all read straight out
        # of the single fused ``qkv`` tensor (the "5 results").  Once the paged
        # caches are bound the kernel also inserts k/v and the index key into
        # them; the initial memory-profiling run (caches unbound, no slot_mapping)
        # short-circuits to zeros below. k/v and index_k are rewritten in place
        # inside qkv (and scatter-inserted into the caches); q and index_q are
        # de-interleaved
        # straight into the dedicated contiguous ``q``/``index_q`` buffers below.

        cos_sin_cache = self.rotary_emb.cos_sin_cache
        rotary_dim = self.rotary_emb.rotary_dim
        eps = self.q_norm.variance_epsilon
        num_tokens = qkv.shape[0]

        fwd_slot_mapping = get_forward_context().slot_mapping
        if (
            not isinstance(fwd_slot_mapping, dict)
            or self.layer_name not in fwd_slot_mapping
        ):
            # Memory-profiling run: caches not yet bound, slot_mapping is empty.
            return qkv.new_zeros((num_tokens, self.hidden_size))

        main_slot_mapping = fwd_slot_mapping[self.layer_name]
        index_slot_mapping = fwd_slot_mapping[self.indexer.index_cache.prefix]
        q = qkv.new_empty((num_tokens, self.q_size))
        # index_q matches the index-K cache dtype (e4m3 for the fp8 score path);
        # the fused kernel emits fp8 directly when this buffer is e4m3.
        index_q = qkv.new_empty(
            (num_tokens, self.index_q_size),
            dtype=self.indexer.index_cache.dtype,
        )
        _m3_qkv(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            rotary_dim,
            eps,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            self.num_idx_heads,
            main_slot_mapping,
            index_slot_mapping,
            self.kv_cache,
            self.indexer.index_cache.kv_cache,
            self.kv_cache.size(2),  # paged-cache block size
            q,
            index_q,
            self.kv_cache_dtype,
        )

        output = torch.empty_like(q)
        attn_output = self._run_attention(q, index_q, output)
        output, _ = self.o_proj(attn_output)
        return output

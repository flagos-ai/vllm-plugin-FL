# SPDX-License-Identifier: Apache-2.0
"""Inference-only GLM-5 (GlmMoeDsa) model.

GLM-5 uses a DeepSeek V2/V3-style architecture with MLA (Multi-head Latent
Attention) and Mixture of Experts.  The HF model type is ``glm_moe_dsa`` and
the architecture class is ``GlmMoeDsaForCausalLM``.

This thin wrapper inherits from vLLM's ``DeepseekV2ForCausalLM`` which already
handles MLA and MoE.  The DSA (Dynamic Sparse Attention) indexer requires
deep_gemm FP8 kernels; when deep_gemm is unavailable, we disable the indexer
by temporarily hiding the ``index_topk`` config attribute during construction.
"""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    Indexer,
)
from vllm.utils.import_utils import has_deep_gemm


def _patched_indexer_forward(
    self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
) -> torch.Tensor:
    """Fixed Indexer.forward that handles RoPE output dimensions correctly."""
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)
    q_pe, q_nope = torch.split(
        q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
    )

    k, _ = self.wk(hidden_states)
    k = self.k_norm(k)
    k_pe, k_nope = torch.split(
        k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
    )

    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
    # RoPE can introduce extra leading dimensions during compilation,
    # so reshape back to token-flattened shapes.
    q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
    k_pe = k_pe.reshape(-1, 1, self.rope_dim)

    q = torch.cat([q_pe, q_nope], dim=-1)
    k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

    # We only quant q here since k quant is fused with cache insertion.
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    q = q.view(-1, self.head_dim)
    q_fp8, q_scale = per_token_group_quant_fp8(
        q,
        self.quant_block_size,
        column_major_scales=False,
        use_ue8m0=self.scale_fmt is not None,
    )
    q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
    q_scale = q_scale.view(-1, self.n_head, 1)

    weights, _ = self.weights_proj(hidden_states)
    weights = (
        weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
    )
    weights = weights.squeeze(-1)

    return torch.ops.vllm.sparse_attn_indexer(
        hidden_states,
        self.k_cache.prefix,
        self.k_cache.kv_cache[0],
        q_fp8,
        k,
        weights,
        self.quant_block_size,
        self.scale_fmt,
        self.topk_tokens,
        self.head_dim,
        self.max_model_len,
        self.max_total_seq_len,
        self.topk_indices_buffer,
    )

def patch_is_deepseek_mla():
    """Patch ``ModelConfig.is_deepseek_mla`` to recognise ``glm_moe_dsa``."""
    from vllm.config.model import ModelConfig

    _orig = ModelConfig.is_deepseek_mla.fget

    @property  # type: ignore[misc]
    def _patched(self):
        if (
            hasattr(self.hf_text_config, "model_type")
            and self.hf_text_config.model_type == "glm_moe_dsa"
            and getattr(self.hf_text_config, "kv_lora_rank", None) is not None
        ):
            return True
        return _orig(self)

    ModelConfig.is_deepseek_mla = _patched

# Monkey-patch the Indexer.forward to fix dimension mismatch in the
# installed vLLM 0.13.0.
Indexer.forward = _patched_indexer_forward


class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    """GLM-5 model for causal language modelling."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config

        # The DSA indexer requires deep_gemm FP8 MQA kernels.
        # When deep_gemm is not available, disable the indexer by
        # temporarily removing the index_topk attribute so that
        # DeepseekV2Attention skips indexer construction.
        _saved_index_topk = getattr(config, "index_topk", None)
        self._indexer_disabled = False
        if _saved_index_topk is not None and not has_deep_gemm():
            delattr(config, "index_topk")
            self._indexer_disabled = True

        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            # Restore the config attribute
            if _saved_index_topk is not None and not hasattr(config, "index_topk"):
                config.index_topk = _saved_index_topk

    def load_weights(self, weights):
        # When the DSA indexer is disabled, the model has no indexer
        # parameters, but the checkpoint still contains them.
        # Filter them out to avoid KeyError during weight loading.
        if self._indexer_disabled:
            weights = (
                (name, weight) for name, weight in weights
                if ".indexer." not in name
            )
        return super().load_weights(weights)

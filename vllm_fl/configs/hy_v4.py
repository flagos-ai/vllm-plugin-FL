# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the HY4 preview model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import PretrainedConfig

# HY4 preview (``tencent/Hy4-preview`` and ``tencent/Hy4-preview-FP8``) is a
# text-only MoE language model.  The official configs have no ``vision_config``,
# image-token ids, processor, or vision/projector weights, and upstream vLLM's
# ``hy_v4`` implementation is a plain ``SupportsPP`` causal LM.  The plugin
# therefore serves the text path only and must fail fast if a checkpoint ever
# advertises a vision tower, rather than silently dropping those parameters.
_MULTIMODAL_KEY_PREFIXES = ("vision_", "image_", "video_", "audio_", "pixel_")
_MULTIMODAL_KEY_EXACT = frozenset(
    {
        "multi_modal_projector",
        "mm_projector",
        "mm_projector_type",
        "vision_tower",
    }
)


def _unsupported_multimodal_keys(mapping: Mapping[str, Any]) -> list[str]:
    """Return config keys that advertise an unsupported multimodal tower."""
    rejected = [
        key
        for key in mapping
        if key in _MULTIMODAL_KEY_EXACT or key.startswith(_MULTIMODAL_KEY_PREFIXES)
    ]
    return sorted(rejected)


class HYV4Config(PretrainedConfig):
    """Configuration bridge for checkpoints with ``model_type=hy_v4``."""

    model_type = "hy_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Explicit capability contract consumed by vLLM's multimodal registry.
    supports_multimodal = False

    def __init__(
        self,
        vocab_size: int = 120_832,
        hidden_size: int = 6_144,
        intermediate_size: int = 18_432,
        num_hidden_layers: int = 78,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1_048_576,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.006,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        moe_intermediate_size: int = 2_048,
        num_experts_per_tok: int = 8,
        routed_scaling_factor: float = 2.827,
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int = 2_048,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        qk_head_dim: int | None = 256,
        v_head_dim: int = 256,
        index_topk: int = 2_048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        hc_mult: int = 4,
        hc_magnitude: float = 2.0,
        hc_eps: float = 1e-6,
        swiglu_limit: float = 10.0,
        rope_parameters: dict[str, Any] | None = None,
        mlp_layer_types: list[str] | None = None,
        layer_types: list[str] | None = None,
        indexer_types: list[str] | None = None,
        enable_lm_head_fp32: bool = True,
        enable_ihc: bool = True,
        gated_mla: bool = True,
        gating_type: str = "elementwise",
        learnable_sink: bool = True,
        learnable_sink_init: float = 0.0,
        num_nextn_predict_layers: int = 1,
        mtp_loss_factor: float = 0.1,
        bos_token_id: int = 120_000,
        eos_token_id: int = 120_025,
        pad_token_id: int = 120_002,
        **kwargs: Any,
    ) -> None:
        multimodal_keys = _unsupported_multimodal_keys(kwargs)
        if multimodal_keys:
            raise ValueError(
                "HY4 preview is a text-only checkpoint family; multimodal "
                "config keys are not supported: "
                f"{multimodal_keys}. Use the text-only HY4 checkpoint "
                "(tencent/Hy4-preview) or a dedicated multimodal model."
            )
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim or qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.hc_mult = hc_mult
        self.hc_magnitude = hc_magnitude
        self.hc_eps = hc_eps
        self.swiglu_limit = swiglu_limit
        self.rope_parameters = rope_parameters or {
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
        }
        self.mlp_layer_types = mlp_layer_types or ["dense"] + ["sparse"] * (
            num_hidden_layers - 1
        )
        self.layer_types = (
            layer_types or ["deepseek_sparse_attention"] * num_hidden_layers
        )
        self.indexer_types = indexer_types or ["full"] * num_hidden_layers
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.enable_ihc = enable_ihc
        self.gated_mla = gated_mla
        self.gating_type = gating_type
        self.learnable_sink = learnable_sink
        self.learnable_sink_init = learnable_sink_init
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_factor = mtp_loss_factor

        # Compatibility attributes consumed by vLLM's no-aux MoE router.
        self.first_k_dense_replace = 1
        self.moe_layer_freq = 1
        self.topk_method = "noaux_tc"
        self.scoring_func = "sigmoid"
        self._validate_hy4()

    @property
    def rope_theta(self) -> float:
        """Return the RoPE base in the legacy Transformers form."""
        return float(self.rope_parameters.get("rope_theta", 10_000_000.0))

    @rope_theta.setter
    def rope_theta(self, value: float) -> None:
        """Accept vLLM's normalized legacy RoPE assignment."""
        self.rope_parameters["rope_theta"] = float(value)

    def _validate_hy4(self) -> None:
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.n_routed_experts <= 0:
            raise ValueError("n_routed_experts must be positive")
        if self.qk_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim:
            raise ValueError("qk_head_dim must equal no-PE plus RoPE dimensions")
        if self.hc_mult <= 0:
            raise ValueError("hc_mult must be positive")
        for name in ("mlp_layer_types", "layer_types", "indexer_types"):
            values = getattr(self, name)
            if len(values) != self.num_hidden_layers:
                raise ValueError(
                    f"{name} has {len(values)} entries; "
                    f"expected {self.num_hidden_layers}"
                )


__all__ = ["HYV4Config"]

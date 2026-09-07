# SPDX-License-Identifier: Apache-2.0
"""Transformers config classes for GLM5-Next.

These classes preserve the released checkpoint's nested text/vision schema
and expose the aliases consumed by vLLM's KDA, MLA, MoE, and multimodal paths.
"""

from transformers.configuration_utils import PretrainedConfig


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-5,
        head_dim: int | None = None,
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        mla_use_nope: bool = True,
        n_shared_experts: int = 1,
        n_routed_experts: int | None = 288,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        first_k_dense_replace: int = 3,
        moe_layer_freq: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        moe_router_dtype: str = "float32",
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        linear_attn_config: dict | None = None,
        index_topk: int | None = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        indexer_rope_interleave: bool = True,
        index_kpool: int = 4,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
        mhc: bool = True,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        mhc_tau: float = 0.05,
        mhc_no_norm_weight: bool = False,
        mhc_post_mult_value: float = 2.0,
        swiglu_limit: float | None = 10.0,
        num_nextn_predict_layers: int = 1,
        rope_parameters: dict | None = None,
        pad_token_id: int | None = 154820,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default"}
        if linear_attn_config is None:
            linear_attn_config = {
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            }

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # GLM5-Next uses an explicit zero to represent its NoPE layout.
        # Only fall back to the conventional per-head width when the field is
        # absent; treating zero as false silently turns the real config into a
        # 64-dimensional RoPE layout.
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_nope = mla_use_nope
        self.mla = True
        self.rope_parameters = rope_parameters

        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.n_group = n_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.moe_router_dtype = moe_router_dtype

        # Aliases used by KimiLinear's v0.24 KDA/MoE implementation.
        self.num_experts = n_routed_experts
        self.num_shared_experts = n_shared_experts
        self.num_experts_per_token = num_experts_per_tok
        self.moe_renormalize = norm_topk_prob
        self.use_grouped_topk = True
        self.num_expert_group = n_group
        self.moe_router_activation_func = scoring_func

        self.layer_types = layer_types or [
            "deepseek_sparse_attention" if (i + 1) % 4 == 0
            else "linear_attention"
            for i in range(num_hidden_layers)
        ]
        self.mlp_layer_types = mlp_layer_types or (
            ["dense"] * first_k_dense_replace
            + ["sparse"] * (num_hidden_layers - first_k_dense_replace)
        )
        self.linear_attn_config = linear_attn_config
        self.linear_num_heads = linear_attn_config["num_heads"]
        self.linear_head_dim = linear_attn_config["head_dim"]
        self.linear_conv_kernel_dim = linear_attn_config[
            "short_conv_kernel_size"
        ]
        self.linear_lower_bound = linear_attn_config.get("gate_lower_bound")

        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.indexer_rope_interleave = indexer_rope_interleave
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool_always_select_tail = index_kpool_always_select_tail

        self.mhc = mhc
        self.hc_mult = hc_mult
        self.mhc_num_residual_streams = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.mhc_sinkhorn_iterations = hc_sinkhorn_iters
        self.mhc_tau = mhc_tau
        self.mhc_no_norm_weight = mhc_no_norm_weight
        self.mhc_post_mult_value = mhc_post_mult_value
        self.swiglu_limit = swiglu_limit
        self.num_nextn_predict_layers = num_nextn_predict_layers
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            **kwargs,
        )

    @property
    def is_moe(self) -> bool:
        return self.n_routed_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return any(t == "linear_attention" for t in self.layer_types)

    def is_kda_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "linear_attention"

    @property
    def layers_block_type(self) -> list[str]:
        return [
            "linear_attention" if t == "linear_attention" else "attention"
            for t in self.layer_types
        ]


class Glm5NextVisionConfig(PretrainedConfig):
    model_type = "glm5_next_vision"
    base_config_key = "vision_config"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {
        "text_config": Glm5NextTextConfig,
        "vision_config": Glm5NextVisionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.text_config = (
            Glm5NextTextConfig(**text_config)
            if isinstance(text_config, dict)
            else text_config or Glm5NextTextConfig(**kwargs)
        )
        self.vision_config = (
            Glm5NextVisionConfig(**vision_config)
            if isinstance(vision_config, dict)
            else vision_config or Glm5NextVisionConfig()
        )
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

    def get_text_config(self, decoder: bool = False):
        del decoder
        # Transformers 5.12 validates token IDs from PretrainedConfig.__init__,
        # before this class has constructed the nested text config.
        return getattr(self, "text_config", self)

    def __getattr__(self, key):
        # Several vLLM 0.24 MLA metadata builders still read text attributes
        # from hf_config rather than hf_text_config.
        text_config = self.__dict__.get("text_config")
        if text_config is not None and hasattr(text_config, key):
            return getattr(text_config, key)
        raise AttributeError(key)

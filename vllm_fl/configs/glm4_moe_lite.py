# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7-Flash (Glm4MoeLite) config bridge for vLLM plugin.

transformers 4.57.6 does not recognise model_type ``glm4_moe_lite``.
This config bridge lets vLLM load the HuggingFace checkpoint without
upgrading transformers.

GLM-4.7-Flash uses MLA (Multi-head Latent Attention) from DeepSeek V2/V3
and MoE architecture from GLM-4 MoE.
"""

from transformers.models.glm4_moe import Glm4MoeConfig


class Glm4MoeLiteConfig(Glm4MoeConfig):
    model_type = "glm4_moe_lite"

    def __init__(
        self,
        # MLA (Multi-head Latent Attention) fields
        qk_nope_head_dim: int = 0,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 0,
        kv_lora_rank: int = 0,
        q_lora_rank: int | None = None,
        # DSA Indexer fields (v32 variant)
        index_topk: int | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        indexer_rope_interleave: bool | None = None,
        # MoE frequency
        moe_layer_freq: int = 1,
        # MTP (speculative decoding)
        num_nextn_predict_layers: int = 0,
        # Additional fields
        dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # MLA fields
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        # DSA Indexer fields
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.indexer_rope_interleave = indexer_rope_interleave
        # MoE frequency
        self.moe_layer_freq = moe_layer_freq
        # MTP fields
        self.num_nextn_predict_layers = num_nextn_predict_layers
        # Additional
        self.dtype = dtype

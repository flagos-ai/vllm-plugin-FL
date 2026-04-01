# SPDX-License-Identifier: Apache-2.0
"""GLM-5 (GlmMoeDsa) config bridge for vLLM plugin.

transformers 4.57.6 does not recognise model_type ``glm_moe_dsa``.
This config bridge lets vLLM load the HuggingFace checkpoint without
upgrading transformers.
"""

from transformers import DeepseekV2Config


class GlmMoeDsaConfig(DeepseekV2Config):
    model_type = "glm_moe_dsa"

    def __init__(
        self,
        # GLM-5-specific fields (DSA Indexer)
        index_topk=2048,
        index_n_heads=32,
        index_head_dim=128,
        indexer_rope_interleave=True,
        # MTP (speculative decoding)
        num_nextn_predict_layers=1,
        # MoE extras
        moe_layer_freq=1,
        scoring_func="sigmoid",
        ep_size=1,
        # Additional fields
        head_dim=None,
        rope_parameters=None,
        dtype="bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.indexer_rope_interleave = indexer_rope_interleave
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.moe_layer_freq = moe_layer_freq
        self.scoring_func = scoring_func
        self.ep_size = ep_size
        if head_dim is not None:
            self.head_dim = head_dim
        if rope_parameters is not None:
            self.rope_theta = rope_parameters.get(
                "rope_theta", getattr(self, "rope_theta", 10000.0)
            )
        self.dtype = dtype

"""HY4 multimodal capability contract.

HY4 preview (``tencent/Hy4-preview``, ``tencent/Hy4-preview-FP8``) is a
text-only checkpoint family: neither released config contains a
``vision_config``, image-token id, image processor, or vision/projector
weights, and upstream vLLM's ``hy_v4`` model is a plain ``SupportsPP`` causal
LM.  These tests pin that contract so a future vision-bearing checkpoint fails
fast instead of being silently served with its vision tower dropped.
"""

import pytest

from vllm.model_executor.models.interfaces import supports_multimodal

from vllm_fl.configs.hy_v4 import HYV4Config, _unsupported_multimodal_keys
from vllm_fl.models.hy_v4 import HYV4ForCausalLM

# Keys of the official ``tencent/Hy4-preview`` config.json (vLLM 0.24 ABI).
# Source: https://huggingface.co/tencent/Hy4-preview/raw/main/config.json
_OFFICIAL_HY4_CONFIG_KEYS = (
    "architectures",
    "attention_bias",
    "attention_dropout",
    "bitwise_backward_align",
    "bos_token_id",
    "dtype",
    "enable_ihc",
    "enable_lm_head_fp32",
    "eos_token_id",
    "gated_mla",
    "gating_type",
    "hc_eps",
    "hc_magnitude",
    "hc_mult",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
    "indexer_types",
    "initializer_range",
    "intermediate_size",
    "kv_lora_rank",
    "layer_types",
    "learnable_sink",
    "learnable_sink_init",
    "max_position_embeddings",
    "mlp_layer_types",
    "model_type",
    "moe_intermediate_size",
    "mtp_loss_factor",
    "n_group",
    "n_routed_experts",
    "n_shared_experts",
    "norm_topk_prob",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_nextn_predict_layers",
    "pad_token_id",
    "q_lora_rank",
    "qk_head_dim",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "rms_norm_eps",
    "rope_parameters",
    "routed_scaling_factor",
    "swiglu_limit",
    "tie_word_embeddings",
    "topk_group",
    "torch_dtype",
    "transformers_version",
    "use_cache",
    "use_dsa",
    "use_mla",
    "v_head_dim",
    "vocab_size",
)


def test_official_hy4_config_declares_no_multimodal_tower():
    assert _unsupported_multimodal_keys(dict.fromkeys(_OFFICIAL_HY4_CONFIG_KEYS)) == []


def test_hy4_config_constructs_without_multimodal_attributes():
    config = HYV4Config(num_hidden_layers=4)

    assert config.supports_multimodal is False
    assert getattr(config, "vision_config", None) is None
    assert getattr(config, "image_token_id", None) is None
    assert getattr(config, "mm_projector", None) is None


@pytest.mark.parametrize(
    "multimodal_key",
    [
        "vision_config",
        "image_token_id",
        "image_token_index",
        "vision_start_token_id",
        "mm_projector",
        "multi_modal_projector",
        "pixel_values",
    ],
)
def test_hy4_config_rejects_multimodal_checkpoint_keys(multimodal_key):
    with pytest.raises(ValueError, match="text-only checkpoint family"):
        HYV4Config(num_hidden_layers=2, **{multimodal_key: {"kind": "stub"}})


def test_hy4_language_model_is_registered_as_text_only():
    assert HYV4ForCausalLM.supports_multimodal is False
    assert supports_multimodal(HYV4ForCausalLM) is False


def test_multimodal_key_detector_ignores_text_keys():
    text_keys = {"use_mla": True, "use_dsa": True, "head_dim": 64, "imagination": 1}
    assert _unsupported_multimodal_keys(text_keys) == []

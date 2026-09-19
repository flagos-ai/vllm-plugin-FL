"""Single-GPU runtime smoke for the HY4 vLLM 0.24 plugin path."""

import json

import pytest

from vllm import LLM, SamplingParams


@pytest.mark.gpu
def test_hy_v4_plugin_dummy_generate(tmp_path):
    config = {
        "architectures": ["HYV4ForCausalLM"],
        "model_type": "hy_v4",
        "dtype": "bfloat16",
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 128,
        "num_experts_per_tok": 2,
        "routed_scaling_factor": 2.827,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "q_lora_rank": 32,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "index_topk": 128,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "hc_mult": 4,
        "hc_magnitude": 2.0,
        "hc_eps": 1e-6,
        "swiglu_limit": 10.0,
        "rope_parameters": {
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
        },
        "mlp_layer_types": ["dense", "sparse"],
        "layer_types": [
            "deepseek_sparse_attention",
            "deepseek_sparse_attention",
        ],
        "indexer_types": ["full", "shared"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    llm = LLM(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=32,
        gpu_memory_utilization=0.2,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": [1, 7, 11, 13]}],
        SamplingParams(temperature=0.0, max_tokens=4),
    )
    assert len(outputs[0].outputs[0].token_ids) == 4

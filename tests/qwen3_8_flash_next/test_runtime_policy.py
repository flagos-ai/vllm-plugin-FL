"""Core model configuration and standard FlagGems list contracts."""

from types import SimpleNamespace

import pytest

from vllm_fl.patches.qwen3_8_flash_next import Qwen3_8FlashNextForCausalLMConfig
from vllm_fl.utils import get_flag_gems_whitelist_blacklist


@pytest.mark.parametrize(
    "architecture,vendor,whitelist,blacklist,expected_whitelist,expected_blacklist",
    [
        ("Qwen4ExpForCausalLM", "nvidia", "", "", None, None),
        ("Qwen4ExpForCausalLM", "amd", "", "", None, ["copy_"]),
        ("LlamaForCausalLM", "nvidia", "", "", None, ["copy_"]),
        ("Qwen4ExpForCausalLM", "nvidia", "mm", "", ["mm"], None),
        ("Qwen4ExpForCausalLM", "nvidia", "", "add", None, ["add"]),
    ],
)
def test_standard_lists_respect_model_vendor_and_user_overrides(
    monkeypatch,
    architecture,
    vendor,
    whitelist,
    blacklist,
    expected_whitelist,
    expected_blacklist,
):
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    monkeypatch.setitem(
        MODELS_CONFIG_MAP, "Qwen4ExpForCausalLM", Qwen3_8FlashNextForCausalLMConfig
    )
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", whitelist)
    monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", blacklist)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", raising=False)
    monkeypatch.setattr(
        "vllm_fl.dispatch.config.get_flagos_blacklist", lambda: ["copy_"]
    )
    config = SimpleNamespace(model_config=SimpleNamespace(architectures=[architecture]))
    actual_whitelist, actual_blacklist = get_flag_gems_whitelist_blacklist(
        vllm_config=config,
        vendor_name=vendor,
    )
    assert actual_whitelist == expected_whitelist
    if (
        architecture.startswith("Qwen")
        and vendor == "nvidia"
        and not (whitelist or blacklist)
    ):
        assert "mm" in actual_blacklist
        assert "sort_stable" in actual_blacklist
        assert "silu" not in actual_blacklist
        assert "conv1d" not in actual_blacklist
        assert actual_blacklist.count("copy_") == 1
        assert len(actual_blacklist) == len(set(actual_blacklist))
    else:
        assert actual_blacklist == expected_blacklist


@pytest.mark.parametrize("async_scheduling", [None, True])
def test_gpu_history_preserves_async_choice(async_scheduling):
    text_config = SimpleNamespace(
        hc_count=4,
        ple_layer_ids=[1],
        mamba_ssm_dtype="float32",
        indexer_n_heads=None,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text_config,
            hf_config=text_config,
            multimodal_config=None,
        ),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            enable_dbo=False, ubatch_size=1, pipeline_parallel_size=1
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    Qwen3_8FlashNextForCausalLMConfig.verify_and_update_config(config)
    assert config.scheduler_config.async_scheduling is async_scheduling

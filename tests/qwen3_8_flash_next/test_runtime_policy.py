"""Model-scoped FlagGems policy tests for Qwen3.8-Flash-Next."""

from types import SimpleNamespace

import pytest

from vllm_fl.patches.qwen3_8_flash_next import (
    apply_native_index_select_policy,
    needs_native_index_select,
)


def _config(model_type: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
    )


def test_qwen4_merges_native_index_select_with_platform_blacklist():
    config = _config("qwen4_exp_text")
    assert needs_native_index_select(config)
    whitelist, blacklist = apply_native_index_select_policy(
        config, None, ["copy_", "index"]
    )
    assert whitelist is None
    assert blacklist == ["copy_", "index", "index_select"]


def test_policy_is_idempotent_and_preserves_explicit_whitelist():
    config = _config("qwen3_8_flash_next_text")
    _, blacklist = apply_native_index_select_policy(config, None, ["index_select"])
    assert blacklist == ["index_select"]
    whitelist, blacklist = apply_native_index_select_policy(
        config, ["add"], ["copy_"]
    )
    assert whitelist == ["add"]
    assert blacklist == ["copy_"]


def test_policy_rejects_unsafe_explicit_whitelist():
    config = _config("qwen4_exp_text")
    with pytest.raises(ValueError, match="requires native PLE state row I/O"):
        apply_native_index_select_policy(config, ["add", "index_select"], ["copy_"])


def test_policy_does_not_change_other_models():
    config = _config("qwen3_text")
    assert not needs_native_index_select(config)
    whitelist, blacklist = apply_native_index_select_policy(config, None, ["copy_"])
    assert whitelist is None
    assert blacklist == ["copy_"]

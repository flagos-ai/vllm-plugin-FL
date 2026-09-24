# SPDX-License-Identifier: Apache-2.0
"""The worker resolves model FlagGems policy through a generic factory instead
of importing a specific model integration."""

import inspect
from types import SimpleNamespace

from vllm_fl import flaggems_policy
from vllm_fl.patches import qwen3_8_flash_next
from vllm_fl.worker import worker as worker_module


def _config(model_type: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
    )


def test_worker_has_no_model_specific_import():
    source = inspect.getsource(worker_module)
    assert "qwen3_8_flash_next" not in source
    assert "resolve_flag_gems_policy" in source


def test_qwen_provider_matches_only_qwen():
    flaggems_policy.register_flag_gems_policy_provider(
        qwen3_8_flash_next._qwen3_8_flash_next_flag_gems_policy
    )

    qwen = flaggems_policy.resolve_flag_gems_policy(
        _config("qwen4_exp_text"), None, ["copy_"], vendor_name="nvidia"
    )
    assert qwen.blacklist[0] == "copy_"
    assert "index_select" in qwen.blacklist
    assert qwen.skip_generic_aten is True
    assert any("PLE" in message for message in qwen.log_messages)

    other = flaggems_policy.resolve_flag_gems_policy(
        _config("llama"), None, ["copy_"], vendor_name="nvidia"
    )
    assert other.blacklist == ["copy_"]
    assert other.whitelist is None
    assert other.skip_generic_aten is False
    assert other.log_messages == ()


def test_provider_chain_merges_decisions():
    def provider_one(config, whitelist, blacklist, *, vendor_name=None):
        del config, whitelist, vendor_name
        return flaggems_policy.FlagGemsModelPolicy(
            whitelist=None, blacklist=[*(blacklist or []), "x"]
        )

    def provider_skip(config, whitelist, blacklist, *, vendor_name=None):
        del config, whitelist, blacklist, vendor_name
        return flaggems_policy.FlagGemsModelPolicy(
            whitelist=None, blacklist=None, skip_generic_aten=True, log_messages=("m",)
        )

    def provider_none(config, whitelist, blacklist, *, vendor_name=None):
        del config, whitelist, blacklist, vendor_name
        return None

    policy = flaggems_policy.resolve_flag_gems_policy(
        None, None, ["a"], providers=(provider_one, provider_none, provider_skip)
    )
    assert policy.blacklist == ["a", "x"]
    assert policy.skip_generic_aten is True
    assert policy.log_messages == ("m",)


def test_unregister_is_idempotent():
    def provider(config, whitelist, blacklist, *, vendor_name=None):
        return None

    flaggems_policy.register_flag_gems_policy_provider(provider)
    flaggems_policy.unregister_flag_gems_policy_provider(provider)
    flaggems_policy.unregister_flag_gems_policy_provider(provider)
    assert provider not in flaggems_policy.iter_flag_gems_policy_providers()

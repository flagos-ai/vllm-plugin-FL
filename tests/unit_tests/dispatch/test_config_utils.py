# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

from vllm_fl.dispatch.config.utils import get_config_path, load_platform_config
from vllm_fl.utils import get_flag_gems_whitelist_blacklist


def _platform_with_capability(major: int, minor: int = 0):
    return SimpleNamespace(
        vendor_name="nvidia",
        get_device_capability=lambda: SimpleNamespace(major=major, minor=minor),
    )


def test_hopper_optimization_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
    platform = _platform_with_capability(9)
    with patch("vllm.platforms.current_platform", platform):
        path = get_config_path("nvidia")
        config = load_platform_config("nvidia")

    assert path is not None
    assert path.name == "nvidia.yaml"
    assert config["op_backends"]["attention_backend"][0] == "flagos"
    assert {"linear", "empty", "cat", "mm", "mm_out"}.issubset(
        config["flagos_blacklist"]
    )


def test_hopper_uses_architecture_specific_config_when_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
    platform = _platform_with_capability(9)
    with patch("vllm.platforms.current_platform", platform):
        path = get_config_path("nvidia")
        config = load_platform_config("nvidia")

    assert path is not None
    assert path.name == "nvidia_hopper.yaml"
    assert config["op_backends"]["attention_backend"][0] == "vendor"
    assert {"mm", "mm_out"}.issubset(config["flagos_blacklist"])


def test_non_hopper_nvidia_keeps_vendor_wide_config(monkeypatch):
    monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
    platform = _platform_with_capability(8)
    with patch("vllm.platforms.current_platform", platform):
        path = get_config_path("nvidia")
        config = load_platform_config("nvidia")

    assert path is not None
    assert path.name == "nvidia.yaml"
    assert config["op_backends"]["attention_backend"][0] == "flagos"
    assert "mm" not in config["flagos_blacklist"]


def test_kunlunxin_numerical_fallbacks_keep_cat_enabled():
    config = load_platform_config("kunlunxin")
    blacklist = config["flagos_blacklist"]

    assert "_scaled_dot_product_flash_attention" in blacklist
    assert "mul" in blacklist
    assert "cat" not in blacklist
    assert "cat_out" not in blacklist


def test_capability_probe_failure_falls_back_to_vendor_config(monkeypatch):
    monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "true")
    platform = SimpleNamespace(
        get_device_capability=lambda: (_ for _ in ()).throw(RuntimeError("no GPU"))
    )
    with patch("vllm.platforms.current_platform", platform):
        path = get_config_path("nvidia")

    assert path is not None
    assert path.name == "nvidia.yaml"


def test_hopper_gemm_defaults_preserve_explicit_registration_policy(monkeypatch):
    for name in (
        "VLLM_FL_HOPPER_LONG_CONTEXT_OPT",
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    ):
        monkeypatch.delenv(name, raising=False)
    with patch("vllm.platforms.current_platform", _platform_with_capability(9)):
        whitelist, blacklist = get_flag_gems_whitelist_blacklist()
        assert whitelist is None
        assert {"linear", "mm", "mm_out"}.issubset(blacklist)
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND", "mm,addmm")
        _, appended = get_flag_gems_whitelist_blacklist()
        assert appended.count("mm") == 1 and "addmm" in appended
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "linear")
        monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST_APPEND")
        assert get_flag_gems_whitelist_blacklist() == (None, ["linear"])
        monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST")
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "mm")
        assert get_flag_gems_whitelist_blacklist() == (["mm"], None)


def test_hopper_gemm_defaults_do_not_leak_into_other_architectures(monkeypatch):
    monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
    for major in (9, 8, 10):
        with patch("vllm.platforms.current_platform", _platform_with_capability(major)):
            config = load_platform_config("nvidia")
        assert ("mm" in config["flagos_blacklist"]) == (major == 9)
        # The vendor-wide config also disables mm_out independently of the
        # Hopper-specific mm default.
        assert "mm_out" in config["flagos_blacklist"]

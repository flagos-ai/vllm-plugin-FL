# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

from vllm_fl.dispatch.config.utils import (
    get_config_path,
    get_effective_config,
    get_flagos_blacklist,
    load_platform_config,
)


def test_mthreads_resolves_existing_musa_config():
    assert get_config_path("mthreads") == get_config_path("musa")
    config = load_platform_config("mthreads")
    assert config is not None
    assert config["op_backends"]["attention_backend"][0] == "vendor:musa"
    assert "scaled_dot_product_attention" in config["flagos_blacklist"]


def test_detected_mthreads_loads_musa_policy_and_operator_blacklist(monkeypatch):
    monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
    platform = SimpleNamespace(vendor_name="mthreads")
    with patch("vllm.platforms.current_platform", platform):
        path = get_config_path()
        config = get_effective_config()
        blacklist = get_flagos_blacklist()

    assert path is not None
    assert path.name == "musa.yaml"
    assert config["prefer"] == "flagos"
    assert config["op_backends"]["attention_backend"][0] == "vendor:musa"
    assert blacklist == ["scaled_dot_product_attention"]


def test_unknown_platform_keeps_no_config():
    assert get_config_path("unknown_platform") is None
    assert load_platform_config("unknown_platform") is None


def _platform_with_capability(major: int, minor: int = 0):
    return SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=major, minor=minor)
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
    assert config["flagos_blacklist"] == [
        "topk",
        "_scaled_dot_product_attention_math",
        "maximum",
        "true_divide",
        "scatter_",
    ]
    assert config["oot_blacklist"] == []


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


def test_capability_probe_failure_falls_back_to_vendor_config(monkeypatch):
    monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "true")
    platform = SimpleNamespace(
        get_device_capability=lambda: (_ for _ in ()).throw(RuntimeError("no GPU"))
    )
    with patch("vllm.platforms.current_platform", platform):
        path = get_config_path("nvidia")

    assert path is not None
    assert path.name == "nvidia.yaml"

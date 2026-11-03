# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

from vllm_fl.dispatch.config.utils import get_config_path, load_platform_config


def _platform_with_capability(major: int, minor: int = 0):
    return SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=major, minor=minor)
    )


def test_gcu_vendor_alias_loads_enflame_native_config():
    with patch("vllm.platforms.current_platform", SimpleNamespace(vendor_name="gcu")):
        path = get_config_path("gcu")
        detected_path = get_config_path()
        config = load_platform_config()

    assert path is not None
    assert path.name == "enflame.yaml"
    assert detected_path == path
    assert load_platform_config("gcu") == config
    assert load_platform_config("enflame") == config
    assert config["op_backends"]["attention_backend"] == [
        "vendor:gcu",
        "flagos",
        "reference",
    ]
    assert config["flagos_blacklist"] == [
        "scaled_dot_product_attention",
        "index_put_",
        "_index_put_impl_",
        "mm_out",
    ]


def test_enflame_default_policy_keeps_unsupported_variants_native(monkeypatch):
    from vllm_fl.utils import (
        get_flag_gems_whitelist_blacklist,
        use_flaggems_op,
    )

    for key in (
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "True")
    monkeypatch.setenv("VLLM_FL_PREFER", "flagos")
    monkeypatch.setenv("USE_FLAGGEMS", "1")

    for vendor in ("gcu", "enflame"):
        with patch(
            "vllm.platforms.current_platform", SimpleNamespace(vendor_name=vendor)
        ):
            assert get_flag_gems_whitelist_blacklist() == (
                None,
                [
                    "scaled_dot_product_attention",
                    "index_put_",
                    "_index_put_impl_",
                    "mm_out",
                ],
            )
            assert use_flaggems_op("index_put_") is False
            assert use_flaggems_op("_index_put_impl_") is False
            # Native GCU index_fill_ aborts for penalty int32 rows / int64 indices.
            assert use_flaggems_op("index_fill_") is True
            assert use_flaggems_op("index_fill") is True
            assert use_flaggems_op("mm_out") is False
            assert use_flaggems_op("mm") is True
            assert use_flaggems_op("index_put") is True
            assert use_flaggems_op("rms_norm") is True


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

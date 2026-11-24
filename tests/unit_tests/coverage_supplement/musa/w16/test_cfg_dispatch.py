# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for vllm_fl.dispatch.config.utils residual branches."""

import sys
from types import SimpleNamespace

import pytest

import vllm_fl.dispatch.config as config_pkg
import vllm_fl.dispatch.config.utils as cfg


def _platform(**attrs):
    return SimpleNamespace(**attrs)


# --------------------------------------------------------------------------- #
# get_platform_name
# --------------------------------------------------------------------------- #


def test_get_platform_name_detects_current_platform(monkeypatch):
    import vllm.platforms as vp

    monkeypatch.setattr(vp, "current_platform", _platform(vendor_name="musa"))
    assert cfg.get_platform_name() == "musa"


def test_get_platform_name_env_override_on_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)
    monkeypatch.setenv("VLLM_FL_PLATFORM", "  ASCEND ")
    assert cfg.get_platform_name() == "ascend"


def test_get_platform_name_unknown_without_override(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)
    monkeypatch.delenv("VLLM_FL_PLATFORM", raising=False)
    assert cfg.get_platform_name() == "unknown"


# --------------------------------------------------------------------------- #
# _get_arch_config_name / get_config_path
# --------------------------------------------------------------------------- #


def test_get_arch_config_name_non_nvidia(monkeypatch):
    monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
    assert cfg._get_arch_config_name("ascend") is None


def test_get_arch_config_name_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
    assert cfg._get_arch_config_name("nvidia") is None


def test_get_arch_config_name_capability_none(monkeypatch):
    import vllm.platforms as vp

    monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "true")
    monkeypatch.setattr(
        vp,
        "current_platform",
        _platform(get_device_capability=lambda: None),
    )
    assert cfg._get_arch_config_name("nvidia") is None


def test_get_arch_config_name_hopper(monkeypatch):
    import vllm.platforms as vp

    monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
    monkeypatch.setattr(
        vp,
        "current_platform",
        _platform(get_device_capability=lambda: _platform(major=9)),
    )
    assert cfg._get_arch_config_name("nvidia") == "nvidia_hopper"


def test_get_config_path_alias_mthreads_to_musa():
    assert cfg.get_config_path("mthreads").name == "musa.yaml"


def test_get_config_path_alias_gcu_to_enflame():
    assert cfg.get_config_path("gcu").name == "enflame.yaml"


def test_get_config_path_unknown_returns_none():
    assert cfg.get_config_path("no_such_platform") is None


# --------------------------------------------------------------------------- #
# load_platform_config
# --------------------------------------------------------------------------- #


def test_load_platform_config_non_dict_returns_none(monkeypatch, tmp_path):
    path = tmp_path / "scalar.yaml"
    path.write_text("42\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "get_config_path", lambda platform=None: path)
    assert cfg.load_platform_config() is None


def test_load_platform_config_invalid_yaml_returns_none(monkeypatch, tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("a: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "get_config_path", lambda platform=None: path)
    assert cfg.load_platform_config() is None


# --------------------------------------------------------------------------- #
# get_per_op_order
# --------------------------------------------------------------------------- #


def test_get_per_op_order_load_none(monkeypatch):
    monkeypatch.setattr(cfg, "load_platform_config", lambda *a, **k: None)
    assert cfg.get_per_op_order() is None


def test_get_per_op_order_non_dict_section():
    assert cfg.get_per_op_order({"per_op": ["bad"]}) is None


def test_get_per_op_order_coerces_values():
    result = cfg.get_per_op_order(
        {"per_op": {"a": ["x", 1], "b": "y", "c": 5}}
    )
    assert result == {"a": ["x", "1"], "b": ["y"]}


def test_get_per_op_order_empty_section():
    assert cfg.get_per_op_order({"per_op": {}}) is None


# --------------------------------------------------------------------------- #
# flagos / oot list extractors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "func",
    [cfg.get_flagos_blacklist, cfg.get_flagos_whitelist, cfg.get_oot_blacklist],
)
def test_list_extractors_without_config(monkeypatch, func):
    monkeypatch.setattr(cfg, "load_platform_config", lambda *a, **k: None)
    assert func() is None


def test_flagos_blacklist_non_list_returns_none():
    assert cfg.get_flagos_blacklist({"flagos_blacklist": "not-a-list"}) is None


def test_flagos_blacklist_list_coerced():
    assert cfg.get_flagos_blacklist({"flagos_blacklist": [1, "a"]}) == ["1", "a"]


def test_flagos_whitelist_list_coerced():
    assert cfg.get_flagos_whitelist({"flagos_whitelist": ["rms_norm"]}) == [
        "rms_norm"
    ]


def test_oot_blacklist_non_list_returns_none():
    assert cfg.get_oot_blacklist({"oot_blacklist": 5}) is None


def test_oot_blacklist_list_coerced():
    assert cfg.get_oot_blacklist({"oot_blacklist": [2]}) == ["2"]


# --------------------------------------------------------------------------- #
# get_effective_config
# --------------------------------------------------------------------------- #


def test_effective_config_user_file_wins(monkeypatch, tmp_path):
    path = tmp_path / "user.yaml"
    path.write_text("prefer: flagos\n", encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_CONFIG", str(path))
    assert cfg.get_effective_config() == {"prefer": "flagos"}


def test_effective_config_invalid_user_file_falls_back(monkeypatch, tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("a: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_CONFIG", str(path))
    monkeypatch.setattr(cfg, "load_platform_config", lambda *a, **k: None)
    assert cfg.get_effective_config() == {}


def test_effective_config_non_dict_user_file_falls_back(monkeypatch, tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    monkeypatch.setenv("VLLM_FL_CONFIG", str(path))
    monkeypatch.setattr(
        cfg, "load_platform_config", lambda *a, **k: {"platform": True}
    )
    assert cfg.get_effective_config() == {"platform": True}


def test_effective_config_uses_platform_config(monkeypatch):
    monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
    monkeypatch.setattr(cfg, "load_platform_config", lambda *a, **k: {"p": 1})
    assert cfg.get_effective_config() == {"p": 1}


def test_effective_config_empty_when_nothing_found(monkeypatch):
    monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
    monkeypatch.setattr(cfg, "load_platform_config", lambda *a, **k: None)
    assert cfg.get_effective_config() == {}


# --------------------------------------------------------------------------- #
# get_vendor_device_map
# --------------------------------------------------------------------------- #


def test_vendor_device_map_not_dict(monkeypatch):
    monkeypatch.setattr(cfg, "VENDOR_DEVICE_MAP", ["bad"])
    assert cfg.get_vendor_device_map() == {}


def test_vendor_device_map_filters_invalid_entries(monkeypatch):
    monkeypatch.setattr(
        cfg,
        "VENDOR_DEVICE_MAP",
        {
            "good": {"device_type": "cuda", "device_name": "nvidia"},
            5: {"device_type": "cuda", "device_name": "x"},
            "empty_type": {"device_type": "", "device_name": "x"},
            "missing_name": {"device_type": "cuda"},
            "scalar": "nope",
        },
    )
    assert cfg.get_vendor_device_map() == {
        "good": {"device_type": "cuda", "device_name": "nvidia"}
    }


# --------------------------------------------------------------------------- #
# package re-exports
# --------------------------------------------------------------------------- #


def test_config_package_reexports_utils():
    assert config_pkg.get_vendor_device_map is cfg.get_vendor_device_map
    assert config_pkg.get_platform_name is cfg.get_platform_name


def test_configs_package_imports_cleanly():
    import vllm_fl.configs as configs_pkg

    assert configs_pkg.__name__ == "vllm_fl.configs"

# Copyright (c) 2026 BAAI. All rights reserved.

"""Extra branch coverage for ``vllm_fl.dispatch.config.utils``.

Exercises platform detection fallbacks, architecture-specific config
selection, config extraction helpers and the vendor-device map sanitiser.
"""

import sys
import types

import pytest

from vllm_fl.dispatch.config import utils as u


def _patch_platform(monkeypatch, obj):
    import vllm.platforms as vp

    monkeypatch.setattr(vp, "current_platform", obj)


class TestGetPlatformName:
    def test_uses_current_platform(self, monkeypatch):
        _patch_platform(monkeypatch, types.SimpleNamespace(vendor_name="musa"))
        assert u.get_platform_name() == "musa"

    def test_import_error_uses_env_override(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)
        monkeypatch.setenv("VLLM_FL_PLATFORM", "  MUSA  ")
        assert u.get_platform_name() == "musa"

    def test_import_error_unknown(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)
        monkeypatch.delenv("VLLM_FL_PLATFORM", raising=False)
        assert u.get_platform_name() == "unknown"


class TestArchConfigName:
    def test_non_nvidia(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        assert u._get_arch_config_name("musa") is None

    def test_opt_disabled(self, monkeypatch):
        monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
        assert u._get_arch_config_name("nvidia") is None

    def test_capability_probe_raises(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "true")
        platform = types.SimpleNamespace(
            get_device_capability=lambda: (_ for _ in ()).throw(RuntimeError("no gpu"))
        )
        _patch_platform(monkeypatch, platform)
        assert u._get_arch_config_name("nvidia") is None

    def test_capability_none(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        _patch_platform(
            monkeypatch, types.SimpleNamespace(get_device_capability=lambda: None)
        )
        assert u._get_arch_config_name("nvidia") is None

    def test_hopper_capability(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        cap = types.SimpleNamespace(major=9, minor=0)
        _patch_platform(
            monkeypatch, types.SimpleNamespace(get_device_capability=lambda: cap)
        )
        assert u._get_arch_config_name("nvidia") == "nvidia_hopper"

    def test_non_hopper_capability(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        cap = types.SimpleNamespace(major=8, minor=0)
        _patch_platform(
            monkeypatch, types.SimpleNamespace(get_device_capability=lambda: cap)
        )
        assert u._get_arch_config_name("nvidia") is None


class TestGetConfigPath:
    def test_alias_mthreads(self):
        assert u.get_config_path("mthreads") == u.get_config_path("musa")
        assert u.get_config_path("mthreads").name == "musa.yaml"

    def test_gcu_maps_to_enflame(self):
        assert u.get_config_path("gcu") == u.get_config_path("enflame")
        assert u.get_config_path("gcu").name == "enflame.yaml"

    def test_unknown_platform_none(self):
        assert u.get_config_path("no_such_platform") is None

    def test_arch_config_used_when_present(self, monkeypatch):
        monkeypatch.setattr(u, "_get_arch_config_name", lambda p: "nvidia_hopper")
        assert u.get_config_path("nvidia").name == "nvidia_hopper.yaml"

    def test_arch_config_missing_falls_back(self, monkeypatch):
        monkeypatch.setattr(u, "_get_arch_config_name", lambda p: "missing_arch")
        assert u.get_config_path("nvidia").name == "nvidia.yaml"

    def test_none_platform_autodetects(self, monkeypatch):
        monkeypatch.setattr(u, "get_platform_name", lambda: "musa")
        assert u.get_config_path().name == "musa.yaml"


class TestLoadPlatformConfig:
    def test_none_when_missing(self):
        assert u.load_platform_config("no_such_platform") is None

    def test_invalid_yaml_returns_none(self, monkeypatch, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("a: [unclosed\n", encoding="utf-8")
        monkeypatch.setattr(u, "get_config_path", lambda platform=None: bad)
        assert u.load_platform_config() is None

    def test_non_dict_yaml_returns_none(self, monkeypatch, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- 1\n- 2\n", encoding="utf-8")
        monkeypatch.setattr(u, "get_config_path", lambda platform=None: p)
        assert u.load_platform_config() is None

    def test_real_musa_config(self):
        config = u.load_platform_config("musa")
        assert isinstance(config, dict)


class TestPerOpOrder:
    def test_no_config(self, monkeypatch):
        monkeypatch.setattr(u, "load_platform_config", lambda platform=None: None)
        assert u.get_per_op_order() is None

    def test_explicit_config_none_perop_type(self):
        assert u.get_per_op_order({"per_op": []}) is None

    def test_empty_perop(self):
        assert u.get_per_op_order({"per_op": {}}) is None

    def test_list_and_scalar_values(self):
        got = u.get_per_op_order(
            {"per_op": {"mm": ["flagos", "reference"], "attn": "vendor", "bad": 7}}
        )
        assert got == {"mm": ["flagos", "reference"], "attn": ["vendor"]}

    def test_loads_platform_when_none(self, monkeypatch):
        monkeypatch.setattr(
            u, "load_platform_config", lambda platform=None: {"per_op": {"x": "a"}}
        )
        assert u.get_per_op_order() == {"x": ["a"]}


class TestLists:
    def test_blacklist_empty_list(self):
        assert u.get_flagos_blacklist({}) == []

    def test_blacklist_non_list(self):
        assert u.get_flagos_blacklist({"flagos_blacklist": "x"}) is None

    def test_blacklist_coerces_to_str(self):
        assert u.get_flagos_blacklist({"flagos_blacklist": [1, "two"]}) == ["1", "two"]

    def test_blacklist_no_config(self, monkeypatch):
        monkeypatch.setattr(u, "load_platform_config", lambda platform=None: None)
        assert u.get_flagos_blacklist() is None

    def test_whitelist(self):
        assert u.get_flagos_whitelist({}) == []
        assert u.get_flagos_whitelist({"flagos_whitelist": [1]}) == ["1"]
        assert u.get_flagos_whitelist({"flagos_whitelist": "x"}) is None

    def test_whitelist_no_config(self, monkeypatch):
        monkeypatch.setattr(u, "load_platform_config", lambda platform=None: None)
        assert u.get_flagos_whitelist() is None

    def test_oot(self):
        assert u.get_oot_blacklist({}) == []
        assert u.get_oot_blacklist({"oot_blacklist": ["a"]}) == ["a"]
        assert u.get_oot_blacklist({"oot_blacklist": "x"}) is None

    def test_oot_no_config(self, monkeypatch):
        monkeypatch.setattr(u, "load_platform_config", lambda platform=None: None)
        assert u.get_oot_blacklist() is None


class TestEffectiveConfig:
    def test_user_config_dict_wins(self, monkeypatch, tmp_path):
        cfg = tmp_path / "user.yaml"
        cfg.write_text("prefer: vendor\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        assert u.get_effective_config() == {"prefer": "vendor"}

    def test_user_config_invalid_yaml_falls_back(self, monkeypatch, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("a: [unclosed\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        monkeypatch.setattr(
            u, "load_platform_config", lambda platform=None: {"from": "platform"}
        )
        assert u.get_effective_config() == {"from": "platform"}

    def test_user_config_non_dict_falls_back(self, monkeypatch, tmp_path):
        cfg = tmp_path / "list.yaml"
        cfg.write_text("- 1\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        monkeypatch.setattr(
            u, "load_platform_config", lambda platform=None: {"from": "platform"}
        )
        assert u.get_effective_config() == {"from": "platform"}

    def test_missing_user_config_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VLLM_FL_CONFIG", str(tmp_path / "nope.yaml"))
        monkeypatch.setattr(u, "load_platform_config", lambda platform=None: {})
        assert u.get_effective_config() == {}

    def test_platform_config_used(self, monkeypatch):
        monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
        monkeypatch.setattr(
            u, "load_platform_config", lambda platform=None: {"k": 1}
        )
        assert u.get_effective_config() == {"k": 1}


class TestVendorDeviceMap:
    def test_non_dict_map(self, monkeypatch):
        monkeypatch.setattr(u, "VENDOR_DEVICE_MAP", [])
        assert u.get_vendor_device_map() == {}

    def test_filters_invalid_entries(self, monkeypatch):
        monkeypatch.setattr(
            u,
            "VENDOR_DEVICE_MAP",
            {
                "cuda": {"device_type": "cuda", "device_name": "cuda"},
                3: {"device_type": "t", "device_name": "d"},
                "scalar": 5,
                "empty_type": {"device_type": "", "device_name": "d"},
                "bad_name": {"device_type": "t", "device_name": 5},
            },
        )
        got = u.get_vendor_device_map()
        assert got == {"cuda": {"device_type": "cuda", "device_name": "cuda"}}

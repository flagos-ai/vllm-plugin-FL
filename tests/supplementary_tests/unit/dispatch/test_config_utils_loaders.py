# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for ``vllm_fl.dispatch.config.utils`` loader branches.

``test_discovery_builtin_ops.py`` already covers ``get_platform_name`` /
``_get_arch_config_name`` / ``get_config_path`` and ``test_manager_policy_edges``
covers policy consumers.  This file targets the still-uncovered loader bodies:

* ``load_platform_config`` (yaml round-trip, non-dict document, read failure);
* ``get_per_op_order`` (list values, string shorthand, non-dict section, empty
  result, no-config);
* ``get_flagos_blacklist`` / ``get_flagos_whitelist`` / ``get_oot_blacklist``
  (list coercion and non-list rejection);
* ``get_effective_config`` (user file, non-dict user file, unreadable user
  file, platform fallback, empty result);
* ``get_vendor_device_map`` (entry filtering and non-dict guard).
"""

import sys
import types
from pathlib import Path

import pytest
import yaml

import vllm_fl.dispatch.config.utils as config_utils
from vllm_fl.dispatch.config.utils import (
    get_effective_config,
    get_flagos_blacklist,
    get_flagos_whitelist,
    get_oot_blacklist,
    get_per_op_order,
    get_vendor_device_map,
    load_platform_config,
)

pytestmark = [pytest.mark.supplementary]

_CONFIG_DIR = Path(config_utils.__file__).parent


def _write_yaml(path: Path, payload) -> Path:
    """Write *payload* as YAML and return the path."""
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _make_vllm_platforms(monkeypatch, current_platform=None):
    """Install a fake ``vllm.platforms`` module into ``sys.modules``.

    When *current_platform* is None the attribute is omitted entirely so that
    ``from vllm.platforms import current_platform`` raises ImportError.
    """
    stub = types.ModuleType("vllm.platforms")
    if current_platform is not None:
        stub.current_platform = current_platform
    monkeypatch.setitem(sys.modules, "vllm.platforms", stub)


class _Platform:
    """current_platform double with just a vendor_name."""

    def __init__(self, vendor_name):
        self.vendor_name = vendor_name


class TestLoadPlatformConfig:
    """``load_platform_config`` file-level branches."""

    def test_reads_real_config_yaml(self):
        """The shipped ascend.yaml loads as a dict (hand-checked key)."""
        config = load_platform_config("ascend")
        assert isinstance(config, dict)
        # Independent oracle: parse the file ourselves and compare a key.
        with open(_CONFIG_DIR / "ascend.yaml", encoding="utf-8") as f:
            expected = yaml.safe_load(f)
        assert config == expected

    def test_unknown_platform_returns_none(self):
        """A platform without a config file resolves to None."""
        assert load_platform_config("does_not_exist_xyz") is None

    def test_non_dict_document_returns_none(self, tmp_path, monkeypatch):
        """A yaml document that is not a mapping is rejected."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n", encoding="utf-8")

        class _Platform:
            vendor_name = "badyaml"

        _make_vllm_platforms(monkeypatch, _Platform())
        # Point the config dir at tmp_path by overriding the resolved name.
        monkeypatch.setattr(config_utils, "_CONFIG_DIR", tmp_path)
        assert load_platform_config("badyaml") is None

    def test_read_failure_returns_none(self, tmp_path, monkeypatch):
        """An unreadable config file degrades to None."""
        broken = tmp_path / "broken.yaml"
        broken.write_text("key: [unclosed\n", encoding="utf-8")

        class _Platform:
            vendor_name = "broken"

        _make_vllm_platforms(monkeypatch, _Platform())
        monkeypatch.setattr(config_utils, "_CONFIG_DIR", tmp_path)
        assert load_platform_config("broken") is None


class TestGetPerOpOrder:
    """``get_per_op_order`` extraction branches."""

    def test_list_and_string_values(self):
        """List backends are coerced to strings; strings are single-item."""
        config = {
            "per_op": {
                "opA": ["flagos", "reference"],
                "opB": "vendor",
                "opC": 42,  # neither list nor str -> dropped
            }
        }
        result = get_per_op_order(config)
        assert result == {"opA": ["flagos", "reference"], "opB": ["vendor"]}

    def test_non_dict_section_returns_none(self):
        """A per_op section that is not a mapping is rejected."""
        assert get_per_op_order({"per_op": ["flat", "list"]}) is None

    def test_empty_section_returns_none(self):
        """An empty per_op section resolves to None (no restrictions)."""
        assert get_per_op_order({"per_op": {}}) is None

    def test_no_config_returns_none(self, monkeypatch):
        """With no platform config at all the order is None."""
        monkeypatch.setattr(config_utils, "load_platform_config", lambda: None)
        assert get_per_op_order(None) is None

    def test_loads_from_platform_config(self, monkeypatch):
        """A None config argument falls back to load_platform_config()."""
        monkeypatch.setattr(
            config_utils,
            "load_platform_config",
            lambda: {"per_op": {"opZ": ["reference"]}},
        )
        assert get_per_op_order(None) == {"opZ": ["reference"]}


class TestBlackWhitelists:
    """Blacklist/whitelist coercion branches."""

    def test_flagos_blacklist_coerces_to_strings(self):
        """Non-string entries are stringified."""
        result = get_flagos_blacklist({"flagos_blacklist": ["op.a", 7]})
        assert result == ["op.a", "7"]

    def test_flagos_blacklist_default_empty(self):
        """A missing or empty blacklist yields an empty list."""
        assert get_flagos_blacklist({}) == []
        assert get_flagos_blacklist({"flagos_blacklist": []}) == []

    def test_flagos_blacklist_non_list_returns_none(self):
        """A non-list blacklist value is rejected (None)."""
        assert get_flagos_blacklist({"flagos_blacklist": "op.a"}) is None

    def test_flagos_whitelist_coerces_to_strings(self):
        """Whitelist entries are stringified like the blacklist."""
        assert get_flagos_whitelist({"flagos_whitelist": ["w", 1]}) == ["w", "1"]

    def test_flagos_whitelist_non_list_returns_none(self):
        """A non-list whitelist value is rejected (None)."""
        assert get_flagos_whitelist({"flagos_whitelist": 3}) is None

    def test_flagos_whitelist_no_config(self, monkeypatch):
        """Without a platform config the whitelist is None."""
        monkeypatch.setattr(config_utils, "load_platform_config", lambda: None)
        assert get_flagos_whitelist(None) is None

    def test_oot_blacklist_coerces_to_strings(self):
        """OOT blacklist entries are stringified."""
        assert get_oot_blacklist({"oot_blacklist": ["o", 2]}) == ["o", "2"]

    def test_oot_blacklist_non_list_returns_none(self):
        """A non-list OOT blacklist value is rejected (None)."""
        assert get_oot_blacklist({"oot_blacklist": {"not": "a list"}}) is None

    def test_oot_blacklist_no_config(self, monkeypatch):
        """Without a platform config the OOT blacklist is None."""
        monkeypatch.setattr(config_utils, "load_platform_config", lambda: None)
        assert get_oot_blacklist(None) is None


class TestGetEffectiveConfig:
    """``get_effective_config`` priority ladder."""

    def test_user_config_file_wins(self, tmp_path, monkeypatch):
        """VLLM_FL_CONFIG pointing at a dict yaml file is returned as-is."""
        user = _write_yaml(tmp_path / "user.yaml", {"per_op": {"u": ["reference"]}})
        monkeypatch.setenv("VLLM_FL_CONFIG", str(user))
        assert get_effective_config() == {"per_op": {"u": ["reference"]}}

    def test_user_config_non_dict_ignored(self, tmp_path, monkeypatch):
        """A non-dict user yaml falls through to the platform config."""
        bad = tmp_path / "user.yaml"
        bad.write_text("just a scalar\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(bad))
        monkeypatch.setattr(
            config_utils, "load_platform_config", lambda: {"from": "platform"}
        )
        assert get_effective_config() == {"from": "platform"}

    def test_user_config_read_error_ignored(self, tmp_path, monkeypatch):
        """An unreadable user yaml falls through to the platform config."""
        broken = tmp_path / "user.yaml"
        broken.write_text("key: [unclosed\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(broken))
        monkeypatch.setattr(
            config_utils, "load_platform_config", lambda: {"from": "platform"}
        )
        assert get_effective_config() == {"from": "platform"}

    def test_missing_user_file_ignored(self, tmp_path, monkeypatch):
        """A VLLM_FL_CONFIG path that does not exist is skipped."""
        monkeypatch.setenv("VLLM_FL_CONFIG", str(tmp_path / "nope.yaml"))
        monkeypatch.setattr(
            config_utils, "load_platform_config", lambda: {"from": "platform"}
        )
        assert get_effective_config() == {"from": "platform"}

    def test_platform_config_fallback(self, monkeypatch):
        """Without a user file the platform config is returned."""
        monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
        monkeypatch.setattr(
            config_utils, "load_platform_config", lambda: {"flagos_blacklist": ["x"]}
        )
        assert get_effective_config() == {"flagos_blacklist": ["x"]}

    def test_empty_when_nothing_configured(self, monkeypatch):
        """No user file and no platform config yields the empty config."""
        monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
        monkeypatch.setattr(config_utils, "load_platform_config", lambda: None)
        assert get_effective_config() == {}


class TestGetVendorDeviceMap:
    """``get_vendor_device_map`` filtering branches."""

    def test_real_map_has_nvidia_and_cuda(self):
        """The shipped map keeps well-formed vendors (independent oracle)."""
        result = get_vendor_device_map()
        assert result["nvidia"] == {"device_type": "cuda", "device_name": "nvidia"}

    def test_filters_malformed_entries(self, monkeypatch):
        """Malformed vendor entries are dropped from the copy."""
        fake_map = {
            "good": {"device_type": "cuda", "device_name": "good"},
            "no_dtype": {"device_name": "x"},  # missing device_type
            "empty_dtype": {"device_type": "", "device_name": "x"},
            "no_dname": {"device_type": "cuda"},  # missing device_name
            "empty_dname": {"device_type": "cuda", "device_name": ""},
            "non_dict": "cuda",  # value not a mapping
            123: {"device_type": "cuda", "device_name": "n"},
        }
        monkeypatch.setattr(config_utils, "VENDOR_DEVICE_MAP", fake_map)
        result = get_vendor_device_map()
        assert result == {"good": {"device_type": "cuda", "device_name": "good"}}

    def test_non_dict_map_returns_empty(self, monkeypatch):
        """A non-dict VENDOR_DEVICE_MAP degrades to an empty result."""
        monkeypatch.setattr(config_utils, "VENDOR_DEVICE_MAP", ["not", "a", "dict"])
        assert get_vendor_device_map() == {}

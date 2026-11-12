# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Supplementary coverage tests for ``vllm_fl/dispatch/config/utils.py``.

D3 tranche of the flagcicd coverage campaign. Tests assert behavior
values: resolved platform names, selected config paths, merged Hopper
blacklists, extracted per-op orders/blacklists, the VLLM_FL_CONFIG
priority chain, and the vendor device map contents.

Local collection note: importing ``vllm_fl`` pulls in ``flag_gems``
which needs triton (no Windows wheel). ``_flaggems_shim`` installs an
inert placeholder only when the real package cannot import.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from vllm_fl.dispatch.config.utils import (
    get_config_path,
    get_effective_config,
    get_flagos_blacklist,
    get_oot_blacklist,
    get_per_op_order,
    get_platform_name,
    get_vendor_device_map,
    load_platform_config,
)


def _platform(vendor_name, major=None):
    """A platform stub; major=None makes get_device_capability raise."""
    if major is None:
        return SimpleNamespace(
            vendor_name=vendor_name,
            get_device_capability=lambda: (_ for _ in ()).throw(RuntimeError("no GPU")),
        )
    return SimpleNamespace(
        vendor_name=vendor_name,
        get_device_capability=lambda: SimpleNamespace(major=major, minor=0),
    )


class TestGetPlatformName:
    """Platform resolution: vllm platform first, env fallback second."""

    def test_vllm_platform_vendor_name_wins(self):
        """A resolvable vllm platform returns its vendor_name directly."""
        with patch("vllm.platforms.current_platform", _platform("metax")):
            assert get_platform_name() == "metax"

    def test_import_failure_falls_back_to_env_override(self, monkeypatch):
        """With vllm.platforms unimportable, VLLM_FL_PLATFORM takes over."""
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)
        monkeypatch.setenv("VLLM_FL_PLATFORM", "CUDA")
        assert get_platform_name() == "cuda"  # normalized to lowercase

    def test_import_failure_without_env_returns_unknown(self, monkeypatch):
        """No platform and no env override resolve to 'unknown'."""
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)
        monkeypatch.setenv("VLLM_FL_PLATFORM", "   ")
        assert get_platform_name() == "unknown"

    def test_env_fallback_ignores_blank_values(self, monkeypatch):
        """A blank override falls through to 'unknown'."""
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)
        monkeypatch.setenv("VLLM_FL_PLATFORM", "")
        assert get_platform_name() == "unknown"


class TestConfigPathAndLoad:
    """Platform config path selection and Hopper blacklist merging."""

    def test_unknown_platform_has_no_config(self):
        """A platform without a YAML resolves to no path and no config."""
        assert get_config_path("definitely_not_a_platform") is None
        assert load_platform_config("definitely_not_a_platform") is None

    def test_nvidia_resolves_vendor_yaml(self):
        """The nvidia platform selects nvidia.yaml and parses it."""
        path = get_config_path("nvidia")
        assert path is not None and path.name == "nvidia.yaml"
        config = load_platform_config("nvidia")
        assert config["prefer"] == "flagos"
        assert config["strict"] is False
        assert "cuda" in config["allow_vendors"]
        assert config["op_backends"]["rms_norm"] == ["flagos", "vendor", "reference"]

    def test_hopper_merge_appends_mm_to_blacklist(self, monkeypatch):
        """Capability 9 appends mm/mm_out to the nvidia blacklist."""
        monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
        with patch("vllm.platforms.current_platform", _platform("nvidia", 9)):
            config = load_platform_config("nvidia")
        assert "mm" in config["flagos_blacklist"]
        assert config["flagos_blacklist"].count("mm_out") == 1  # no dup

    def test_non_hopper_keeps_public_blacklist(self, monkeypatch):
        """Capability 8 leaves the nvidia blacklist without mm."""
        monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
        with patch("vllm.platforms.current_platform", _platform("nvidia", 8)):
            config = load_platform_config("nvidia")
        assert "mm" not in config["flagos_blacklist"]
        assert "mm_out" in config["flagos_blacklist"]


class TestPerOpOrder:
    """Per-op backend order extraction."""

    def test_none_config_without_platform_yields_none(self):
        """With no platform config resolvable, per-op order is None."""
        with patch("vllm.platforms.current_platform", _platform(None)):
            assert get_per_op_order() is None

    def test_list_and_string_backend_forms(self):
        """List backends are stringified; a lone string becomes a list."""
        cfg = {"per_op": {"mm": ["flagos", "reference"], "silu_and_mul": "vendor"}}
        assert get_per_op_order(cfg) == {
            "mm": ["flagos", "reference"],
            "silu_and_mul": ["vendor"],
        }

    def test_malformed_entries_are_dropped(self):
        """Non-list/non-string backends and non-dict per_op yield None."""
        assert get_per_op_order({"per_op": {"mm": 5}}) is None
        assert get_per_op_order({"per_op": "junk"}) is None
        assert get_per_op_order({"per_op": {}}) is None


class TestBlacklists:
    """FlagOS and OOT blacklist extraction."""

    def test_flagos_blacklist_from_explicit_config(self):
        """A list blacklist is stringified and returned in order."""
        cfg = {"flagos_blacklist": ["mm", 7, "addmm"]}
        assert get_flagos_blacklist(cfg) == ["mm", "7", "addmm"]

    def test_flagos_blacklist_missing_key_defaults_empty(self):
        """A config without the key yields an empty list, not None."""
        assert get_flagos_blacklist({}) == []

    def test_flagos_blacklist_non_list_is_none(self):
        """A non-list blacklist value returns None."""
        assert get_flagos_blacklist({"flagos_blacklist": "junk"}) is None

    def test_oot_blacklist_extraction(self):
        """OOT blacklist mirrors the flagos extraction rules."""
        assert get_oot_blacklist({"oot_blacklist": ["masked_fill"]}) == ["masked_fill"]
        assert get_oot_blacklist({}) == []
        assert get_oot_blacklist({"oot_blacklist": 3}) is None

    def test_metax_oot_blacklist_from_yaml(self):
        """The metax platform config carries a real OOT blacklist."""
        config = load_platform_config("metax")
        assert get_oot_blacklist(config) == ["masked_fill", "masked_fill_"]


class TestEffectiveConfig:
    """VLLM_FL_CONFIG priority chain over the platform config."""

    def test_user_config_file_highest_priority(self, tmp_path, monkeypatch):
        """A valid VLLM_FL_CONFIG file overrides the platform config."""
        user_cfg = tmp_path / "user.yaml"
        user_cfg.write_text("prefer: reference\nstrict: true\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(user_cfg))
        with patch("vllm.platforms.current_platform", _platform("nvidia")):
            effective = get_effective_config()
        assert effective == {"prefer": "reference", "strict": True}

    def test_user_config_non_dict_falls_through(self, tmp_path, monkeypatch):
        """A YAML file whose root is a list falls back to the platform."""
        user_cfg = tmp_path / "listroot.yaml"
        user_cfg.write_text("- just\n- a list\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(user_cfg))
        with patch("vllm.platforms.current_platform", _platform("nvidia")):
            effective = get_effective_config()
        assert effective["prefer"] == "flagos"  # from nvidia.yaml

    def test_unreadable_user_config_falls_through(self, tmp_path, monkeypatch):
        """An unparsable YAML file is skipped, not fatal."""
        user_cfg = tmp_path / "broken.yaml"
        user_cfg.write_text("prefer: [unclosed\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(user_cfg))
        with patch("vllm.platforms.current_platform", _platform("nvidia")):
            effective = get_effective_config()
        assert effective["prefer"] == "flagos"

    def test_missing_user_config_falls_through(self, tmp_path, monkeypatch):
        """A VLLM_FL_CONFIG pointing nowhere falls back to the platform."""
        monkeypatch.setenv("VLLM_FL_CONFIG", str(tmp_path / "nope.yaml"))
        with patch("vllm.platforms.current_platform", _platform("nvidia")):
            effective = get_effective_config()
        assert effective["op_backends"]["silu_and_mul"] == [
            "flagos",
            "vendor",
            "reference",
        ]

    def test_no_config_anywhere_is_empty(self, monkeypatch):
        """No user file and no resolvable platform yields {}."""
        monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
        with patch("vllm.platforms.current_platform", _platform(None)):
            assert get_effective_config() == {}


class TestVendorDeviceMap:
    """VENDOR_DEVICE_MAP projection into the dispatch registry form."""

    def test_known_vendors_keep_device_fields(self):
        """Selected vendors map to their device_type/device_name pairs."""
        device_map = get_vendor_device_map()
        assert device_map["nvidia"] == {"device_type": "cuda", "device_name": "nvidia"}
        assert device_map["ascend"] == {"device_type": "npu", "device_name": "npu"}
        assert device_map["mthreads"] == {"device_type": "musa", "device_name": "musa"}
        assert device_map["sunrise"] == {"device_type": "ptpu", "device_name": "ptpu"}

    def test_every_entry_is_well_formed(self):
        """All projected entries carry non-empty string fields."""
        device_map = get_vendor_device_map()
        assert len(device_map) >= 10
        for vendor, info in device_map.items():
            assert isinstance(vendor, str) and vendor
            assert set(info) == {"device_type", "device_name"}
            for value in info.values():
                assert isinstance(value, str) and value

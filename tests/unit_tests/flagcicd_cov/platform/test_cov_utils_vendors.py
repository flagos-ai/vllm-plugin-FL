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

"""vllm_fl.utils OOT lists, vendor device map and DeviceInfo (E1).

Companion to test_cov_utils.py: OOT whitelist/blacklist env parsing and
platform-config defaults, the VENDOR_DEVICE_MAP surface and the DeviceInfo
probe helpers. Environment variables and vllm.platforms.current_platform
are the CONTRACT R2 sanctioned boundaries used throughout.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

# The shim import is load-bearing: it must precede vllm_fl/vllm
# imports on hosts without triton. Do not alphabetize.
from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)
import vllm_fl.utils as fl_utils

_VENDOR_ENV_VARS = (
    "VLLM_FL_OOT_WHITELIST",
    "VLLM_FL_OOT_BLACKLIST",
    "VLLM_FL_OOT_ENABLED",
    "VLLM_FL_PREFER_ENABLED",
    "FLAGOS_DEVICE_CONTROL_ENV_VAR",
)


@pytest.fixture(autouse=True)
def _clean_vendor_env(monkeypatch):
    """Remove every env var under test before each test."""
    for name in _VENDOR_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestOotLists:
    """OOT whitelist/blacklist and the OOT enable gate."""

    def test_oot_whitelist_unset_returns_none(self):
        """Without VLLM_FL_OOT_WHITELIST all OOT ops register."""
        assert fl_utils.get_oot_whitelist() is None

    def test_oot_whitelist_parses(self, monkeypatch):
        """The whitelist env var parses with whitespace stripping."""
        monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", " silu_and_mul , rms_norm ,")
        assert fl_utils.get_oot_whitelist() == ["silu_and_mul", "rms_norm"]

    def test_oot_blacklist_ignored_when_whitelist_set(self, monkeypatch):
        """A whitelist makes the blacklist irrelevant."""
        monkeypatch.setenv("VLLM_FL_OOT_WHITELIST", "silu_and_mul")
        monkeypatch.setenv("VLLM_FL_OOT_BLACKLIST", "rms_norm")
        assert fl_utils.get_oot_blacklist() is None

    def test_oot_blacklist_env_parses(self, monkeypatch):
        """The OOT blacklist env var parses with whitespace stripping."""
        monkeypatch.setenv("VLLM_FL_OOT_BLACKLIST", "a, b , c,")
        assert fl_utils.get_oot_blacklist() == ["a", "b", "c"]

    def test_oot_blacklist_platform_config_for_metax(self, monkeypatch):
        """The metax.yaml OOT blacklist loads as platform default."""
        from vllm import platforms as vllm_platforms

        monkeypatch.setattr(
            vllm_platforms,
            "current_platform",
            NS(vendor_name="metax"),
            raising=False,
        )
        assert fl_utils.get_oot_blacklist() == ["masked_fill", "masked_fill_"]

    def test_is_oot_enabled_default_and_gates(self, monkeypatch):
        """OOT registration defaults on; both env gates can disable it."""
        assert fl_utils.is_oot_enabled() is True

        monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "0")
        assert fl_utils.is_oot_enabled() is False

        monkeypatch.delenv("VLLM_FL_OOT_ENABLED")
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "0")
        assert fl_utils.is_oot_enabled() is False


class TestVendorDeviceMap:
    """VENDOR_DEVICE_MAP lookups and their validation."""

    @pytest.mark.parametrize(
        ("vendor_name", "device_type", "device_name"),
        [
            ("nvidia", "cuda", "nvidia"),
            ("ascend", "npu", "npu"),
            ("iluvatar", "cuda", "cuda"),
            ("metax", "cuda", "metax"),
            ("mthreads", "musa", "musa"),
            ("sunrise", "ptpu", "ptpu"),
            ("hygon", "cuda", "cuda"),
            ("amd", "cuda", "cuda"),
            ("thead", "cuda", "thead"),
            ("tsingmicro", "tsingmicro", "txda"),
            ("gcu", "gcu", "gcu"),
            ("enflame", "gcu", "gcu"),
            ("kunlunxin", "cuda", "kunlunxin"),
            ("biren", "cuda", "supa"),
        ],
    )
    def test_get_device_type_and_name(self, vendor_name, device_type, device_name):
        """Both map lookups resolve for every registered vendor."""
        assert fl_utils.get_device_type(vendor_name) == device_type
        assert fl_utils.get_device_name(vendor_name) == device_name

    def test_empty_vendor_name_rejected(self):
        """An empty or whitespace vendor name is rejected."""
        with pytest.raises(ValueError, match="non-empty string"):
            fl_utils._get_vendor_device_field("", "device_type")
        assert "unknown_vendor" not in fl_utils.VENDOR_DEVICE_MAP

    def test_unknown_vendor_rejected(self):
        """A vendor missing from the map raises with the vendor name."""
        with pytest.raises(ValueError, match="not found in VENDOR_DEVICE_MAP"):
            fl_utils._get_vendor_device_field("unknown_vendor", "device_type")
        assert "unknown_vendor" not in fl_utils.VENDOR_DEVICE_MAP

    def test_missing_field_rejected(self):
        """A field absent for a vendor is an error, not a silent None."""
        with pytest.raises(ValueError, match="missing or empty"):
            fl_utils._get_vendor_device_field("nvidia", "no_such_field")
        assert "unknown_vendor" not in fl_utils.VENDOR_DEVICE_MAP

    def test_device_control_env_var_matrix(self, monkeypatch):
        """Validated vendors map to their env var; others get the placeholder."""
        assert fl_utils.get_device_control_env_var("nvidia") == "CUDA_VISIBLE_DEVICES"
        assert (
            fl_utils.get_device_control_env_var("kunlunxin") == "CUDA_VISIBLE_DEVICES"
        )
        assert (
            fl_utils.get_device_control_env_var("amd")
            == "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"
        )

        monkeypatch.setenv("FLAGOS_DEVICE_CONTROL_ENV_VAR", "MY_VISIBLE_DEVICES")
        assert fl_utils.get_device_control_env_var("amd") == "MY_VISIBLE_DEVICES"


class TestDeviceInfoSurface:
    """DeviceInfo probe surface and capability helpers."""

    def test_device_info_reports_probe_fields(self):
        """DeviceInfo exposes the detector's vendor/device/dispatch fields."""
        info = fl_utils.DeviceInfo()
        assert info.vendor_name == "nvidia"
        assert info.device_type == "cuda"
        assert info.dispatch_key == "CUDA"
        assert "nvidia" in info.supported_device

    def test_get_supported_device_accepts_registered_vendors(self):
        """A registered vendor passes the supported-device check."""
        assert fl_utils.DeviceInfo().get_supported_device() is True

    def test_get_flaggems_all_ops_from_full_config(self, monkeypatch):
        """get_flaggems_all_ops reads op names from flag_gems._FULL_CONFIG."""
        monkeypatch.setattr(
            fl_utils.flag_gems, "_FULL_CONFIG", (("op_a", None), ("op_b", None, 1))
        )
        assert fl_utils.get_flaggems_all_ops() == ["op_a", "op_b"]

    def test_get_flaggems_all_ops_empty_on_missing_attribute(self, monkeypatch):
        """A flag_gems without _FULL_CONFIG yields an empty list, not a crash."""
        monkeypatch.delattr(fl_utils.flag_gems, "_FULL_CONFIG", raising=False)
        assert fl_utils.get_flaggems_all_ops() == []

    def test_has_native_triton_moe_false_without_cuda(self):
        """A non-CUDA current platform never reports native Triton MoE."""
        assert fl_utils.has_native_triton_moe() is False

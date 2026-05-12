# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests to verify dispatch config files match registered vendor names.
"""

from pathlib import Path

from vllm_fl.utils import VENDOR_DEVICE_MAP

_CONFIG_DIR = (
    Path(__file__).parent.parent.parent.parent / "vllm_fl" / "dispatch" / "config"
)


class TestDispatchConfigCoverage:
    def test_all_vendors_have_config_file(self):
        """Every vendor_name in VENDOR_DEVICE_MAP must have a matching dispatch config yaml."""
        existing = {p.stem for p in _CONFIG_DIR.glob("*.yaml")}
        missing = [v for v in VENDOR_DEVICE_MAP if v not in existing]
        assert not missing, (
            f"Missing dispatch config yaml for vendor(s): {missing}. "
            f"Create vllm_fl/dispatch/config/<vendor_name>.yaml for each."
        )

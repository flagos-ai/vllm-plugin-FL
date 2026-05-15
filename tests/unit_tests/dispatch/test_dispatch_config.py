# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests to verify dispatch config files are valid when present.
"""

import warnings
from pathlib import Path

import pytest
import yaml

from vllm_fl.utils import VENDOR_DEVICE_MAP

_CONFIG_DIR = (
    Path(__file__).parent.parent.parent.parent / "vllm_fl" / "dispatch" / "config"
)


class TestDispatchConfigCoverage:
    def test_config_files_are_valid_yaml(self):
        """Every dispatch config yaml must be parseable."""
        yaml_files = list(_CONFIG_DIR.glob("*.yaml"))
        for path in yaml_files:
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                assert data is not None, f"{path.name} is empty"
            except yaml.YAMLError as e:
                raise AssertionError(f"{path.name} is not valid YAML: {e}") from e

    @pytest.mark.filterwarnings("default::UserWarning")
    def test_config_files_have_registered_vendors(self):
        """Every dispatch config yaml must correspond to a registered vendor."""
        yaml_files = list(_CONFIG_DIR.glob("*.yaml"))
        registered = set(VENDOR_DEVICE_MAP.keys())
        unregistered = [p.stem for p in yaml_files if p.stem not in registered]
        if unregistered:
            warnings.warn(
                f"Config file(s) without a registered vendor: {unregistered}. "
                f"Either register the vendor in VENDOR_DEVICE_MAP or remove the orphaned yaml.",
                UserWarning,
                stacklevel=1,
            )

    @pytest.mark.filterwarnings("default::UserWarning")
    def test_all_vendors_have_config_file(self):
        """Warn if any vendor in VENDOR_DEVICE_MAP is missing a dispatch config yaml."""
        existing = {p.stem for p in _CONFIG_DIR.glob("*.yaml")}
        missing = [v for v in VENDOR_DEVICE_MAP if v not in existing]
        if missing:
            warnings.warn(
                f"Missing dispatch config yaml for vendor(s): {missing}. ",
                UserWarning,
                stacklevel=1,
            )

#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Load a platform's CI config (.github/configs/<platform>.yml) and emit it
as a JSON string to the GitHub Actions step output ``config``.

Usage (in a workflow step)::

    - id: load
      run: python .github/scripts/load_config.py --platform ${{ inputs.platform }}

Downstream steps can then access the values via
``fromJson(steps.load.outputs.config)``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def merge_device_config(platform_config: dict, device: str) -> dict:
    """Return an effective config for *device* by merging platform defaults
    with per-device overrides declared under ``devices.<device>``.

    Any field present in the device block overrides the platform-level value.
    Fields absent from the device block are inherited from the platform.
    """
    base = {
        k: v
        for k, v in platform_config.items()
        if k not in ("devices", "unit_devices", "device_runner_labels")
    }
    overrides = platform_config.get("devices", {}).get(device, {})
    merged = {**base, **overrides}
    merged["device"] = device
    return merged


def build_device_configs(config: dict) -> dict[str, dict]:
    """Return per-device effective configs for all devices declared in
    ``devices:``.  Returns an empty dict for configs without that key
    (backward-compatible with platforms that use the old schema).
    """
    devices = config.get("devices", {})
    if not devices:
        return {}
    return {device: merge_device_config(config, device) for device in devices}


def load_platform_config(platform: str) -> dict:
    """Read and return the YAML config for *platform*."""
    config_path = CONFIGS_DIR / f"{platform}.yml"
    if not config_path.exists():
        print(f"::error::Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        print(
            f"::error::Invalid config (expected mapping): {config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure the platform key is present
    config.setdefault("platform", platform)
    # Backward-compat: ensure device_runner_labels is always present so
    # workflows that use the old schema can still do device_runner_labels[x].
    config.setdefault("device_runner_labels", {})
    # Derive device_runner_labels and unit_devices from the new devices: map
    # so that workflows not yet migrated to device_configs continue to work.
    devices_map = config.get("devices", {})
    if devices_map:
        for dev, dev_cfg in devices_map.items():
            if "runner_labels" in dev_cfg:
                config["device_runner_labels"].setdefault(dev, dev_cfg["runner_labels"])
    return config


def set_output(name: str, value: str) -> None:
    """Write a key=value pair to $GITHUB_OUTPUT (or print for local runs)."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            # Use heredoc delimiter for multi-line safe output
            f.write(f"{name}<<EOF\n{value}\nEOF\n")
    else:
        # Local / debug run — just print
        print(f"{name}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load platform CI config")
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform name (must match a file in .github/configs/)",
    )
    args = parser.parse_args(argv)

    config = load_platform_config(args.platform)
    device_configs = build_device_configs(config)

    config_json = json.dumps(config, separators=(",", ":"))
    device_configs_json = json.dumps(device_configs, separators=(",", ":"))
    set_output("config", config_json)
    set_output("device_configs", device_configs_json)

    # Also print a human-readable summary for the workflow log
    print(f"Platform:       {config.get('platform')}")
    print(f"Image:          {config.get('ci_image')}")
    print(f"Labels:         {config.get('runner_labels')}")
    if device_configs:
        print(f"Devices:        {list(device_configs.keys())}")
        for dev, dcfg in device_configs.items():
            print(
                f"  {dev}: labels={dcfg.get('runner_labels')} image={dcfg.get('ci_image', '(inherited)')}"
            )
    print(
        f"Config:         {config_json[:200]}{'...' if len(config_json) > 200 else ''}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

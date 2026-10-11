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
import re
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


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
    config.setdefault("device_runner_labels", {})
    if platform == "musa":
        config = resolve_musa_config(config)
    return config


def resolve_musa_config(config: dict) -> dict:
    """Validate runner variables before constructing Docker's concrete options."""
    def required(name: str) -> str:
        value = os.environ.get(name, "")
        if not value:
            raise ValueError(f"Set repository variable {name} before enabling MUSA CI")
        return value

    runtime = required("MUSA_CI_RUNTIME")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", runtime):
        raise ValueError("Invalid MUSA_CI_RUNTIME")
    physical = required("MUSA_CI_PHYSICAL_DEVICES")
    logical = required("MUSA_CI_LOGICAL_DEVICES")
    ids = physical.split(",")
    if len(ids) != 4 or any(not re.fullmatch(r"0|[1-9][0-9]*", i) for i in ids) or len({int(i) for i in ids}) != 4:
        raise ValueError("MUSA requires exactly four unique physical device IDs")
    if logical != "0,1,2,3":
        raise ValueError("MUSA_CI_LOGICAL_DEVICES must be 0,1,2,3 for the allocated container")
    pci = required("MUSA_CI_PCI_BUS_IDS")
    if len(pci.split(",")) != 4 or len(set(pci.split(","))) != 4 or any(
        not re.fullmatch(r"0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", i) for i in pci.split(",")
    ):
        raise ValueError("MUSA_CI_PCI_BUS_IDS must provide four canonical PCI IDs in logical order")

    def data_path(name: str, suffix: str = "") -> str:
        value = required(name)
        if not value.startswith("/data/") or ".." in value.split("/") or not re.fullmatch(r"[/A-Za-z0-9_.+-]+", value):
            raise ValueError(f"{name} must be an absolute /data path without shell metacharacters")
        if suffix and not value.endswith(suffix):
            raise ValueError(f"{name} must end with {suffix}")
        return value

    wheel = data_path("MUSA_CI_EMPTY_WHEEL", ".whl")
    sha = required("MUSA_CI_EMPTY_WHEEL_SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid empty wheel SHA256")
    models = {"MUSA_GATE_MODEL_27B": data_path("MUSA_GATE_MODEL_27B"),
              "MUSA_GATE_MODEL_35B_A3B": data_path("MUSA_GATE_MODEL_35B_A3B")}
    image = os.environ.get("MUSA_CI_IMAGE")
    if image:
        if not re.fullmatch(r"[A-Za-z0-9./_:@-]+", image):
            raise ValueError("Invalid MUSA_CI_IMAGE")
        config["ci_image"] = image
    profile = os.environ.get("MUSA_CI_DEPENDENCY_PROFILE", "") or "baseline"
    if profile not in {"baseline", "latest"}:
        raise ValueError("MUSA_CI_DEPENDENCY_PROFILE must be baseline or latest")
    profile_env = {"MUSA_CI_DEPENDENCY_PROFILE": profile}
    if profile == "latest":
        profile_env["MUSA_CI_DEPENDENCY_MANIFEST"] = data_path("MUSA_CI_DEPENDENCY_MANIFEST", ".json")
        dependency_sha = required("MUSA_CI_DEPENDENCY_MANIFEST_SHA256")
        if not re.fullmatch(r"[0-9a-f]{64}", dependency_sha):
            raise ValueError("Invalid MUSA dependency manifest SHA256")
        profile_env["MUSA_CI_DEPENDENCY_MANIFEST_SHA256"] = dependency_sha
    elif os.environ.get("MUSA_CI_DEPENDENCY_MANIFEST") or os.environ.get("MUSA_CI_DEPENDENCY_MANIFEST_SHA256"):
        raise ValueError("Baseline must not silently ignore a latest dependency manifest")
    env = {"MTHREADS_VISIBLE_DEVICES": physical, "MUSA_VISIBLE_DEVICES": logical,
           "MUSA_CI_PCI_BUS_IDS": pci, "MUSA_CI_EMPTY_WHEEL": wheel,
           "MUSA_CI_EMPTY_WHEEL_SHA256": sha, **models, **profile_env}
    config["container_options"] += " --runtime=" + runtime
    for key, value in env.items():
        config["container_options"] += f" --env {key}={value}"
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

    config_json = json.dumps(config, separators=(",", ":"))
    set_output("config", config_json)

    # Also print a human-readable summary for the workflow log
    print(f"Platform:  {config.get('platform')}")
    print(f"Image:     {config.get('ci_image')}")
    print(f"Labels:    {config.get('runner_labels')}")
    print(f"Config:    {config_json[:200]}{'...' if len(config_json) > 200 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Detect which platforms to test in CI.

Priority:
  1. If .github/configs/platforms.yml exists, read it and return only
     platforms with ``enabled: true``.
  2. Otherwise, fall back to auto-scanning .github/configs/*.yml,
     excluding ``template`` and ``platforms`` (the registry file itself).

Usage (in a workflow step)::

    - id: detect
      run: python3 .github/scripts/detect_platforms.py

Sets the GitHub Actions output ``platforms`` to a JSON array of platform
names, e.g. ``["cuda", "ascend"]``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
REGISTRY_FILE = CONFIGS_DIR / "platforms.yml"

# Names to exclude when falling back to auto-scan
AUTO_SCAN_EXCLUDE = {"template", "platforms"}


def from_registry() -> list[str] | None:
    """Read platforms.yml and return enabled platform names, or None if
    the file does not exist."""
    if not REGISTRY_FILE.exists():
        return None

    with open(REGISTRY_FILE) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "platforms" not in data:
        print(
            "::warning::platforms.yml exists but has no 'platforms' key",
            file=sys.stderr,
        )
        return []

    platforms = data["platforms"]
    if not isinstance(platforms, dict):
        print("::warning::platforms.yml 'platforms' is not a mapping", file=sys.stderr)
        return []

    enabled = [
        name
        for name, cfg in platforms.items()
        if isinstance(cfg, dict) and cfg.get("enabled", False)
    ]
    return enabled


def from_auto_scan() -> list[str]:
    """Scan .github/configs/*.yml and return platform names (minus exclusions)."""
    if not CONFIGS_DIR.is_dir():
        print(f"::error::Configs directory not found: {CONFIGS_DIR}", file=sys.stderr)
        return []

    return sorted(
        p.stem for p in CONFIGS_DIR.glob("*.yml") if p.stem not in AUTO_SCAN_EXCLUDE
    )


def set_output(name: str, value: str) -> None:
    """Write a key=value pair to $GITHUB_OUTPUT (or print for local runs)."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}<<EOF\n{value}\nEOF\n")
    else:
        print(f"{name}={value}")


def main() -> int:
    platforms = from_registry()

    if platforms is not None:
        source = "platforms.yml"
    else:
        print(
            "::notice::platforms.yml not found, falling back to auto-scan",
            file=sys.stderr,
        )
        platforms = from_auto_scan()
        source = "auto-scan"

    if not platforms:
        print("::warning::No platforms detected", file=sys.stderr)

    result = json.dumps(platforms)
    set_output("platforms", result)

    print(f"Source:     {source}")
    print(f"Platforms:  {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

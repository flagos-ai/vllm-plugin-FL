#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Emit the install-smoke matrix: one entry per enabled platform.

Each entry carries the same container wiring the unit tests use
(ci_image / runner_labels / container_volumes / container_options), taken
from .github/configs/<platform>.yml, so the smoke job lands on exactly the
same runner and image as the real "Install project" step.

Writes a JSON array to the ``entries`` step output for ``fromJson()``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def set_output(name: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}<<EOF\n{value}\nEOF\n")
    else:
        print(f"{name}={value}")


def main() -> int:
    registry = yaml.safe_load((CONFIGS_DIR / "platforms.yml").read_text())
    entries = []
    for platform, meta in registry.get("platforms", {}).items():
        if not meta.get("enabled", True):
            print(f"skip disabled platform: {platform}")
            continue
        config_path = CONFIGS_DIR / f"{platform}.yml"
        config = yaml.safe_load(config_path.read_text())
        entries.append(
            {
                "platform": platform,
                "ci_image": config["ci_image"],
                "runner_labels": json.dumps(config["runner_labels"]),
                "container_volumes": json.dumps(config.get("container_volumes", [])),
                "container_options": config.get("container_options", ""),
            }
        )
        print(f"platform {platform}: image={config['ci_image']}")

    set_output("entries", json.dumps({"include": entries}))
    set_output("has_entries", "true" if entries else "false")
    print(f"entries: {len(entries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright 2025 BAAI. All rights reserved.

"""Build the multi-platform supplementary-coverage CI matrix.

Reads the repository's own platform CI configs (single source of truth —
runner labels and images are never duplicated here) and emits a JSON
matrix for ``coverage_supplementary_multi.yml``.

Selection rules:

* platforms limited to the supplementary suite's four vendors
  (kunlunxin, hygon, iluvatar, thead) by default;
* a platform is dropped when ``.github/configs/platforms.yml`` disables it;
* device resolution: ``unit_devices[0]`` from the CI config when present,
  otherwise the first device section of ``tests/platforms/<platform>.yaml``.

Emits the matrix JSON to ``$GITHUB_OUTPUT`` (``matrix=...``) when running
under Actions, and always prints it to stdout for local runs.

Usage::

    python .github/scripts/generate_coverage_matrix.py
    python .github/scripts/generate_coverage_matrix.py --platforms hygon,thead
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

DEFAULT_PLATFORMS = ("kunlunxin", "hygon", "iluvatar", "thead")


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def platform_enabled(registry: dict, platform: str) -> bool:
    entry = registry.get("platforms", {}).get(platform, {})
    return bool(entry.get("enabled", False))


def resolve_device(ci_cfg: dict, platform: str) -> str:
    # 1) explicit unit_devices from the CI config (hygon, thead)
    unit_devices = ci_cfg.get("unit_devices")
    if isinstance(unit_devices, list) and unit_devices:
        return str(unit_devices[0])
    # 2) first device section of the platform YAML (kunlunxin, iluvatar)
    platform_yaml = REPO_ROOT / "tests" / "platforms" / f"{platform}.yaml"
    cfg = load_yaml(platform_yaml)
    for key, value in cfg.items():
        if key in ("platform", "vendor", "device_types", "tolerance",
                   "device_overrides", "unsupported_features", "env_defaults"):
            continue
        if isinstance(value, dict) and "tests" in value:
            return str(key)
    raise SystemExit(
        f"[matrix] error: cannot resolve device for platform '{platform}' "
        f"from {ci_cfg.get('platform', platform)} config or {platform_yaml}"
    )


def build_matrix(platforms: list[str]) -> list[dict]:
    registry = load_yaml(REPO_ROOT / ".github" / "configs" / "platforms.yml")
    entries = []
    for platform in platforms:
        if not platform_enabled(registry, platform):
            print(f"[matrix] skipping {platform}: disabled in platforms.yml")
            continue
        ci_cfg = load_yaml(
            REPO_ROOT / ".github" / "configs" / f"{platform}.yml"
        )
        entry = {
            "platform": platform,
            "device": resolve_device(ci_cfg, platform),
            "image": ci_cfg["ci_image"],
            "runner_labels": json.dumps(ci_cfg.get("runner_labels", [])),
            "container_volumes": json.dumps(
                ci_cfg.get("container_volumes", [])
            ),
            "container_options": ci_cfg.get("container_options", "") or "",
        }
        entries.append(entry)
        print(
            f"[matrix]   - {entry['platform']} device={entry['device']} "
            f"labels={entry['runner_labels']}"
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platforms",
        default=",".join(DEFAULT_PLATFORMS),
        help="comma-separated platform list (default: all four)",
    )
    args = parser.parse_args()

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    entries = build_matrix(platforms)
    matrix = json.dumps(entries)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"matrix={matrix}\n")
    print(matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Detect which platforms to test in CI.

Priority:
  1. If .github/configs/platforms.yml exists, read it and return only
     platforms with ``enabled: true``, filtered by diff-based routing
     when ``--changed-files`` is provided.
  2. Otherwise, fall back to auto-scanning .github/configs/*.yml,
     excluding ``template`` and ``platforms`` (the registry file itself).

Diff-based routing logic (when --changed-files is given):
  - If any changed file matches ``global_trigger_paths`` → return all enabled
    platforms (full run). Paths carved out by a platform's
    ``exempt_from_global`` never count as global triggers — they route only
    via the owning platform's ``trigger_paths``.
  - Otherwise, return only platforms whose ``trigger_paths`` match at least
    one changed file.
  - If no platform matches → return empty list (only lint+build will run).
  - Platforms with empty ``trigger_paths`` are always included when enabled.

Usage (in a workflow step)::

    - id: detect
      run: |
        python3 .github/scripts/detect_platforms.py \\
          --changed-files /tmp/changed_files.txt

Sets the GitHub Actions output ``platforms`` to a JSON array of platform
names, e.g. ``["nvidia", "ascend"]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
REGISTRY_FILE = CONFIGS_DIR / "platforms.yml"

# Names to exclude when falling back to auto-scan
AUTO_SCAN_EXCLUDE = {"template", "platforms"}


def _validate_and_collect_exemptions(
    enabled: dict, global_paths: list[str]
) -> list[str]:
    """Collect exempt_from_global prefixes from enabled platforms.

    Guards two-list drift in both directions:
      error   - exempt entry missing from the same platform's trigger_paths
                (files matching it would route to NO platform -> exit 2).
      warning - trigger_path under a global prefix but not exempted
                (changes there trigger a full run -> over-trigger only).
      warning - exempt entry not under any global prefix (no-op / stale).
    """
    exempt_paths: list[str] = []
    for name, cfg in enabled.items():
        triggers: list[str] = cfg.get("trigger_paths", [])
        exempts: list[str] = cfg.get("exempt_from_global", [])
        for tp in triggers:
            if tp not in exempts and any(tp.startswith(gp) for gp in global_paths):
                print(
                    f"::warning::platform '{name}': trigger_path '{tp}' is under "
                    f"a global path but not in exempt_from_global — changes "
                    f"there will trigger a full run",
                    file=sys.stderr,
                )
        for ep in exempts:
            if ep not in triggers:
                print(
                    f"::error::platform '{name}': exempt_from_global entry "
                    f"'{ep}' is not in trigger_paths — files matching it "
                    f"would route to no platform",
                    file=sys.stderr,
                )
                sys.exit(2)
            if not any(ep.startswith(gp) or gp.startswith(ep) for gp in global_paths):
                print(
                    f"::warning::platform '{name}': exempt_from_global entry "
                    f"'{ep}' is not under any global path (no-op)",
                    file=sys.stderr,
                )
            exempt_paths.append(ep)
    return exempt_paths


def from_registry(changed_files: list[str] | None = None) -> list[str] | None:
    """Read platforms.yml and return enabled platform names, filtered by
    diff-based routing when changed_files is provided. Returns None if the
    file does not exist."""
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

    enabled = {
        name: cfg
        for name, cfg in platforms.items()
        if isinstance(cfg, dict) and cfg.get("enabled", False)
    }

    if not enabled:
        return []

    global_paths: list[str] = data.get("global_trigger_paths", [])
    exempt_paths = _validate_and_collect_exemptions(enabled, global_paths)

    # No diff filtering → return all enabled platforms (nightly/weekly mode)
    if changed_files is None:
        return list(enabled.keys())

    # Diff-based routing. Two keys, two independent questions:
    #   - exempt_from_global: vendor carve-out; a changed file under one of
    #     these prefixes does NOT count toward global_trigger_paths.
    #   - trigger_paths: answers only "does this platform run?" and has no
    #     effect on the global trigger.
    for f in changed_files:
        if any(f == ep or f.startswith(ep) for ep in exempt_paths):
            continue  # vendor carve-out: not a global trigger
        for gp in global_paths:
            if f.startswith(gp):
                print(
                    f"[detect] '{f}' matches global path '{gp}' → all platforms",
                    file=sys.stderr,
                )
                return list(enabled.keys())

    # Per-platform matching
    triggered: list[str] = []
    for name, cfg in enabled.items():
        trigger_paths: list[str] = cfg.get("trigger_paths", [])

        # Empty trigger_paths → always include when enabled
        if not trigger_paths:
            triggered.append(name)
            continue

        matched = False
        for f in changed_files:
            for tp in trigger_paths:
                if f.startswith(tp) or f == tp:
                    print(
                        f"[detect] '{f}' matches '{tp}' → include platform '{name}'",
                        file=sys.stderr,
                    )
                    matched = True
                    break
            if matched:
                break

        if matched:
            triggered.append(name)

    return triggered


def from_auto_scan() -> list[str]:
    """Scan .github/configs/*.yml and return platform names (minus exclusions)."""
    if not CONFIGS_DIR.is_dir():
        print(f"::error::Configs directory not found: {CONFIGS_DIR}", file=sys.stderr)
        return []

    return sorted(
        p.stem for p in CONFIGS_DIR.glob("*.yml") if p.stem not in AUTO_SCAN_EXCLUDE
    )


def load_changed_files(path: str) -> list[str] | None:
    """Read a newline-delimited file of changed paths. Returns None if empty."""
    p = Path(path)
    if not p.exists():
        print(f"::warning::changed-files path not found: {path}", file=sys.stderr)
        return None
    lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return lines or None


def set_output(name: str, value: str) -> None:
    """Write a key=value pair to $GITHUB_OUTPUT (or print for local runs)."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}<<EOF\n{value}\nEOF\n")
    else:
        print(f"{name}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect CI platforms")
    parser.add_argument(
        "--changed-files",
        metavar="PATH",
        help="Newline-delimited file of changed paths for diff-based routing",
    )
    args = parser.parse_args(argv)

    changed: list[str] | None = None
    if args.changed_files:
        changed = load_changed_files(args.changed_files)
        if changed is not None:
            print(f"[detect] Changed files ({len(changed)}):", file=sys.stderr)
            for f in changed[:20]:
                print(f"  {f}", file=sys.stderr)
            if len(changed) > 20:
                print(f"  ... and {len(changed) - 20} more", file=sys.stderr)
        else:
            print("[detect] No changed files → full run", file=sys.stderr)

    platforms = from_registry(changed)

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

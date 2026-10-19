#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Select CI platforms from configuration, PR changes, labels, or dispatch input.

Selection is conservative: shared, unknown, or incomplete metadata runs every
enabled platform. Documentation-only changes run no hardware jobs. Explicit
``platform:<name>`` labels and workflow-dispatch input override path routing.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
REGISTRY_FILE = CONFIGS_DIR / "platforms.yml"
AUTO_SCAN_EXCLUDE = {"template", "platforms"}
DOC_PATTERNS = (
    "*.md",
    "*.rst",
    "*.txt",
    "**/*.md",
    "**/*.rst",
    "docs/**",
    "examples/**",
    "LICENSE",
    ".github/ISSUE_TEMPLATE/**",
    ".github/PULL_REQUEST_TEMPLATE.md",
)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return value if isinstance(value, dict) else {}


def enabled_platforms() -> list[str]:
    if REGISTRY_FILE.exists():
        registry = _load_yaml(REGISTRY_FILE).get("platforms", {})
        if not isinstance(registry, dict):
            raise ValueError("platforms.yml 'platforms' must be a mapping")
        return [
            name
            for name, config in registry.items()
            if isinstance(config, dict) and config.get("enabled", False)
        ]
    return sorted(
        path.stem
        for path in CONFIGS_DIR.glob("*.yml")
        if path.stem not in AUTO_SCAN_EXCLUDE
    )


def _matches(path: str, pattern: str) -> bool:
    normalized = path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")
    if pattern.endswith("/"):
        return normalized.startswith(pattern)
    return fnmatch.fnmatch(normalized, pattern)


def watch_patterns(platform: str) -> list[str] | None:
    config_path = CONFIGS_DIR / f"{platform}.yml"
    if not config_path.exists():
        return None
    patterns = _load_yaml(config_path).get("paths_watch")
    if not isinstance(patterns, list) or not patterns:
        return None
    return [str(pattern) for pattern in patterns]


def parse_requested(raw: str, enabled: list[str]) -> list[str] | None:
    raw = raw.strip().lower()
    if not raw:
        return None
    values = [item for item in raw.replace(",", " ").split() if item]
    if values in (["all"], ["*"]):
        return enabled
    invalid = sorted(set(values) - set(enabled))
    if invalid:
        raise ValueError(f"unknown platform(s): {', '.join(invalid)}")
    return [name for name in enabled if name in values]


def parse_label_override(raw: str, enabled: list[str]) -> list[str] | None:
    try:
        labels = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid labels JSON: {exc}") from exc
    routes = {
        str(label).removeprefix("platform:").lower()
        for label in labels
        if str(label).lower().startswith("platform:")
    }
    if not routes:
        return None
    if "all" in routes:
        return enabled
    invalid = sorted(routes - set(enabled))
    if invalid:
        print(
            f"::warning::Unknown platform label(s) {invalid}; running all platforms",
            file=sys.stderr,
        )
        return enabled
    return [name for name in enabled if name in routes]


def select_by_changes(changed: list[str], enabled: list[str]) -> list[str]:
    if not changed:
        print("::warning::No changed-file metadata; running all platforms")
        return enabled
    if all(
        any(_matches(path, pattern) for pattern in DOC_PATTERNS) for path in changed
    ):
        return []

    watches = {name: watch_patterns(name) for name in enabled}
    if any(patterns is None for patterns in watches.values()):
        missing = [name for name, patterns in watches.items() if patterns is None]
        print(
            f"::warning::Missing paths_watch for {missing}; running all platforms",
            file=sys.stderr,
        )
        return enabled

    selected: set[str] = set()
    for path in changed:
        owners = {
            name
            for name, patterns in watches.items()
            if patterns and any(_matches(path, pattern) for pattern in patterns)
        }
        if not owners:
            print(f"Unmapped/shared path '{path}' -> all platforms")
            return enabled
        selected.update(owners)
    return [name for name in enabled if name in selected]


def set_output(name: str, value: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"{name}={value}\n")
    else:
        print(f"{name}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-name", default=os.getenv("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--changed-files")
    parser.add_argument("--labels-json", default="[]")
    parser.add_argument("--platforms", default="")
    args = parser.parse_args(argv)

    enabled = enabled_platforms()
    try:
        selected = parse_requested(args.platforms, enabled)
        source = "dispatch input"
        if selected is None:
            selected = parse_label_override(args.labels_json, enabled)
            source = "PR labels"
        if selected is None and args.event_name == "pull_request":
            changed = None
            if args.changed_files and Path(args.changed_files).exists():
                changed = [
                    line.strip()
                    for line in Path(args.changed_files)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
            selected = select_by_changes(changed or [], enabled)
            source = "changed files"
        if selected is None:
            selected = enabled
            source = "full coverage event"
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    result = json.dumps(selected, separators=(",", ":"))
    set_output("platforms", result)
    print(f"Enabled:  {json.dumps(enabled)}")
    print(f"Selected: {result}")
    print(f"Source:   {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

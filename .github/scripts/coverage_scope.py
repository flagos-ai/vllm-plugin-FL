#!/usr/bin/env python3
"""Create a platform-effective coverage.py JSON report.

The native coverage.py report remains the full-package baseline.  The
effective report removes backend directories that cannot be executed by the
selected platform while retaining all shared code, including platform-specific
branches in shared modules such as ``vllm_fl/platform.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(".github/configs/coverage_scope.yml")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by CI setup
        raise SystemExit(
            "PyYAML is required to compute platform-effective coverage"
        ) from exc

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Coverage scope config must be a mapping: {path}")
    return config


def _platform_scope(config: dict[str, Any], platform: str) -> tuple[str | None, dict[str, Any]]:
    platforms = config.get("platforms")
    if not isinstance(platforms, dict) or platform not in platforms:
        known = ", ".join(sorted(platforms or {}))
        raise SystemExit(
            f"Unknown coverage platform {platform!r}; configured platforms: {known}"
        )
    entry = platforms[platform]
    if not isinstance(entry, dict):
        raise SystemExit(f"Coverage platform {platform!r} has no mapping")
    backend = entry.get("backend")
    mode = entry.get("mode", "backend_plus_shared")
    if backend is None and mode != "shared_only":
        raise SystemExit(
            f"Coverage platform {platform!r} has no backend mapping or shared_only mode"
        )
    selected_backend = None if mode == "shared_only" else str(backend)
    return selected_backend, entry


def _vendor_name(path: str, vendor_root: str) -> str | None:
    normalized = path.replace("\\", "/")
    marker = f"/{vendor_root.strip('/')}/"
    rooted = f"/{normalized.lstrip('/')}"
    if marker not in rooted:
        return None

    suffix = rooted.split(marker, 1)[1]
    name = suffix.split("/", 1)[0]
    # The package __init__.py belongs to shared dispatch scaffolding.  It is
    # not an implementation directory and remains in every denominator.
    return None if "/" not in suffix else name


def _sum_totals(files: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for file_data in files.values():
        summary = file_data.get("summary", {})
        if not isinstance(summary, dict):
            continue
        for key in (
            "covered_lines",
            "num_statements",
            "missing_lines",
            "excluded_lines",
            "covered_branches",
            "num_branches",
            "missing_branches",
            "covered_functions",
            "num_functions",
            "missing_functions",
        ):
            if key in summary:
                totals[key] = totals.get(key, 0) + int(summary[key])

    statements = totals.get("num_statements", 0)
    covered_lines = totals.get("covered_lines", 0)
    percent = 100.0 * covered_lines / statements if statements else 100.0
    totals["percent_covered"] = percent
    totals["percent_covered_display"] = f"{percent:.0f}"

    branches = totals.get("num_branches", 0)
    if branches:
        branch_percent = 100.0 * totals.get("covered_branches", 0) / branches
        totals["percent_covered_branches"] = branch_percent
        totals["percent_covered_branches_display"] = f"{branch_percent:.0f}"

    functions = totals.get("num_functions", 0)
    if functions:
        function_percent = 100.0 * totals.get("covered_functions", 0) / functions
        totals["percent_covered_functions"] = function_percent
        totals["percent_covered_functions_display"] = f"{function_percent:.0f}"

    return totals


def _copy_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--scope", default="unknown")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--effective-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    vendor_root = str(config.get("vendor_root", "vllm_fl/dispatch/backends/vendor"))
    backend, platform_scope = _platform_scope(config, args.platform.lower())

    with args.input.open(encoding="utf-8") as stream:
        full_report = json.load(stream)
    if not isinstance(full_report, dict) or not isinstance(full_report.get("files"), dict):
        raise SystemExit(f"Not a coverage.py JSON report: {args.input}")

    full_files = full_report["files"]
    selected_files: dict[str, Any] = {}
    excluded: dict[str, dict[str, int]] = {}
    for path, file_data in full_files.items():
        vendor = _vendor_name(path, vendor_root)
        if vendor is None or vendor == backend:
            selected_files[path] = file_data
            continue

        summary = file_data.get("summary", {})
        bucket = excluded.setdefault(
            vendor,
            {"files": 0, "num_statements": 0, "covered_lines": 0, "missing_lines": 0},
        )
        bucket["files"] += 1
        for key in ("num_statements", "covered_lines", "missing_lines"):
            bucket[key] += int(summary.get(key, 0))

    effective_report = dict(full_report)
    effective_report["files"] = selected_files
    effective_report["totals"] = _sum_totals(selected_files)

    metadata = {
        "schema_version": "flagos.coverage.scope/v1",
        "platform": args.platform.lower(),
        "runtime_backend": platform_scope.get("backend"),
        "resolved_backend": backend,
        "runtime_alias": platform_scope.get("runtime_alias"),
        "scope_mode": platform_scope.get("mode", "backend_plus_shared"),
        "notes": platform_scope.get("notes"),
        "scope": args.scope,
        "source_sha": args.source_sha,
        "input_report": str(args.input),
        # This is the only authoritative total for the platform metric.  The
        # full-package baseline is retained as a separate JSON artifact.
        "totals": effective_report["totals"],
        "excluded_vendor_directories": excluded,
        "scope_rule": (
            "all vllm_fl files plus the selected backend directory; "
            "other vendor directories are excluded"
            if backend
            else "all vllm_fl files except vendor backend directories"
        ),
    }

    _copy_json(full_report, args.full_output)
    _copy_json(effective_report, args.effective_output)
    _copy_json(metadata, args.metadata_output)

    print(
        f"[coverage] {args.platform}: backend={backend or 'shared-only'}, "
        f"effective={effective_report['totals']['percent_covered']:.2f}%, "
        f"excluded_vendor_dirs={len(excluded)}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Generate test matrices from a platform config.

Reads ``tests/platforms/<platform>.yaml``, expands the per-device test
definitions into flat JSON arrays, and writes them to ``$GITHUB_OUTPUT``
so that downstream GitHub Actions jobs can use ``fromJson()`` to fan out.

Usage (in a workflow step)::

    - id: matrix
      run: python .github/scripts/generate_matrix.py --platform ${{ inputs.platform }}

Outputs:
    functional — JSON array of ``{device, timeout}`` objects (one per device, runs all functional tests).
    e2e        — JSON array of ``{task, device, timeout}`` objects grouped by (task, device).
    unit       — JSON array of ``{device, include, exclude}`` objects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORMS_DIR = REPO_ROOT / "tests" / "platforms"
sys.path.insert(0, str(REPO_ROOT))

# YAML task key → test directory name
TASK_DIR_MAP: dict[str, str] = {
    "serve": "serving",
}

# Default timeout (minutes) per task category
DEFAULT_TIMEOUT: dict[str, int] = {
    "inference": 60,
    "serving": 60,
}


def load_platform(platform: str) -> dict:
    """Load and return the platform YAML config."""
    path = PLATFORMS_DIR / f"{platform}.yaml"
    if not path.exists():
        print(f"::error::Platform config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_task_dir(task_key: str) -> str:
    """Map a YAML task key to the corresponding test directory name."""
    return TASK_DIR_MAP.get(task_key, task_key)


def resolve_e2e_timeout(config: dict, device: str, task_dir: str) -> int:
    """Return timeout for an E2E task, allowing platform YAML overrides."""
    default = DEFAULT_TIMEOUT.get(task_dir, 60)
    timeouts = (
        config.get(device, {}).get("tests", {}).get("timeouts", {}).get("e2e", {})
    )
    value = timeouts.get(task_dir, default)
    return int(value)


def get_device_sections(config: dict) -> list[str]:
    """Return device section names present in the config.

    Device sections are top-level keys that have a nested ``tests`` mapping.
    """
    devices = []
    for key, value in config.items():
        if isinstance(value, dict) and "tests" in value:
            devices.append(key)
    return devices


def build_e2e_matrix(
    config: dict,
    devices: list[str],
    unsupported: list[str],
) -> list[dict]:
    """End-to-end model tests (inference, serving).

    These are slow and require model files.  Entries are grouped by
    ``(task, device)`` so that all model/case combos for the same task
    run inside a single job, avoiding repeated container startup and
    project installation.
    """
    # Collect per-(task, device) groups
    groups: dict[tuple[str, str], list[dict]] = {}
    for device in devices:
        section = config[device]
        e2e = section.get("tests", {}).get("e2e", {})

        for task_key, models in e2e.items():
            if not isinstance(models, dict):
                continue
            task_dir = resolve_task_dir(task_key)

            for model, cases in models.items():
                if any(feat in model for feat in unsupported):
                    continue

                if not isinstance(cases, list):
                    cases = [cases]

                key = (task_dir, device)
                groups.setdefault(key, [])
                for case in cases:
                    groups[key].append({"model": model, "case": str(case)})

    # Build one matrix entry per (task, device) group
    entries = []
    for (task_dir, device), case_list in groups.items():
        timeout = resolve_e2e_timeout(config, device, task_dir)
        entries.append(
            {
                "task": task_dir,
                "device": device,
                "cases": json.dumps(case_list, separators=(",", ":")),
                "timeout": timeout,
            }
        )
    return entries


def check_model_paths(
    platform: str,
    device: str | None = None,
    cases: str | None = None,
) -> int:
    """Validate that the selected CI matrix model directories exist."""
    from tests.utils.model_config import ModelConfig

    if not cases:
        print("[check-models] No E2E model cases selected.")
        return 0

    selected_cases = json.loads(cases)
    if not isinstance(selected_cases, list):
        print(
            "::error title=Invalid cases::--cases must be a JSON array", file=sys.stderr
        )
        return 1

    missing = 0
    print(f"[check-models] Platform: {platform}")
    print(f"[check-models] Device:   {device or ''}")
    print(f"[check-models] Cases:    {len(selected_cases)}")

    for case_entry in selected_cases:
        model = str(case_entry["model"])
        case = str(case_entry["case"])
        try:
            model_cfg = ModelConfig.load(
                model,
                case,
                platform=platform,
                device=device or None,
            )
        except Exception as exc:
            message = f"{model}/{case}: {exc}"
            print(
                f"::error title=Invalid model config::{message}",
                file=sys.stderr,
            )
            missing += 1
            continue

        model_path = model_cfg.model
        exists = bool(model_path) and Path(model_path).exists()
        status = "OK" if exists else "MISSING"
        print(f"[check-models] {status}: {model}/{case} -> {model_path}")
        if not exists:
            message = f"{model}/{case} requires model path: {model_path}"
            print(
                f"::error file=tests/models/{model}/{case}.yaml,"
                f"title=Missing test model::{message}",
                file=sys.stderr,
            )
            missing += 1

    if missing:
        print(f"[check-models] Missing or invalid model cases: {missing}")
        return 1

    print("[check-models] All selected model paths exist.")
    return 0


# ---------------------------------------------------------------------------
# Changed-file filtering for PR-only smart skip
# ---------------------------------------------------------------------------

# Any changed file starting with one of these prefixes triggers a full e2e run.
_FULL_RUN_PREFIXES: list[str] = [
    "vllm_fl/",
    "csrc/",
    "setup.py",
    "pyproject.toml",
    "requirements/",
    "tests/e2e_tests/",
    "tests/utils/",
    "tests/run.py",
    "tests/platforms/",
    "tests/conftest.py",
    ".github/",
]


def load_changed_files(path: str) -> list[str] | None:
    """Read a newline-delimited file of changed paths.

    Returns ``None`` if the file is empty (signals "run everything").
    """
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines or None


def filter_e2e_by_changes(
    entries: list[dict],
    changed_files: list[str],
) -> list[dict]:
    """Narrow the e2e matrix when only model config files changed.

    Rules:
    - If *any* changed file matches a full-run prefix → return all entries.
    - If all changes are under ``tests/models/<model>/`` → keep only entries
      whose cases reference an affected model.
    - Unknown paths (not matching any known prefix) → full run for safety.
    """
    for f in changed_files:
        if any(f.startswith(p) for p in _FULL_RUN_PREFIXES):
            print("[filter] Core/CI file changed → full e2e run")
            return entries

    affected_models: set[str] = set()
    for f in changed_files:
        if f.startswith("tests/models/"):
            parts = f.split("/")
            if len(parts) >= 3:
                affected_models.add(parts[2])
        else:
            # Unknown path outside known safe-to-ignore dirs → full run
            print(f"[filter] Unknown path '{f}' → full e2e run")
            return entries

    if not affected_models:
        return entries

    print(f"[filter] Only model configs changed, affected: {sorted(affected_models)}")

    filtered: list[dict] = []
    for entry in entries:
        cases = json.loads(entry["cases"])
        kept = [c for c in cases if c["model"] in affected_models]
        if kept:
            filtered.append({**entry, "cases": json.dumps(kept, separators=(",", ":"))})
    return filtered


def build_unit_matrix(config: dict, devices: list[str]) -> list[dict]:
    """Expand per-device unit test definitions into a flat matrix."""
    matrix = []
    for device in devices:
        section = config[device]
        unit = section.get("tests", {}).get("unit", {})
        if not unit:
            continue
        matrix.append(
            {
                "device": device,
                "include": unit.get("include", "*"),
                "exclude": unit.get("exclude", []),
            }
        )
    return matrix


def set_output(name: str, value: str) -> None:
    """Write a key=value pair to $GITHUB_OUTPUT (or print for local runs)."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}<<EOF\n{value}\nEOF\n")
    else:
        print(f"{name}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CI test matrices")
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform name (must match tests/platforms/<platform>.yaml)",
    )
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Path to a newline-delimited file listing changed paths. "
        "When provided and non-empty, the e2e matrix is filtered to only "
        "include tests affected by those changes (PR smart-skip).",
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help=(
            "Validate that selected E2E model paths exist instead of "
            "generating matrices."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device name to check when --check-models is set.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help=(
            "JSON array of {model, case} entries to check when --check-models is set."
        ),
    )
    args = parser.parse_args(argv)

    if args.check_models:
        return check_model_paths(
            args.platform,
            device=args.device,
            cases=args.cases,
        )

    config = load_platform(args.platform)
    devices = get_device_sections(config)
    unsupported = config.get("unsupported_features", [])

    if not devices:
        print(f"::warning::No device sections found in {args.platform}.yaml")

    e2e_matrix = build_e2e_matrix(config, devices, unsupported)
    unit_matrix = build_unit_matrix(config, devices)

    # Apply PR smart-skip filtering when changed files are provided
    if args.changed_files:
        changed = load_changed_files(args.changed_files)
        if changed:
            e2e_matrix = filter_e2e_by_changes(e2e_matrix, changed)

    e2e_json = json.dumps(e2e_matrix, separators=(",", ":"))
    unit_json = json.dumps(unit_matrix, separators=(",", ":"))

    set_output("e2e", e2e_json)
    set_output("unit", unit_json)

    # Human-readable summary for CI logs
    print(f"Platform:    {args.platform}")
    print(f"Devices:     {devices}")
    print(f"E2E:         {len(e2e_matrix)} job(s)")
    for entry in e2e_matrix:
        case_list = json.loads(entry["cases"])
        print(
            f"  - {entry['task']} (device={entry['device']}, "
            f"timeout={entry['timeout']}m, {len(case_list)} case(s))"
        )
        for c in case_list:
            print(f"      {c['model']}/{c['case']}")
    print(f"Unit:        {len(unit_matrix)} config(s)")
    for entry in unit_matrix:
        print(
            f"  - device={entry['device']} "
            f"include={entry['include']} exclude={entry['exclude']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

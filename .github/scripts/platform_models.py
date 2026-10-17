#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DEBUG PROBE (all-platform model heal): resolve a platform's model-file
dependencies into a download plan, derived from the case files (no hardcoded
model lists, so the plan cannot drift from the tests).

Emits tab-separated plan lines:

  DL<TAB><repo-id><TAB><dest-path>    idempotent download via download_model.py
  SCAN<TAB><dest-path><TAB><reason>   report-only: identity unclear, or the
                                      platform's /data/models is a shared mount
                                      (writing GBs into shared storage is a
                                      decision, not a default)

Sources of truth::

  tests/platforms/<p>.yaml     device sections -> tests.e2e.{inference,serving}
                               + tests.benchmark (smoke cases) when enabled
  tests/models/<m>/<c>.yaml    llm.model -> local model path
  tests/benchmarks/configs/*   benchmark smoke case list
  .github/configs/<p>.yml      container_volumes -> shared-mount detection

Usage::

    python .github/scripts/platform_models.py --platform cuda [--device a100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = "/data/models"


def load(path: Path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def device_sections(platform_yaml: dict) -> list[str]:
    """Keys of sections that look like a device (carry a .tests mapping)."""
    return [
        key
        for key, sec in platform_yaml.items()
        if isinstance(sec, dict) and isinstance(sec.get("tests"), dict)
    ]


def e2e_cases(platform_yaml: dict, device: str) -> set[tuple[str, str]]:
    """(model, case) pairs for the device's e2e inference+serving tests."""
    tests = platform_yaml[device].get("tests") or {}
    e2e = tests.get("e2e") or {}
    pairs: set[tuple[str, str]] = set()
    for task in ("inference", "serving"):
        for model, cases in (e2e.get(task) or {}).items():
            for case in cases or []:
                pairs.add((model, case))
    return pairs


def benchmark_cases(platform_yaml: dict, device: str) -> set[tuple[str, str]]:
    """(model, case) pairs from the device's benchmark smoke config, if enabled."""
    benchmark = ((platform_yaml[device].get("tests") or {}).get("benchmark")) or {}
    if not benchmark.get("enabled"):
        return set()
    config = load(REPO_ROOT / benchmark.get("config", "tests/benchmarks/configs/smoke.yaml"))
    pairs: set[tuple[str, str]] = set()
    for section in config.values():
        if not isinstance(section, list):
            continue
        for entry in section:
            if isinstance(entry, dict) and entry.get("model") and entry.get("case"):
                pairs.add((entry["model"], entry["case"]))
    return pairs


def case_model_path(model: str, case: str) -> str | None:
    case_file = REPO_ROOT / "tests" / "models" / model / f"{case}.yaml"
    if not case_file.exists():
        print(f"[platform-models] warning: case file missing: {case_file}", file=sys.stderr)
        return None
    llm = load(case_file).get("llm") or {}
    path = llm.get("model")
    if not path or not str(path).startswith(MODELS_ROOT):
        return None
    return str(path)


def shared_mount(platform: str) -> bool:
    config = load(REPO_ROOT / ".github" / "configs" / f"{platform}.yml")
    for vol in config.get("container_volumes") or []:
        source = str(vol).split(":", 1)[0]
        if source.startswith("/mnt/"):
            return True
    return False


def plan_line(path: str, platform: str) -> str | None:
    """Turn a local model path into a DL or SCAN plan line."""
    if shared_mount(platform):
        return f"SCAN\t{path}\tshared mount ({platform})"
    tail = path[len(MODELS_ROOT) :].strip("/")
    if "/" not in tail:
        # No org component (e.g. /data/models/MiniCPM) — the hub repo id is a
        # naming choice someone made by hand; confirm identity before pulling.
        return f"SCAN\t{path}\trepo id not derivable from path"
    return f"DL\t{tail}\t{path}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument(
        "--device",
        default="",
        help="restrict to one device section; default = first device with e2e cases "
        "(the default-label pool's device)",
    )
    args = parser.parse_args()

    platform_yaml = load(REPO_ROOT / "tests" / "platforms" / f"{args.platform}.yaml")
    devices = device_sections(platform_yaml)
    if args.device:
        device = args.device
        if device not in devices:
            print(f"[platform-models] error: no device section '{device}'", file=sys.stderr)
            return 2
    else:
        device = next((d for d in devices if e2e_cases(platform_yaml, d)), devices[0])
        print(f"[platform-models] device: {device} (first with e2e cases)", file=sys.stderr)

    pairs = e2e_cases(platform_yaml, device) | benchmark_cases(platform_yaml, device)
    paths = sorted({p for m, c in pairs if (p := case_model_path(m, c))})
    for path in paths:
        line = plan_line(path, args.platform)
        if line:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

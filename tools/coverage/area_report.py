#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Per-area coverage aggregation from a coverage.py JSON report.

Reads a report (baseline, supplementary, or combined), applies the reporting
scope from a scope JSON, and emits:

* ``coverage-<tag>-areas.json`` — machine-readable per-area numbers;
* ``coverage-<tag>-areas.md`` — a markdown table for the job summary.

An area's scope value may be a single path prefix (string) or a list of
prefixes; a file is claimed by the first area (longest-prefix order) any of
whose prefixes matches. This lets e.g. a ``platform_config`` area span
``vllm_fl/configs/``, ``vllm_fl/platform.py`` and
``vllm_fl/nvidia_platform.py`` at once.

Usage::

    python tools/coverage/area_report.py \
        --report coverage-kunlunxin-combined.json \
        --scope tools/coverage/scope_kunlunxin.json \
        --tag combined \
        --output-json coverage-kunlunxin-areas.json \
        --output-md coverage-kunlunxin-areas.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rel(path_key: str) -> str:
    p = path_key.replace("\\", "/")
    idx = p.find("vllm_fl/")
    return p[idx:] if idx != -1 else p


def _prefixes(scope_value) -> list[str]:
    """Normalize an area's scope value to a list of path prefixes."""
    if isinstance(scope_value, str):
        return [scope_value]
    return list(scope_value)


def aggregate(report: dict, scope: dict) -> dict[str, dict]:
    """Aggregate covered/total statements per area (longest-prefix wins)."""
    # sort area prefixes longest-first so the most specific area claims a file
    areas = sorted(
        (
            (name, _prefixes(value))
            for name, value in scope.get("areas", {}).items()
        ),
        key=lambda kv: max(len(p) for p in kv[1]),
        reverse=True,
    )
    exclude = scope.get("reporting_exclude", [])

    stats: dict[str, dict[str, int]] = {
        name: {"covered": 0, "statements": 0} for name, _ in areas
    }
    stats["_unassigned"] = {"covered": 0, "statements": 0}

    for key, info in report.get("files", {}).items():
        rel = _rel(key)
        if any(rel.startswith(prefix) for prefix in exclude):
            continue
        summary = info.get("summary", {})
        covered = summary.get("covered_lines", 0)
        stmts = summary.get("num_statements", 0)
        for name, prefixes in areas:
            # dispatch.core (vllm_fl/dispatch/) must not swallow
            # dispatch.backends.* sub-areas: longest prefix is checked first.
            if any(rel.startswith(prefix) for prefix in prefixes):
                stats[name]["covered"] += covered
                stats[name]["statements"] += stmts
                break
        else:
            stats["_unassigned"]["covered"] += covered
            stats["_unassigned"]["statements"] += stmts

    out: dict[str, dict] = {}
    for name, s in stats.items():
        pct = (100.0 * s["covered"] / s["statements"]) if s["statements"] else 0.0
        out[name] = {
            "covered_lines": s["covered"],
            "num_statements": s["statements"],
            "percent_covered": round(pct, 2),
        }
    return out


def to_markdown(areas: dict[str, dict], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| area | covered | statements | percent |",
        "|---|---:|---:|---:|",
    ]
    for name, s in sorted(areas.items()):
        lines.append(
            f"| {name} | {s['covered_lines']} | {s['num_statements']} | "
            f"{s['percent_covered']:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--tag", default="report")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    report = _load(args.report)
    scope = _load(args.scope)

    areas = aggregate(report, scope)
    payload = {
        "tag": args.tag,
        "commit_sha": report.get("meta", {}).get("commit_sha", ""),
        "source_report": str(args.report),
        "scope_file": str(args.scope),
        "areas": areas,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    md = to_markdown(areas, f"Per-area coverage — {args.tag}")
    args.output_md.write_text(md, encoding="utf-8")
    print(f"[areas] wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

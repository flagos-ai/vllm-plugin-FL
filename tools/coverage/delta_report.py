#!/usr/bin/env python3
# Copyright 2025 BAAI. All rights reserved.

"""Per-area coverage delta between the core baseline and a combined report.

Consumes two per-area reports produced by ``area_report.py`` (same scope
file, same commit) and emits the per-area and overall core → combined delta:

* ``coverage-<platform>-delta.json`` — machine-readable deltas;
* ``coverage-<platform>-delta.md`` — a markdown table for the job summary.

Usage::

    python tools/coverage/delta_report.py \
        --baseline-areas coverage-kunlunxin-areas-core.json \
        --after-areas coverage-kunlunxin-areas-combined.json \
        --platform kunlunxin \
        --output-json coverage-kunlunxin-delta.json \
        --output-md coverage-kunlunxin-delta.md \
        [--commit-sha <sha>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def delta(baseline: dict, after: dict) -> dict[str, dict]:
    """Compute per-area covered/statements/percent deltas (baseline → after)."""
    base_areas = baseline.get("areas", {})
    after_areas = after.get("areas", {})
    if set(base_areas) != set(after_areas):
        raise SystemExit(
            "[delta] error: area sets differ between baseline and after "
            f"reports ({sorted(set(base_areas) ^ set(after_areas))}); "
            "both must be generated from the same scope file"
        )

    out: dict[str, dict] = {}
    for name in base_areas:
        b = base_areas[name]
        a = after_areas[name]
        out[name] = {
            "baseline_covered_lines": b["covered_lines"],
            "after_covered_lines": a["covered_lines"],
            "delta_covered_lines": a["covered_lines"] - b["covered_lines"],
            "baseline_num_statements": b["num_statements"],
            "after_num_statements": a["num_statements"],
            "delta_num_statements": a["num_statements"] - b["num_statements"],
            "baseline_percent_covered": b["percent_covered"],
            "after_percent_covered": a["percent_covered"],
            "delta_percent_points": round(
                a["percent_covered"] - b["percent_covered"], 2
            ),
        }
    return out


def to_markdown(areas: dict[str, dict], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| area | baseline % | combined % | Δ pp | baseline stmts | after stmts | Δ covered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in sorted(areas.items(), key=lambda kv: -kv[1]["delta_percent_points"]):
        lines.append(
            f"| {name} | {s['baseline_percent_covered']:.2f}% | "
            f"{s['after_percent_covered']:.2f}% | "
            f"{s['delta_percent_points']:+.2f} | "
            f"{s['baseline_num_statements']} | {s['after_num_statements']} | "
            f"{s['delta_covered_lines']:+d} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-areas", required=True, type=Path)
    parser.add_argument("--after-areas", required=True, type=Path)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--commit-sha", default="")
    args = parser.parse_args()

    baseline = _load(args.baseline_areas)
    after = _load(args.after_areas)

    areas = delta(baseline, after)

    base_totals = baseline.get("areas", {}).get("_unassigned")
    after_totals = after.get("areas", {}).get("_unassigned")
    payload = {
        "platform": args.platform,
        "commit_sha": args.commit_sha,
        "baseline_source": str(args.baseline_areas),
        "after_source": str(args.after_areas),
        "areas": areas,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    md = to_markdown(
        areas, f"Supplementary coverage delta — {args.platform} (core → combined)"
    )
    args.output_md.write_text(md, encoding="utf-8")
    print(f"[delta] wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

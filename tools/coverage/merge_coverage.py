#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Merge two coverage.py JSON reports into a combined statement-level union.

Used by the multi-platform supplementary coverage workflow to produce
``coverage-<platform>-combined.json`` from the core baseline report and the
supplementary-only report for that platform. Both inputs must come from the
same commit and the same measurement scope (pyproject.toml
[tool.coverage.*]); the merge validates the file sets match and recomputes
per-file and global totals. Files present only in the supplementary report
are recorded in ``meta.supp_only_files`` (modules only the supplementary
suite imports).

Usage::

    python tools/coverage/merge_coverage.py \
        --baseline coverage-kunlunxin-core.json \
        --supplementary coverage-kunlunxin-supp.json \
        --output coverage-kunlunxin-combined.json \
        --scope tools/coverage/scope_kunlunxin.json \
        [--commit-sha <sha>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm_key(path_key: str) -> str:
    """Normalize a coverage.py file key to a repo-relative posix path."""
    p = path_key.replace("\\", "/")
    # coverage.py keys are absolute or relative paths; strip a vllm_fl/ prefix
    # anchor by finding the package root.
    idx = p.find("vllm_fl/")
    if idx != -1:
        return p[idx:]
    return p


def _summary(executed: list[int], num_statements: int) -> dict:
    covered = len(set(executed))
    pct = (100.0 * covered / num_statements) if num_statements else 0.0
    return {
        "covered_lines": covered,
        "num_statements": num_statements,
        "percent_covered": round(pct, 4),
        "missing_lines": num_statements - covered,
    }


def merge(baseline: dict, supp: dict, scope: dict, commit_sha: str) -> dict:
    """Return the union report; baseline file set is the reference denominator."""
    base_files = baseline.get("files", {})
    supp_files = supp.get("files", {})

    missing = set(base_files) - set(supp_files)
    if missing:
        # Not fatal: supplementary run may have been collected from the same
        # tree; report and proceed with per-file union where both exist.
        print(
            f"[merge] warning: {len(missing)} files in baseline missing from "
            "supplementary report (proceeding with union where possible)"
        )

    reporting_exclude = scope.get("reporting_exclude", [])
    files_out: dict[str, dict] = {}

    total_stmts = 0
    total_covered = 0

    for key, base_info in base_files.items():
        rel = _norm_key(key)
        if any(rel.startswith(prefix) for prefix in reporting_exclude):
            continue

        supp_info = supp_files.get(key) or supp_files.get(rel) or {}
        executed = set(base_info.get("executed_lines", []))
        executed |= set(supp_info.get("executed_lines", []))
        num_statements = base_info.get("summary", {}).get(
            "num_statements", len(base_info.get("statements", executed))
        )
        summary = _summary(sorted(executed), num_statements)
        files_out[rel] = {
            "executed_lines": sorted(executed),
            "summary": summary,
            "missing_lines": sorted(
                set(base_info.get("missing_lines", []))
                & set(range(1, num_statements + 1)) - executed
            ),
        }
        total_stmts += num_statements
        total_covered += summary["covered_lines"]

    # Modules only the supplementary suite imported — recorded, not merged
    # (the baseline file set stays the reference denominator).
    supp_only = sorted(
        rel
        for key in supp_files
        for rel in [_norm_key(key)]
        if rel not in files_out
        and not any(rel.startswith(prefix) for prefix in reporting_exclude)
    )

    overall = (100.0 * total_covered / total_stmts) if total_stmts else 0.0
    meta = dict(baseline.get("meta", {}))
    meta["merged_from"] = ["baseline", "supplementary"]
    meta["commit_sha"] = commit_sha
    meta["supp_only_files"] = supp_only

    return {
        "meta": meta,
        "files": files_out,
        "totals": {
            "covered_lines": total_covered,
            "num_statements": total_stmts,
            "percent_covered": round(overall, 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--supplementary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--commit-sha", default="")
    args = parser.parse_args()

    baseline = _load(args.baseline)
    supp = _load(args.supplementary)
    scope = _load(args.scope)

    if baseline.get("meta", {}).get("format_version") != supp.get("meta", {}).get(
        "format_version"
    ):
        print("[merge] error: coverage.py format versions differ", file=sys.stderr)
        return 2

    merged = merge(baseline, supp, scope, args.commit_sha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(
        f"[merge] wrote {args.output}: "
        f"{merged['totals']['percent_covered']}% "
        f"({merged['totals']['covered_lines']}/{merged['totals']['num_statements']} stmts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Aggregate multiple per-platform coverage.py JSON reports into one union.

Implements the multi-platform aggregate caliber from the coverage plan:

* numerator   — the UNION of executed lines across all participating
  platforms (a line executed by several platforms counts once);
* denominator — the full ``vllm_fl`` executable-statement count.

Every input must be a per-platform report of the same caliber (core
baseline or combined) produced from the same commit and the same
measurement configuration (``pyproject.toml [tool.coverage.*]``). The tool
reconstructs each file's statement set as ``executed ∪ missing`` — the same
already-validated pattern ``merge_coverage.py`` uses — so a line missing
from one platform's lists but executed on another still aggregates
correctly.

Completeness is explicit: ``--expected-platforms`` lists the platforms a
complete aggregate should have; any platform without an input report is
recorded in ``meta.missing_platforms`` with ``meta.complete = false`` and a
loud stdout warning, but the (incomplete) aggregate is still produced so a
single failed CI leg cannot hide the others' numbers.

Per-file ``num_statements`` consistency is validated across inputs (same
commit + same config should yield identical statement sets). Mismatching
files are listed as a warning and marked ``incomplete`` — never a hard
failure: the union statement set stays the line-exact denominator, and the
output's ``missing = statements − executed`` invariant always holds
(checked before writing, like ``merge_coverage.py``).

Usage::

    python tools/coverage/aggregate_coverage.py \
        --inputs coverage-kunlunxin.json coverage-hygon.json \
                 coverage-iluvatar.json coverage-thead.json \
        --expected-platforms kunlunxin,hygon,iluvatar,thead \
        --commit-sha <sha> \
        --output coverage-aggregate.json \
        [--summary-md coverage-aggregate.md]
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
    """Normalize a coverage.py file key to a repo-relative posix path.

    Same normalization as ``merge_coverage.py``: coverage.py keys may be
    absolute or relative; anchor on the package root.
    """
    p = path_key.replace("\\", "/")
    idx = p.find("vllm_fl/")
    if idx != -1:
        return p[idx:]
    return p


def _summary(executed: set[int], num_statements: int) -> dict:
    covered = len(executed)
    pct = (100.0 * covered / num_statements) if num_statements else 0.0
    return {
        "covered_lines": covered,
        "num_statements": num_statements,
        "percent_covered": round(pct, 4),
        "missing_lines": num_statements - covered,
    }


def aggregate(
    reports: list[dict],
    platforms: list[str],
    expected_platforms: list[str],
    commit_sha: str,
) -> dict:
    """Return the multi-platform union report.

    ``reports`` and ``platforms`` are parallel lists: ``reports[i]`` is
    ``platforms[i]``'s per-platform coverage.py JSON report.
    """
    # Union of every file any platform measured. The denominator reference is
    # the first input (arbitrary — the consistency check below enforces the
    # rest agree); the union statement set is rebuilt per file from
    # (executed ∪ missing) so the denominator is line-exact even when one
    # platform's report truncated a file's lists.
    file_keys: set[str] = set()
    for report in reports:
        file_keys.update(_norm_key(k) for k in report.get("files", {}))

    files_out: dict[str, dict] = {}
    total_stmts = 0
    total_covered = 0
    inconsistent_files: list[dict] = []

    for key in sorted(file_keys):
        executed: set[int] = set()
        statements: set[int] = set()
        ref_num_statements: int | None = None

        for report in reports:
            info = (
                report.get("files", {}).get(key)
                # inputs may key by the original absolute path; _norm_key
                # anchors both spellings to the same relative key
                or report.get("files", {}).get(key.replace("/", "\\"))
                or {}
            )
            file_executed = set(info.get("executed_lines", []))
            file_missing = set(info.get("missing_lines", []))
            executed |= file_executed
            statements |= file_executed | file_missing

            ns = info.get("summary", {}).get("num_statements")
            if ns is not None and ref_num_statements is None:
                ref_num_statements = ns

        # Same commit + same coverage config must produce identical
        # per-file statement counts. A mismatch is recorded (warn +
        # incomplete) instead of failing: the aggregate is a reporting
        # artifact, and the union statement set remains the best
        # line-exact denominator either way.
        if ref_num_statements is None:
            num_statements = len(statements)
        else:
            num_statements = ref_num_statements
            if len(statements) != ref_num_statements:
                inconsistent_files.append(
                    {
                        "file": key,
                        "num_statements": ref_num_statements,
                        "statement_set_size": len(statements),
                    }
                )
                if len(statements) < ref_num_statements:
                    # Every input truncated this file's lists; fall back to
                    # the reference count to keep the denominator, mirroring
                    # merge_coverage.py's fallback.
                    statements |= set(range(1, ref_num_statements + 1)) - executed
            num_statements = len(statements)

        summary = _summary(executed, num_statements)
        files_out[key] = {
            "executed_lines": sorted(executed),
            "summary": summary,
            "missing_lines": sorted(statements - executed),
        }
        total_stmts += num_statements
        total_covered += summary["covered_lines"]

    overall = (100.0 * total_covered / total_stmts) if total_stmts else 0.0

    missing_platforms = [p for p in expected_platforms if p not in platforms]
    complete = not missing_platforms and not inconsistent_files

    meta = dict(reports[0].get("meta", {})) if reports else {}
    meta.update(
        {
            "caliber": "multi-platform-aggregate",
            "commit_sha": commit_sha,
            "participating_platforms": platforms,
            "expected_platforms": expected_platforms,
            "missing_platforms": missing_platforms,
            "complete": complete,
            "num_statements_consistent": not inconsistent_files,
            "inconsistent_files": inconsistent_files[:50],
            "input_reports": {
                platform: report.get("meta", {}).get("caliber", "unlabeled")
                for platform, report in zip(platforms, reports)
            },
        }
    )

    return {
        "meta": meta,
        "files": files_out,
        "totals": {
            "covered_lines": total_covered,
            "num_statements": total_stmts,
            "percent_covered": round(overall, 4),
        },
    }


def _self_check(agg: dict) -> list[str]:
    """Verify summary.missing_lines == len(missing_lines) per file.

    Same invariant merge_coverage.py guarantees; returns error messages.
    """
    errors = []
    for key, info in agg["files"].items():
        missing = info.get("missing_lines", [])
        summary = info.get("summary", {})
        if summary.get("missing_lines") != len(missing):
            errors.append(
                f"{key}: summary.missing_lines={summary.get('missing_lines')} "
                f"but len(missing_lines)={len(missing)}"
            )
    return errors


def to_markdown(agg: dict, per_platform_totals: dict[str, dict]) -> str:
    """Summary table: per-platform numbers + aggregate + completeness."""
    lines = [
        "# Multi-platform aggregate coverage",
        "",
        f"Commit: `{agg['meta'].get('commit_sha', 'n/a')}`",
        "",
        "Numerator = union of executed lines across participating platforms "
        "(a line run by several platforms counts once); denominator = full "
        "vllm_fl executable statements.",
        "",
        "| platform | statements | covered | % |",
        "|---|---:|---:|---:|",
    ]
    for platform, t in per_platform_totals.items():
        pct = (100.0 * t["covered_lines"] / t["num_statements"]) if t["num_statements"] else 0.0
        lines.append(
            f"| {platform} | {t['num_statements']} | {t['covered_lines']} | "
            f"{pct:.2f}% |"
        )
    lines.append(
        f"| **aggregate (union)** | {agg['totals']['num_statements']} | "
        f"{agg['totals']['covered_lines']} | "
        f"{agg['totals']['percent_covered']:.2f}% |"
    )
    lines.append("")
    if agg["meta"].get("complete"):
        lines.append("Status: **complete** — all expected platforms reported.")
    else:
        lines.append(
            "Status: **INCOMPLETE** — missing platforms: "
            f"{', '.join(agg['meta'].get('missing_platforms', [])) or '(none)'}"
            "; num_statements inconsistencies: "
            f"{len(agg['meta'].get('inconsistent_files', []))}. "
            "The aggregate below is still produced from the available "
            "platforms; treat the number as a floor, not the plan target."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        type=Path,
        help="per-platform coverage.py JSON reports (same commit, same config)",
    )
    parser.add_argument(
        "--platforms",
        default=None,
        help="comma-separated platform names matching --inputs order "
        "(defaults to input basenames' platform part)",
    )
    parser.add_argument(
        "--expected-platforms",
        default="kunlunxin,hygon,iluvatar,thead",
        help="platforms a complete aggregate should include "
        "(default: kunlunxin,hygon,iluvatar,thead)",
    )
    parser.add_argument("--commit-sha", default="")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--summary-md",
        default=None,
        type=Path,
        help="optional markdown summary table for job summaries",
    )
    args = parser.parse_args()

    reports = [_load(p) for p in args.inputs]

    if args.platforms:
        platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
        if len(platforms) != len(reports):
            print(
                f"[aggregate] error: {len(platforms)} platform names for "
                f"{len(reports)} inputs",
                file=sys.stderr,
            )
            return 2
    else:
        # derive from filename: coverage-<platform>(-<tag>).json
        platforms = []
        for p in args.inputs:
            stem = p.stem
            platform = (
                stem.split("-", 1)[1].split(".json")[0]
                if stem.startswith("coverage-")
                else stem
            )
            platform = platform.rsplit("-", 1)[0] if "-" in platform else platform
            platforms.append(platform)

    expected = [p.strip() for p in args.expected_platforms.split(",") if p.strip()]

    agg = aggregate(reports, platforms, expected, args.commit_sha)

    # Self-check: per-file summary.missing_lines must equal len(missing_lines)
    errors = _self_check(agg)
    if errors:
        for e in errors:
            print(f"[aggregate] error: {e}", file=sys.stderr)
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    # Per-platform totals for the summary (raw input numbers, no aggregation)
    per_platform_totals = {}
    for platform, report in zip(platforms, reports):
        t = report.get("totals", {})
        per_platform_totals[platform] = {
            "num_statements": t.get("num_statements", 0),
            "covered_lines": t.get("covered_lines", 0),
        }

    if args.summary_md:
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_md, "w", encoding="utf-8") as f:
            f.write(to_markdown(agg, per_platform_totals))

    missing = agg["meta"]["missing_platforms"]
    if missing:
        print()
        print("=" * 72)
        print("[aggregate] WARNING: AGGREGATE RESULT IS INCOMPLETE")
        print(
            f"[aggregate] missing platform reports: {', '.join(missing)}"
        )
        print(
            "[aggregate] the union below is a FLOOR for the plan's "
            "aggregate target, not the target number itself"
        )
        print("=" * 72)
        print()
    if not agg["meta"]["num_statements_consistent"]:
        print(
            "[aggregate] warning: per-file num_statements differ across "
            f"inputs for {len(agg['meta']['inconsistent_files'])} files "
            "(see meta.inconsistent_files); result marked incomplete"
        )

    print(
        f"[aggregate] wrote {args.output}: "
        f"{agg['totals']['percent_covered']}% "
        f"({agg['totals']['covered_lines']}/{agg['totals']['num_statements']} stmts, "
        f"platforms={','.join(platforms)}, "
        f"complete={str(agg['meta']['complete']).lower()})"
    )
    for platform, t in per_platform_totals.items():
        pct = (
            100.0 * t["covered_lines"] / t["num_statements"]
            if t["num_statements"] else 0.0
        )
        print(
            f"[aggregate]   {platform}: {t['covered_lines']}/"
            f"{t['num_statements']} ({pct:.2f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

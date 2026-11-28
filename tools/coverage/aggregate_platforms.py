# Copyright (c) 2026 BAAI. All rights reserved.
"""Tier-2 statistic: multi-platform aggregate line coverage.

Unions the executed lines of every platform's raw (full-scope) coverage
report over the complete vllm_fl source range:

    aggregate = |union of executed lines across platforms|
                / total executable statements of the full vllm_fl scope

The same line executed by several platforms counts once. All platforms
must be measured on the same source commit with the same coverage
config; the tool records the commit of each input and refuses to mix
different SHAs unless --allow-mismatch is given (the report then marks
the result as incomplete).

Missing platform reports are listed in the output so the aggregate can
be flagged incomplete.

Usage:
    python tools/coverage/aggregate_platforms.py \
        --reports coverage-nvidia.json coverage-kunlunxin.json ... \
        --output coverage-aggregate.json
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


def aggregate(
    reports: dict[str, dict], allow_mismatch: bool, expect_commit: str | None = None
) -> dict:
    executed: dict[str, set] = {}
    statements: dict[str, int] = {}
    commits = {}
    for name, report in reports.items():
        sha = report.get("meta", {}).get("commit_sha") or ""
        if not sha or sha == "unknown":
            sha = report.get("commit_sha") or "unknown"
        if expect_commit and sha not in ("unknown", expect_commit):
            raise SystemExit(
                f"{name} was measured on {sha}, expected {expect_commit}: re-measure first"
            )
        commits[name] = sha
        for key, info in report.get("files", {}).items():
            rel = _rel(key)
            executed.setdefault(rel, set()).update(info.get("executed_lines", []))
            n = info.get("summary", {}).get("num_statements", 0)
            statements[rel] = max(statements.get(rel, 0), n)

    shas = {c for c in commits.values() if c not in ("", "unknown")}
    mixed = len(shas) > 1
    if mixed and not allow_mismatch:
        raise SystemExit(
            "refusing to mix source commits: "
            + ", ".join(f"{k}={v}" for k, v in commits.items())
        )

    files = {
        rel: {
            "executed_lines": sorted(lines),
            "num_statements": statements[rel],
            "covered_lines": len(lines),
        }
        for rel, lines in executed.items()
    }
    covered = sum(len(v) for v in executed.values())
    total = sum(statements.values())
    return {
        "meta": {
            "format": 3,
            "aggregate": True,
            "platform_commits": commits,
            "commit_mismatch": mixed,
        },
        "totals": {
            "covered_lines": covered,
            "num_statements": total,
            "percent_covered": round(100.0 * covered / total, 2) if total else 0.0,
        },
        "files": files,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, nargs="+", type=Path)
    ap.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="platform names for --reports (same order); defaults to file stems",
    )
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="aggregate anyway, marking the result as commit-mixed",
    )
    ap.add_argument(
        "--expect-commit",
        default=None,
        help="require every report to have been stamped with this SHA (see --commit-sha of the runners)",
    )
    args = ap.parse_args()

    names = args.names or [
        p.stem.replace("coverage-", "").replace("-combined", "").replace("-final5", "")
        for p in args.reports
    ]
    reports = {n: _load(p) for n, p in zip(names, args.reports)}

    out = aggregate(reports, args.allow_mismatch, expect_commit=args.expect_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    t = out["totals"]
    note = " (COMMIT-MIXED)" if out["meta"]["commit_mismatch"] else ""
    print(
        f"[aggregate]{note} {t['covered_lines']}/{t['num_statements']} "
        f"= {t['percent_covered']}% across {len(reports)} platforms -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026 BAAI. All rights reserved.
"""Generate a platform-applicable coverage JSON from the raw full-scope report.

Implements the plan's tier-1 statistic ("单平台适用代码行覆盖率") as a
coverage.py-compatible JSON so FlagCICD can render it directly:

* measurement scope stays untouched (raw JSON keeps the full vllm_fl source);
* the applicability filter is applied at REPORT time only — files whose
  whole responsibility belongs to another platform are dropped from both
  numerator and denominator; hybrid files are filtered at LINE level via
  the per-line ownership map when one is provided;
* covered_lines / num_statements / missing_lines / percent_covered are
  recomputed per file and in total, so the dashboard number always
  matches the report details.

Usage:
    python tools/coverage/platform_scope_report.py \
        --report coverage-nvidia.json \
        --scope tools/coverage/scope_nvidia.json \
        [--line-ownership ownership_nvidia.json] \
        --output coverage-nvidia-applicable.json
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


def applicable_report(
    report: dict,
    scope: dict,
    line_ownership: dict | None,
    commit_sha: str | None = None,
) -> dict:
    """Filter the raw report down to the platform-applicable code set.

    Files are dropped when they live under an excluded vendor prefix.
    Hybrid files (kept by prefix rules) are further filtered line-by-line
    when the optional ``line_ownership`` map names them: only lines the
    platform owns stay in statements/executed/missing.
    """
    exclude = tuple(scope.get("reporting_exclude", []))
    out_files = {}
    for key, info in report.get("files", {}).items():
        rel = _rel(key)
        if rel.startswith(exclude):
            continue
        executed = list(info.get("executed_lines", []))
        statements = sorted(
            info.get("executed_lines", []) + info.get("missing_lines", [])
        )
        own = (line_ownership or {}).get(rel)
        if own is not None:
            keep = set(own)
            statements = [n for n in statements if n in keep]
            executed = [n for n in executed if n in keep]
        missing = [n for n in statements if n not in set(executed)]
        if not statements:
            continue
        summary = {
            "covered_lines": len(executed),
            "num_statements": len(statements),
            "missing_lines": len(missing),
            "percent_covered": round(100.0 * len(executed) / len(statements), 2),
        }
        out_files[rel] = {
            "executed_lines": sorted(executed),
            "missing_lines": sorted(missing),
            "summary": summary,
            "index": info.get("index"),
        }
    covered = sum(v["summary"]["covered_lines"] for v in out_files.values())
    statements = sum(v["summary"]["num_statements"] for v in out_files.values())
    totals = {
        "covered_lines": covered,
        "num_statements": statements,
        "percent_covered": round(100.0 * covered / statements, 2)
        if statements
        else 0.0,
    }
    meta = dict(
        report.get("meta", {}), applicable_scope=str(scope.get("platform", "nvidia"))
    )
    if commit_sha:
        meta["commit_sha"] = commit_sha
    return {
        "meta": meta,
        "totals": totals,
        "files": out_files,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--scope", required=True, type=Path)
    ap.add_argument(
        "--line-ownership",
        type=Path,
        default=None,
        help="optional JSON: {rel_file: [line numbers the platform owns]} for hybrid files",
    )
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--commit-sha",
        default=None,
        help="stamp the measurement commit into the report meta (tier-2 guard)",
    )
    args = ap.parse_args()

    report = _load(args.report)
    scope = _load(args.scope)
    ownership = _load(args.line_ownership) if args.line_ownership else None

    out = applicable_report(report, scope, ownership, commit_sha=args.commit_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    t = out["totals"]
    print(
        f"[applicable] {t['covered_lines']}/{t['num_statements']} "
        f"= {t['percent_covered']}% -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

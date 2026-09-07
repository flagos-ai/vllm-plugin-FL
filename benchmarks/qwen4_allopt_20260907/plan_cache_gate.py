#!/usr/bin/env python3
"""Fail-closed gate for real worker post-warmup plan-cache evidence.

It intentionally accepts only worker log records from the server log. A
preflight import or a successful legacy ``install()`` call is not evidence of
runtime hits. The output contains the server-log hash and line numbers, never
the raw log text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


STATS_RE = re.compile(
    r"FlagGems ATen plan cache stats \(post-warmup\): "
    r"enabled=(?P<enabled>\w+) installed=(?P<installed>\w+) "
    r"hits=(?P<hits>-?\d+) misses=(?P<misses>-?\d+) "
    r"hit_rate=(?P<hit_rate>[0-9.]+) size=(?P<size>-?\d+) "
    r"evictions=(?P<evictions>-?\d+) bypasses=(?P<bypasses>-?\d+)"
)
RANK_RE = re.compile(r"\(Worker_TP(?P<rank>\d+)\b")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse(log_path: Path, expected_ranks: int) -> dict[str, Any]:
    records: dict[int, dict[str, Any]] = {}
    matched_lines: list[int] = []
    errors: list[str] = []
    if not log_path.is_file():
        errors.append("server log is missing")
    else:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_no, line in enumerate(stream, 1):
                match = STATS_RE.search(line)
                if not match:
                    continue
                rank_match = RANK_RE.search(line)
                if rank_match is None:
                    continue
                rank = int(rank_match.group("rank"))
                matched_lines.append(line_no)
                values: dict[str, Any] = {
                    "rank": rank,
                    "line": line_no,
                    "enabled": match.group("enabled").lower() == "true",
                    "installed": match.group("installed").lower() == "true",
                    "hits": int(match.group("hits")),
                    "misses": int(match.group("misses")),
                    "hit_rate": float(match.group("hit_rate")),
                    "size": int(match.group("size")),
                    "evictions": int(match.group("evictions")),
                    "bypasses": int(match.group("bypasses")),
                }
                # Keep the latest lifecycle record per rank.
                records[rank] = values

    expected = set(range(expected_ranks))
    seen = set(records)
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing:
        errors.append(f"missing worker post-warmup ranks: {missing}")
    if unexpected:
        errors.append(f"unexpected worker ranks: {unexpected}")
    for rank in sorted(expected & seen):
        row = records[rank]
        if not row["enabled"] or not row["installed"]:
            errors.append(f"rank {rank}: enabled/installed is not true")
        if row["hits"] <= 0 or row["misses"] <= 0:
            errors.append(f"rank {rank}: hits/misses must both be > 0")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "expected_ranks": expected_ranks,
        "records": [records[rank] for rank in sorted(records)],
        "matched_line_count": len(matched_lines),
        "server_log_sha256": sha256(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-ranks", type=int, default=8)
    args = parser.parse_args()
    result = parse(args.log, args.expected_ranks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "pass":
        print("plan-cache gate: FAIL")
        return 1
    print("plan-cache gate: PASS " f"ranks={args.expected_ranks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

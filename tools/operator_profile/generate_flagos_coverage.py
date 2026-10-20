#!/usr/bin/env python3
"""Compare native and plugin inventories and emit FlagOS coverage by API."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rule.rule_coverage import classify_coverage, read_flagos_evidence

COVERAGE_FIELDNAMES = [
    "operator_id",
    "operator_name",
    "operator_kind",
    "flagos_covered",
    "flagos_type",
    "kernel_name",
    "plugin_operator_id",
    "evidence",
]
SUMMARY_FIELDNAMES = [
    "covered_operator_count",
    "undetermined_operator_count",
    "total_operator_count",
    "coverage_percent(%)",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def group_by_operator_id(
    rows: list[dict[str, str]],
) -> dict[int, dict[str, set[str]]]:
    groups: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {
            "operator_names": set(),
            "operator_kinds": set(),
            "kernel_names": set(),
        }
    )
    for row in rows:
        operator_id = int(row["operator_id"])
        groups[operator_id]["operator_names"].add(row["operator_name"])
        groups[operator_id]["operator_kinds"].add(row["operator_kind"])
        groups[operator_id]["kernel_names"].add(row["kernel_name"])
    return dict(groups)


def encoded(values: set[str]) -> str:
    return json.dumps(sorted(values), ensure_ascii=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--flaggems-oplist", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline = group_by_operator_id(read_rows(args.baseline))
    plugin = group_by_operator_id(read_rows(args.plugin))
    evidence = read_flagos_evidence(args.flaggems_oplist)

    plugin_by_name: dict[str, set[int]] = defaultdict(set)
    for operator_id, values in plugin.items():
        for operator_name in values["operator_names"]:
            if operator_name.strip().lower() not in {"", "null"}:
                plugin_by_name[operator_name].add(operator_id)

    rows: list[dict[str, Any]] = []
    for operator_id in sorted(baseline):
        values = baseline[operator_id]
        matching_plugin_ids = set()
        for operator_name in values["operator_names"]:
            if operator_name.strip().lower() not in {"", "null"}:
                matching_plugin_ids.update(plugin_by_name.get(operator_name, set()))
        decision = classify_coverage(
            values["operator_names"],
            values["operator_kinds"],
            values["kernel_names"],
            evidence,
        )
        rows.append(
            {
                "operator_id": operator_id,
                "operator_name": encoded(values["operator_names"]),
                "operator_kind": encoded(values["operator_kinds"]),
                "flagos_covered": (
                    "true"
                    if decision.covered is True
                    else "false"
                    if decision.covered is False
                    else ""
                ),
                "flagos_type": decision.flagos_type,
                "kernel_name": encoded(values["kernel_names"]),
                "plugin_operator_id": json.dumps(sorted(matching_plugin_ids)),
                "evidence": decision.evidence,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=COVERAGE_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    numerator = sum(row["flagos_covered"] == "true" for row in rows)
    undetermined = sum(row["flagos_covered"] == "" for row in rows)
    denominator = len(rows)
    percent = numerator / denominator * 100 if denominator else 0.0
    summary_output = args.output.parent / "operator_flagos_coverage_summary.csv"
    with summary_output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=SUMMARY_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "covered_operator_count": numerator,
                "undetermined_operator_count": undetermined,
                "total_operator_count": denominator,
                "coverage_percent(%)": f"{percent:.2f}",
            }
        )
    print(
        json.dumps(
            {
                "numerator": numerator,
                "undetermined": undetermined,
                "denominator": denominator,
                "coverage_percent": round(percent, 2),
                "output": str(args.output),
                "summary_output": str(summary_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

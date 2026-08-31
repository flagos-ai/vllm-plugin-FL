#!/usr/bin/env python3
"""Extract a rank-scoped runtime GPU event inventory from PyTorch traces."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
MetadataKey = tuple[str, str | None, str | None, str]


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        prefix = source.read(4096)
        source.seek(0)
        if '"traceEvents"' not in prefix:
            yield from json.load(source).get("traceEvents", [])
            return
        for line in source:
            if '"traceEvents"' in line:
                break
        else:
            raise ValueError(f"traceEvents not found: {path}")
        event_lines: list[str] = []
        for line in source:
            if not event_lines:
                if line.startswith("  {"):
                    event_lines.append(line)
                elif line.lstrip().startswith("]"):
                    return
                continue
            event_lines.append(line)
            if line.startswith("  }"):
                encoded = "".join(event_lines).rstrip().removesuffix(",")
                yield json.loads(encoded)
                event_lines.clear()


def trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.pt.trace.json.gz"))
    files.extend(sorted(path.rglob("*.pt.trace.json")))
    return files


def runtime_trace_files(path: Path) -> list[Path]:
    return [
        file for file in trace_files(path) if not file.name.startswith("graph_capture_")
    ]


def rank_in_filename(file: Path) -> int:
    match = re.search(r"(?:^|_)rank_?(\d+)(?:[._]|$)", file.name)
    return int(match.group(1)) if match else -1


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def decode(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def metadata(event: dict[str, Any]) -> MetadataKey:
    args = event.get("args", {})
    has_shapes = "Input Dims" in args
    has_dtypes = "Input type" in args
    shapes = canonical(args["Input Dims"]) if has_shapes else None
    dtypes = canonical(args["Input type"]) if has_dtypes else None
    status = (
        "shape_and_dtype"
        if has_shapes and has_dtypes
        else "shape_only"
        if has_shapes
        else "dtype_only"
        if has_dtypes
        else "no_input_metadata"
    )
    return str(event.get("name", "")), shapes, dtypes, status


def external_id(event: dict[str, Any]) -> int | str | None:
    value = event.get("args", {}).get("External id")
    return value if isinstance(value, (int, str)) else None


def mapping(
    event_external_id: int | str | None,
    cpu_by_external_id: dict[int | str, set[MetadataKey]],
) -> dict[str, Any]:
    if event_external_id is None:
        return {
            "mapping_status": "missing_external_id",
            "operator": None,
            "input_shapes": None,
            "input_dtypes": None,
        }
    candidates = sorted(
        cpu_by_external_id.get(event_external_id, set()),
        key=lambda item: (
            item[0],
            item[1] or "",
            item[2] or "",
            item[3],
        ),
    )
    if not candidates:
        return {
            "mapping_status": "no_cpu_op_match",
            "operator": None,
            "input_shapes": None,
            "input_dtypes": None,
        }
    if len(candidates) > 1:
        candidate_rows = [
            {
                "operator": item[0],
                "input_shapes": decode(item[1]),
                "input_dtypes": decode(item[2]),
                "metadata_status": item[3],
            }
            for item in candidates
        ]
        names = {item[0] for item in candidates}
        return {
            "mapping_status": (
                "shape_ambiguous" if len(names) == 1 else "operator_ambiguous"
            ),
            "operator": next(iter(names)) if len(names) == 1 else None,
            "input_shapes": None,
            "input_dtypes": None,
            "candidate_operators": candidate_rows,
        }

    operator, shapes, dtypes, metadata_status = candidates[0]
    status_by_metadata = {
        "shape_and_dtype": "operator_shape_matched",
        "shape_only": "operator_matched_dtype_missing",
        "dtype_only": "operator_matched_shape_missing",
        "no_input_metadata": "operator_matched_metadata_missing",
    }
    return {
        "mapping_status": status_by_metadata[metadata_status],
        "operator": operator,
        "input_shapes": decode(shapes),
        "input_dtypes": decode(dtypes),
    }


def event_details(
    event: dict[str, Any],
    event_external_id: int | str | None,
    link: dict[str, Any],
) -> dict[str, Any]:
    args = event.get("args", {})
    details = {
        "category": str(event.get("cat", "")),
        "duration_us": float(event.get("dur", 0.0)),
        "timestamp_us": event.get("ts"),
        "process_id": event.get("pid"),
        "thread_id": event.get("tid"),
        "device": args.get("device"),
        "stream": args.get("stream"),
        "external_id": event_external_id,
    }
    details.update(link)
    return details


def collect_runtime(
    files: list[Path], rank: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    operator_variants: dict[str, Counter[tuple[str | None, str | None, str]]] = (
        defaultdict(Counter)
    )
    operator_cpu_us: Counter[str] = Counter()
    cpu_event_count = 0
    cpu_metadata_count: Counter[str] = Counter()

    kernel_groups: dict[str, dict[str, Any]] = {}
    non_kernel_groups: dict[str, dict[str, Any]] = {}
    kernel_mapping_count: Counter[str] = Counter()
    kernel_mapping_us: Counter[str] = Counter()
    category_count: Counter[str] = Counter()
    category_us: Counter[str] = Counter()
    operator_kernel_names: dict[str, set[str]] = defaultdict(set)
    operator_kernel_event_count: Counter[str] = Counter()

    for trace_index, trace in enumerate(files):
        cpu_by_external_id: dict[int | str, set[MetadataKey]] = defaultdict(set)
        for event in iter_events(trace):
            if event.get("cat") != "cpu_op":
                continue
            cpu_event_count += 1
            key = metadata(event)
            operator, shapes, dtypes, metadata_status = key
            operator_variants[operator][(shapes, dtypes, metadata_status)] += 1
            operator_cpu_us[operator] += float(event.get("dur", 0.0))
            cpu_metadata_count[metadata_status] += 1
            event_external_id = external_id(event)
            if event_external_id is not None:
                cpu_by_external_id[event_external_id].add(key)

        gpu_event_index = 0
        for event in iter_events(trace):
            category = str(event.get("cat", ""))
            if category not in GPU_CATEGORIES:
                continue
            name = str(event.get("name", ""))
            duration_us = float(event.get("dur", 0.0))
            event_external_id = external_id(event)
            link = mapping(event_external_id, cpu_by_external_id)
            event_id = f"rank{rank}:trace_{trace_index:03d}:gpu_{gpu_event_index:09d}"
            gpu_event_index += 1
            row = event_details(event, event_external_id, link)
            row["trace_file"] = trace.name

            category_count[category] += 1
            category_us[category] += duration_us
            target = kernel_groups if category == "kernel" else non_kernel_groups
            group_key = name if category == "kernel" else f"{category}:{name}"
            group = target.setdefault(
                group_key,
                {
                    "summary": {"total_call_count": 0, "total_time_us": 0.0},
                    "events": {},
                },
            )
            group["summary"]["total_call_count"] += 1
            group["summary"]["total_time_us"] += duration_us
            group["events"][event_id] = row

            if category != "kernel":
                continue
            status = link["mapping_status"]
            kernel_mapping_count[status] += 1
            kernel_mapping_us[status] += duration_us
            operator = link["operator"]
            if operator is not None:
                operator_kernel_names[operator].add(name)
                operator_kernel_event_count[operator] += 1

    kernel_total_us = category_us["kernel"]
    ordered_kernel_groups: dict[str, Any] = {}
    for name, group in sorted(
        kernel_groups.items(),
        key=lambda item: (-item[1]["summary"]["total_time_us"], item[0]),
    ):
        total_us = group["summary"]["total_time_us"]
        group["summary"]["total_time_us"] = round(total_us, 3)
        group["summary"]["profiling_time_ratio"] = (
            total_us / kernel_total_us if kernel_total_us else 0.0
        )
        group["summary"]["profiling_time_pct"] = (
            total_us / kernel_total_us * 100 if kernel_total_us else 0.0
        )
        ordered_kernel_groups[name] = group

    ordered_non_kernel_groups: dict[str, Any] = {}
    for name, group in sorted(
        non_kernel_groups.items(),
        key=lambda item: (-item[1]["summary"]["total_time_us"], item[0]),
    ):
        group["summary"]["total_time_us"] = round(group["summary"]["total_time_us"], 3)
        ordered_non_kernel_groups[name] = group

    operator_index: dict[str, Any] = {}
    for operator in sorted(operator_variants):
        variants = []
        for (shapes, dtypes, status), count in sorted(
            operator_variants[operator].items(),
            key=lambda item: (-item[1], item[0][2], item[0][0] or "", item[0][1] or ""),
        ):
            variants.append(
                {
                    "input_shapes": decode(shapes),
                    "input_dtypes": decode(dtypes),
                    "metadata_status": status,
                    "runtime_call_count": count,
                }
            )
        operator_index[operator] = {
            "summary": {
                "runtime_call_count": sum(operator_variants[operator].values()),
                "runtime_cpu_duration_total_us": round(operator_cpu_us[operator], 3),
                "metadata_variant_count": len(variants),
                "mapped_kernel_event_count": operator_kernel_event_count[operator],
            },
            "metadata_variants": variants,
            "kernel_names": sorted(operator_kernel_names[operator]),
        }

    kernel_inventory_count = sum(
        len(group["events"]) for group in ordered_kernel_groups.values()
    )
    kernel_inventory_us = sum(
        sum(event["duration_us"] for event in group["events"].values())
        for group in ordered_kernel_groups.values()
    )
    summary = {
        "scope": {
            "rank": rank,
            "phase": "runtime_only",
            "graph_capture_included": False,
            "timing_denominator": "sum_of_rank0_runtime_kernel_durations",
        },
        "runtime_trace_files": [str(file) for file in files],
        "cpu_operator_event_count": cpu_event_count,
        "unique_cpu_operator_names": len(operator_variants),
        "cpu_operator_event_count_by_metadata_status": dict(
            sorted(cpu_metadata_count.items())
        ),
        "gpu_event_count": sum(category_count.values()),
        "gpu_event_count_by_category": dict(sorted(category_count.items())),
        "gpu_activity_us_by_category": {
            key: round(value, 3) for key, value in sorted(category_us.items())
        },
        "kernel_event_count": category_count["kernel"],
        "unique_kernel_names": len(kernel_groups),
        "kernel_time_total_us": round(kernel_total_us, 3),
        "kernel_mapping_event_count_by_status": dict(
            sorted(kernel_mapping_count.items())
        ),
        "kernel_mapping_time_us_by_status": {
            key: round(value, 3) for key, value in sorted(kernel_mapping_us.items())
        },
        "conservation": {
            "kernel_event_count_in_trace": category_count["kernel"],
            "kernel_event_count_in_inventory": kernel_inventory_count,
            "kernel_event_count_matches": (
                category_count["kernel"]
                == kernel_inventory_count
                == sum(kernel_mapping_count.values())
            ),
            "kernel_time_us_in_trace": round(kernel_total_us, 3),
            "kernel_time_us_in_inventory": round(kernel_inventory_us, 3),
            "kernel_time_matches": abs(kernel_total_us - kernel_inventory_us) < 0.001,
            "kernel_mapping_time_matches": (
                abs(kernel_total_us - sum(kernel_mapping_us.values())) < 0.001
            ),
        },
    }
    return ordered_kernel_groups, ordered_non_kernel_groups, operator_index, summary


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime_files = [
        file
        for file in runtime_trace_files(args.runtime)
        if rank_in_filename(file) == args.rank
    ]
    if not runtime_files:
        raise FileNotFoundError(
            f"no rank-{args.rank} runtime profiler traces under {args.runtime}"
        )

    kernel_summary, non_kernel_summary, operator_index, summary = collect_runtime(
        runtime_files, args.rank
    )
    write_json(args.output_dir / "kernel_summary.json", kernel_summary)
    write_json(args.output_dir / "non_kernel_gpu_activity.json", non_kernel_summary)
    write_json(args.output_dir / "operator_index.json", operator_index)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

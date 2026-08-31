#!/usr/bin/env python3
"""Extract rank-scoped operator metadata and runtime GPU timing."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def capture_trace_files(path: Path | None) -> list[Path]:
    return [] if path is None else trace_files(path)


def runtime_trace_files(path: Path) -> list[Path]:
    return [
        file for file in trace_files(path) if not file.name.startswith("graph_capture_")
    ]


def rank_in_filename(file: Path) -> int:
    match = re.search(r"(?:^|_)rank_?(\d+)(?:[._]|$)", file.name)
    return int(match.group(1)) if match else -1


def metadata(event: dict[str, Any]) -> tuple[str, str, str, str]:
    args = event.get("args", {})
    has_shapes = "Input Dims" in args
    has_dtypes = "Input type" in args
    shapes = compact(args.get("Input Dims", []))
    dtypes = compact(args.get("Input type", []))
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


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def collect(files: list[Path]) -> tuple[Counter, Counter, dict[str, Any]]:
    aggregate: Counter[tuple[str, str, str, str]] = Counter()
    by_rank: Counter[tuple[int, str, str, str, str]] = Counter()
    total = with_shapes = with_dtypes = 0
    for trace in files:
        rank = rank_in_filename(trace)
        for event in iter_events(trace):
            if event.get("cat") != "cpu_op":
                continue
            total += 1
            key = metadata(event)
            with_shapes += int(key[3] in {"shape_and_dtype", "shape_only"})
            with_dtypes += int(key[3] in {"shape_and_dtype", "dtype_only"})
            aggregate[key] += 1
            by_rank[(rank, *key)] += 1
    summary = {
        "trace_files": len(files),
        "ranks": sorted({rank_in_filename(file) for file in files}),
        "cpu_operator_events": total,
        "cpu_operator_events_with_shapes": with_shapes,
        "cpu_operator_events_with_dtypes": with_dtypes,
        "operator_rows": len(aggregate),
        "effective_operator_rows": sum(
            key[3] == "shape_and_dtype" for key in aggregate
        ),
        "unique_effective_operator_names": len(
            {key[0] for key in aggregate if key[3] == "shape_and_dtype"}
        ),
        "no_input_metadata_events": sum(
            count for key, count in aggregate.items() if key[3] == "no_input_metadata"
        ),
        "no_input_metadata_operator_names": sorted(
            {key[0] for key in aggregate if key[3] == "no_input_metadata"}
        ),
        "csv_event_output_coverage_pct": (
            sum(aggregate.values()) / total * 100 if total else 0.0
        ),
    }
    return aggregate, by_rank, summary


def collect_runtime_timing(
    files: list[Path],
) -> tuple[Counter, Counter, Counter, dict[str, Any]]:
    cpu_us: Counter[tuple[str, str, str]] = Counter()
    kernel_count: Counter[tuple[str, str, str]] = Counter()
    kernel_us: Counter[tuple[str, str, str]] = Counter()
    gpu_rows: Counter[tuple[str, str, str, str, str, str]] = Counter()
    gpu_us_by_row: Counter[tuple[str, str, str, str, str, str]] = Counter()
    category_count: Counter[str] = Counter()
    category_us: Counter[str] = Counter()

    for trace in files:
        cpu_by_external_id: dict[int | str, tuple[str, str, str, str]] = {}
        for event in iter_events(trace):
            if event.get("cat") != "cpu_op":
                continue
            key = metadata(event)
            if key[3] == "shape_and_dtype":
                cpu_us[key[:3]] += float(event.get("dur", 0.0))
            event_external_id = external_id(event)
            if event_external_id is not None:
                cpu_by_external_id[event_external_id] = key

        for event in iter_events(trace):
            category = str(event.get("cat", ""))
            if category not in GPU_CATEGORIES:
                continue
            duration_us = float(event.get("dur", 0.0))
            category_count[category] += 1
            category_us[category] += duration_us
            event_external_id = external_id(event)
            linked = (
                cpu_by_external_id.get(event_external_id)
                if event_external_id is not None
                else None
            )
            if event_external_id is None:
                status = "missing_external_id"
                operator, shapes, dtypes = "", "[]", "[]"
            elif linked is None:
                status = "no_cpu_op_match"
                operator, shapes, dtypes = "", "[]", "[]"
            elif linked[3] != "shape_and_dtype":
                status = "cpu_op_missing_shape_or_dtype"
                operator, shapes, dtypes = linked[:3]
            else:
                status = "attributed_shape_dtype"
                operator, shapes, dtypes = linked[:3]
                if category == "kernel":
                    key = operator, shapes, dtypes
                    kernel_count[key] += 1
                    kernel_us[key] += duration_us
            row = (
                category,
                str(event.get("name", "")),
                status,
                operator,
                shapes,
                dtypes,
            )
            gpu_rows[row] += 1
            gpu_us_by_row[row] += duration_us

    kernel_total_us = category_us["kernel"]
    attributed_kernel_us = sum(kernel_us.values())
    summary = {
        "gpu_event_count": sum(category_count.values()),
        "gpu_activity_total_us": sum(category_us.values()),
        "gpu_event_count_by_category": dict(sorted(category_count.items())),
        "gpu_activity_us_by_category": {
            key: round(value, 3) for key, value in sorted(category_us.items())
        },
        "kernel_time_total_us": round(kernel_total_us, 3),
        "kernel_time_attributed_to_shape_dtype_us": round(attributed_kernel_us, 3),
        "kernel_time_attributed_to_shape_dtype_pct": (
            attributed_kernel_us / kernel_total_us * 100 if kernel_total_us else 0.0
        ),
        "kernel_event_count_attributed_to_shape_dtype": sum(kernel_count.values()),
    }
    return (
        cpu_us,
        kernel_count,
        kernel_us,
        {
            "summary": summary,
            "rows": gpu_rows,
            "row_us": gpu_us_by_row,
        },
    )


def write_phase(output: Path, phase: str, aggregate: Counter, by_rank: Counter) -> None:
    write_csv(
        output / f"{phase}_operator_shape_dtype.csv",
        ["operator", "input_shapes", "input_dtypes", "metadata_status", "call_count"],
        (
            (*key, count)
            for key, count in sorted(
                aggregate.items(), key=lambda item: (-item[1], item[0])
            )
        ),
    )
    write_csv(
        output / f"{phase}_effective_operator_shape_dtype.csv",
        ["operator", "input_shapes", "input_dtypes", "call_count"],
        (
            (key[0], key[1], key[2], count)
            for key, count in sorted(
                aggregate.items(), key=lambda item: (-item[1], item[0])
            )
            if key[3] == "shape_and_dtype"
        ),
    )
    write_csv(
        output / f"{phase}_operator_shape_dtype_by_rank.csv",
        [
            "rank",
            "operator",
            "input_shapes",
            "input_dtypes",
            "metadata_status",
            "call_count",
        ],
        (
            (*key, count)
            for key, count in sorted(
                by_rank.items(), key=lambda item: (item[0][0], -item[1], item[0])
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph_files = capture_trace_files(args.capture)
    actual_capture_files = [
        file
        for file in graph_files
        if re.search(r"graph_capture_rank_\d+_capture_", file.name)
        and rank_in_filename(file) == args.rank
    ]
    runtime_files = [
        file
        for file in runtime_trace_files(args.runtime)
        if rank_in_filename(file) == args.rank
    ]
    if not runtime_files:
        raise FileNotFoundError(f"no runtime profiler traces under {args.runtime}")

    actual_capture, actual_capture_by_rank, actual_capture_summary = collect(
        actual_capture_files
    )
    runtime, runtime_by_rank, runtime_summary = collect(runtime_files)
    runtime_cpu_us, runtime_kernel_count, runtime_kernel_us, gpu_timing = (
        collect_runtime_timing(runtime_files)
    )
    if actual_capture_files:
        write_phase(
            args.output_dir, "actual_capture", actual_capture, actual_capture_by_rank
        )
    write_phase(args.output_dir, "runtime", runtime, runtime_by_rank)

    actual_capture_effective = Counter(
        {
            key[:3]: count
            for key, count in actual_capture.items()
            if key[3] == "shape_and_dtype"
        }
    )
    runtime_effective = Counter(
        {
            key[:3]: count
            for key, count in runtime.items()
            if key[3] == "shape_and_dtype"
        }
    )
    kernel_total_us = gpu_timing["summary"]["kernel_time_total_us"]

    timing_header = [
        "operator",
        "input_shapes",
        "input_dtypes",
        "runtime_call_count",
        "runtime_cpu_duration_total_us",
        "runtime_cpu_duration_avg_us",
        "runtime_kernel_event_count",
        "runtime_kernel_time_total_us",
        "runtime_kernel_time_avg_per_call_us",
        "runtime_kernel_time_pct_of_all_runtime_kernels",
    ]
    write_csv(
        args.output_dir / "runtime_operator_timing.csv",
        timing_header,
        (
            (
                *key,
                runtime_effective[key],
                f"{runtime_cpu_us[key]:.3f}",
                f"{runtime_cpu_us[key] / runtime_effective[key]:.3f}",
                runtime_kernel_count[key],
                f"{runtime_kernel_us[key]:.3f}",
                f"{runtime_kernel_us[key] / runtime_effective[key]:.3f}",
                f"{runtime_kernel_us[key] / kernel_total_us * 100:.6f}"
                if kernel_total_us
                else "0.000000",
            )
            for key in sorted(
                runtime_effective, key=lambda item: (-runtime_kernel_us[item], item)
            )
        ),
    )

    union_keys = set(actual_capture_effective) | set(runtime_effective)
    write_csv(
        args.output_dir / "operator_summary.csv",
        [
            "operator",
            "input_shapes",
            "input_dtypes",
            "actual_capture_call_count",
            "runtime_call_count",
            "runtime_cpu_duration_total_us",
            "runtime_kernel_event_count",
            "runtime_kernel_time_total_us",
            "runtime_kernel_time_avg_per_call_us",
            "runtime_kernel_time_pct_of_all_runtime_kernels",
            "observed_in",
        ],
        (
            (
                *key,
                actual_capture_effective[key],
                runtime_effective[key],
                f"{runtime_cpu_us[key]:.3f}",
                runtime_kernel_count[key],
                f"{runtime_kernel_us[key]:.3f}",
                (
                    f"{runtime_kernel_us[key] / runtime_effective[key]:.3f}"
                    if runtime_effective[key]
                    else "0.000"
                ),
                f"{runtime_kernel_us[key] / kernel_total_us * 100:.6f}"
                if kernel_total_us
                else "0.000000",
                ",".join(
                    phase
                    for phase, phase_rows in (
                        ("actual_capture", actual_capture_effective),
                        ("runtime", runtime_effective),
                    )
                    if key in phase_rows
                ),
            )
            for key in sorted(union_keys)
        ),
    )

    gpu_rows = gpu_timing["rows"]
    gpu_row_us = gpu_timing["row_us"]
    gpu_header = [
        "category",
        "gpu_event",
        "attribution_status",
        "operator",
        "input_shapes",
        "input_dtypes",
        "event_count",
        "duration_total_us",
    ]
    write_csv(
        args.output_dir / "runtime_gpu_event_summary.csv",
        gpu_header,
        (
            (*key, gpu_rows[key], f"{gpu_row_us[key]:.3f}")
            for key in sorted(gpu_rows, key=lambda item: (-gpu_row_us[item], item))
        ),
    )
    write_csv(
        args.output_dir / "runtime_unattributed_kernel_summary.csv",
        gpu_header,
        (
            (*key, gpu_rows[key], f"{gpu_row_us[key]:.3f}")
            for key in sorted(gpu_rows, key=lambda item: (-gpu_row_us[item], item))
            if key[0] == "kernel" and key[2] != "attributed_shape_dtype"
        ),
    )

    summary = {
        "actual_capture": actual_capture_summary,
        "runtime": runtime_summary,
        "runtime_timing": gpu_timing["summary"],
        "parsed_rank": args.rank,
        "logical_union_shape_dtype_rows": len(union_keys),
        "logical_union_unique_operator_names": len({key[0] for key in union_keys}),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

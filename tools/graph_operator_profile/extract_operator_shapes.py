#!/usr/bin/env python3
"""Extract rank-scoped CPU operators with input shape/dtype."""

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
    if path is None:
        return []
    # A rank can capture more than one graph variant with the same token count
    # and profiler worker name. Those files are not duplicates: their operator
    # counters and shapes can differ. Keep every emitted trace.
    return trace_files(path)


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
    if actual_capture_files:
        write_phase(
            args.output_dir,
            "actual_capture",
            actual_capture,
            actual_capture_by_rank,
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
    union_keys = set(actual_capture_effective) | set(runtime_effective)
    write_csv(
        args.output_dir / "operator_summary.csv",
        [
            "operator",
            "input_shapes",
            "input_dtypes",
            "actual_capture_call_count",
            "runtime_call_count",
            "observed_in",
        ],
        (
            (
                *key,
                actual_capture_effective[key],
                runtime_effective[key],
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

    summary = {
        "actual_capture": actual_capture_summary,
        "runtime": runtime_summary,
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

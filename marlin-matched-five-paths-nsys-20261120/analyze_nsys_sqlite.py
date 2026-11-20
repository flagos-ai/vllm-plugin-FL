"""Read an Nsight SQLite export; never run a profiler or import CUDA packages.

Usage: python analyze_nsys_sqlite.py five-paths.sqlite --ranges profile-ranges.json
       --output analysis.json

Attribution uses runtime/driver API correlation IDs and the NVTX calling thread.
Timestamp-only GPU attribution is deliberately not used. Unknown schemas fail
with their table/column inventory rather than silently reporting partial data.
"""

import argparse
import collections
import json
import sqlite3
import statistics
from contextlib import closing
from pathlib import Path


def union_ns(intervals):
    total = 0
    stop = None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        total += max(0, end - max(start, stop if stop is not None else start))
        stop = max(end, stop if stop is not None else end)
    return total


def us(value):
    return round(value / 1000, 6)


def analyze(path, expected=None):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return analyze_db(db, path, expected)


def analyze_db(db, path, expected):
    db.row_factory = sqlite3.Row
    tables = {
        row[0]: [col[1] for col in db.execute(f'PRAGMA table_info("{row[0]}")')]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    def require(table, columns):
        if not set(columns) <= set(tables.get(table, [])):
            raise ValueError(f"Missing columns {columns} in {table}; schema={tables}")

    def rows(table):
        return [dict(row) for row in db.execute(f'SELECT * FROM "{table}"')]

    require("StringIds", ["id", "value"])
    strings = {row["id"]: row["value"] for row in rows("StringIds")}
    require("NVTX_EVENTS", ["start", "end", "globalTid"])

    def name(row, keys):
        for key in keys:
            value = row.get(key)
            if value is not None and value != "":
                return str(strings.get(value, value))
        return "<unknown>"

    ranges = []
    for row in rows("NVTX_EVENTS"):
        label = name(row, ["text", "textId"])
        if not label.startswith("marlin-five/"):
            continue
        if row["end"] is None or row["end"] <= row["start"]:
            raise ValueError(f"Unclosed/invalid NVTX range: {label}")
        row["label"] = label
        ranges.append(row)
    ranges.sort(key=lambda row: row["start"])
    labels = [row["label"] for row in ranges]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate named ranges")
    if expected is not None:
        expected_labels = [row["label"] for row in expected]
        if collections.Counter(labels) != collections.Counter(expected_labels):
            raise ValueError(
                f"Range mismatch: actual={labels}, expected={expected_labels}"
            )
    elif len(ranges) != 30:
        raise ValueError(f"Expected 30 ranges; got {len(ranges)}")

    api_tables = [
        table
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
        if table in tables
    ]
    if not api_tables:
        raise ValueError(f"No CUDA API table; schema={tables}")
    api_by_corr = collections.defaultdict(list)
    api_by_range = collections.defaultdict(list)
    for table in api_tables:
        require(table, ["start", "end", "globalTid", "correlationId", "nameId"])
        for api in rows(table):
            api["api_table"] = table
            api["name"] = name(api, ["nameId"])
            matches = [
                i
                for i, region in enumerate(ranges)
                if region["globalTid"] == api["globalTid"]
                and region["start"] <= api["start"]
                and api["end"] <= region["end"]
            ]
            if len(matches) > 1:
                raise ValueError("Overlapping diagnostic ranges on one API thread")
            if matches:
                api["range_index"] = matches[0]
                api_by_range[matches[0]].append(api)
                api_by_corr[api["correlationId"]].append(api)

    gpu_tables = [
        table
        for table in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        )
        if table in tables
    ]
    if not any("KERNEL" in table for table in gpu_tables):
        raise ValueError(f"No kernel activity table; schema={tables}")
    gpu_by_range = collections.defaultdict(list)
    unassociated = []
    for table in gpu_tables:
        require(table, ["start", "end", "correlationId"])
        for event in rows(table):
            if not any(
                region["start"] <= event["start"] < region["end"] for region in ranges
            ):
                continue
            candidates = api_by_corr.get(event["correlationId"], [])
            # Do not guess Nsight globalPid/globalTid bit packing. If the same
            # correlation ID maps to distinct diagnostic ranges, reject it.
            matches = {api["range_index"] for api in candidates}
            if len(matches) != 1:
                unassociated.append(
                    {"table": table, **event, "ranges": sorted(matches)}
                )
                continue
            index = matches.pop()
            if event["start"] < ranges[index]["start"]:
                raise ValueError("Correlated GPU event precedes its NVTX range")
            event["kind"] = table.removeprefix("CUPTI_ACTIVITY_KIND_")
            event["name"] = name(event, ["demangledName", "shortName", "nameId"])
            event["launch_api"] = [
                {k: api[k] for k in ("api_table", "name", "start", "end", "globalTid")}
                for api in candidates
            ]
            gpu_by_range[index].append(event)
    if unassociated:
        raise ValueError(
            f"Unassociated/ambiguous GPU events inside ranges: {unassociated}"
        )

    result = []
    for index, region in enumerate(ranges):
        events = sorted(gpu_by_range[index], key=lambda event: event["start"])
        kernels = [event for event in events if "KERNEL" in event["kind"]]
        if not kernels:
            raise ValueError(f"Range has no correlated kernels: {region['label']}")
        if any(event["end"] > region["end"] for event in events):
            raise ValueError(
                f"GPU activity exceeds synchronized range: {region['label']}"
            )
        first, last = events[0]["start"], max(event["end"] for event in events)
        activity_union = union_ns([(event["start"], event["end"]) for event in events])
        apis = api_by_range[index]
        syncs = [api for api in apis if "Synchronize" in api["name"]]
        stage_groups = collections.defaultdict(list)
        for event in kernels:
            stage_groups[event["name"]].append(event["end"] - event["start"])
        gaps = []
        running_end = first
        for event in events:
            if event["start"] > running_end:
                gaps.append((running_end, event["start"]))
            running_end = max(running_end, event["end"])
        result.append(
            {
                "label": region["label"],
                "range_us": us(region["end"] - region["start"]),
                "gpu_span_us": us(last - first),
                "gpu_activity_union_us": us(activity_union),
                "gpu_kernel_sum_us": us(
                    sum(event["end"] - event["start"] for event in kernels)
                ),
                "gpu_internal_unoccupied_us": us(last - first - activity_union),
                "range_prefix_before_first_gpu_us": us(first - region["start"]),
                "range_suffix_after_last_gpu_us": us(region["end"] - last),
                "kernel_count": len(kernels),
                "nonkernel_gpu_event_count": len(events) - len(kernels),
                "sync_api_count": len(syncs),
                "sync_api_union_us": us(
                    union_ns([(api["start"], api["end"]) for api in syncs])
                ),
                "all_cuda_api_union_us": us(
                    union_ns([(api["start"], api["end"]) for api in apis])
                ),
                "kernel_groups": [
                    {
                        "name": kernel,
                        "calls": len(times),
                        "sum_us": us(sum(times)),
                        "median_us": us(statistics.median(times)),
                    }
                    for kernel, times in stage_groups.items()
                ],
                "internal_gpu_gaps_us": [us(end - start) for start, end in gaps],
                "gpu_events": events,
                "cuda_apis": apis,
            }
        )
    groups = collections.defaultdict(list)
    for row in result:
        _, case, _, backend = row["label"].split("/")
        groups[(case, backend)].append(row)
    summaries = []
    for (case, backend), samples in groups.items():
        summaries.append(
            {
                "case": case,
                "backend": backend,
                "samples": len(samples),
                **{
                    f"median_{metric}": statistics.median(
                        row[metric] for row in samples
                    )
                    for metric in (
                        "range_us",
                        "gpu_span_us",
                        "gpu_kernel_sum_us",
                        "gpu_internal_unoccupied_us",
                        "kernel_count",
                    )
                },
            }
        )
    return {
        "input_sqlite": str(path.resolve()),
        "attribution": "CUDA API correlation ID + NVTX calling thread",
        "formal_g6": False,
        "limitations": [
            "Range includes the helper's explicit synchronize; sync API duration overlaps GPU work.",
            "GPU internal gaps are empty intervals in captured activity, not proven CPU launch overhead.",
            "Nsight Systems provides no DRAM/L2/atomic/warp-stall hardware counters here.",
            "This is one-call profiling with retained diagnostic inputs, not the formal benchmark timer.",
            "Grid dimensions show launched CTAs; inactive expert CTAs can return immediately.",
        ],
        "schema": tables,
        "range_count": len(result),
        "summaries": summaries,
        "ranges": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--ranges", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.ranges.read_text()) if args.ranges else None
    result = analyze(args.sqlite, expected)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ranges": result["range_count"],
                "case_backend_groups": len(result["summaries"]),
            }
        )
    )


if __name__ == "__main__":
    main()

"""Correlate CUDA activities to worker-local NVTX, retaining sample and rank."""

import csv
import hashlib
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def union_ns(intervals):
    total = 0
    left = right = None
    for a, b in sorted(intervals):
        if left is None:
            left, right = a, b
        elif a <= right:
            right = max(right, b)
        else:
            total += right - left
            left, right = a, b
    return total + (0 if left is None else right - left)


def analyze(backend):
    folder = ROOT / backend
    result = json.loads((folder / "result.json").read_text())
    assert result["completed"]
    pids = {a["pid"]: a["rank"] for a in result["annotations"]}
    connection = sqlite3.connect(folder / "prefill.sqlite")
    connection.row_factory = sqlite3.Row
    strings = dict(connection.execute("select id,value from StringIds"))
    tables = {
        r[0]
        for r in connection.execute("select name from sqlite_master where type='table'")
    }
    nvtx_by_tid = defaultdict(list)
    nvtx_core_count = 0
    for row in connection.execute("select * from NVTX_EVENTS where end is not null"):
        item = dict(row)
        item["name"] = item["text"] or strings.get(item["textId"], "")
        if item["name"].startswith(("request:", "gdn.", "model.stage:", "comm:")):
            nvtx_by_tid[item["globalTid"]].append(item)
            nvtx_core_count += item["name"].startswith("gdn.core:")
    assert nvtx_core_count == 360, (backend, nvtx_core_count)
    apis_by_tid = defaultdict(list)
    for kind in ("RUNTIME", "DRIVER"):
        table = "CUPTI_ACTIVITY_KIND_" + kind
        if table in tables:
            for row in connection.execute("select * from " + table):
                item = dict(row)
                apis_by_tid[item["globalTid"]].append(item)
    correlations = {}
    for tid, apis in apis_by_tid.items():
        ranges = sorted(nvtx_by_tid[tid], key=lambda r: (r["start"], -r["end"]))
        active, index = [], 0
        for api in sorted(apis, key=lambda r: r["start"]):
            active = [r for r in active if r["end"] >= api["end"]]
            while index < len(ranges) and ranges[index]["start"] <= api["start"]:
                if ranges[index]["end"] >= api["end"]:
                    active.append(ranges[index])
                index += 1
            parents = [
                r
                for r in active
                if r["start"] <= api["start"] and r["end"] >= api["end"]
            ]
            parents.sort(key=lambda r: (r["start"], -r["end"]))
            key = (tid & ~0xFFFFFF, api["correlationId"])
            correlations[key] = (api, [r["name"] for r in parents])

    events = []
    for kind in ("KERNEL", "MEMCPY", "MEMSET"):
        table = "CUPTI_ACTIVITY_KIND_" + kind
        if table not in tables:
            continue
        for row in connection.execute("select * from " + table):
            event = dict(row)
            api, parents = correlations.get(
                (event["globalPid"], event["correlationId"]), (None, [])
            )
            request = next((p for p in parents if p.startswith("request:")), None)
            core = next((p for p in parents if p.startswith("gdn.core:")), None)
            stage = next(
                (p for p in reversed(parents) if p.startswith("gdn.stage:")), None
            )
            model_stage = next(
                (p for p in reversed(parents) if p.startswith("model.stage:")), None
            )
            comm = next((p for p in reversed(parents) if p.startswith("comm:")), None)
            # Nsight packs a VM ID above the 24-bit process ID.
            pid = (event["globalPid"] >> 24) & 0xFFFFFF
            event.update(
                kind=kind.lower(),
                rank=pids.get(pid),
                sample=int(request.split(":")[1]) if request else None,
                core=core,
                stage=stage.split(":", 1)[1] if stage else "adapter_other",
                model_stage=model_stage,
                comm=comm,
                name=strings.get(event.get("shortName"), kind.lower()),
                duration_ms=(event["end"] - event["start"]) / 1e6,
                api_name=strings.get(api["nameId"], "") if api else None,
            )
            if core:
                assert event["rank"] is not None and event["sample"] in range(3), event
            events.append(event)
    connection.close()

    core_events = [e for e in events if e["core"]]
    core_groups = defaultdict(list)
    for event in core_events:
        core_groups[event["rank"], event["sample"], event["core"]].append(event)
    assert len(core_groups) == 360, (backend, len(core_groups))
    per_layer = []
    for (rank, sample, core), items in sorted(core_groups.items()):
        per_layer.append(
            {
                "rank": rank,
                "sample": sample,
                "layer": core.split(":", 1)[1],
                "gpu_sum_ms": sum(e["duration_ms"] for e in items),
                "gpu_union_ms": union_ns((e["start"], e["end"]) for e in items) / 1e6,
                "gpu_span_ms": (
                    max(e["end"] for e in items) - min(e["start"] for e in items)
                )
                / 1e6,
                "kernel_count": sum(e["kind"] == "kernel" for e in items),
                "memcpy_bytes": sum(
                    e.get("bytes", 0) for e in items if e["kind"] == "memcpy"
                ),
            }
        )
    per_request = []
    stage_names = sorted({e["stage"] for e in core_events})
    for rank in range(4):
        for sample in range(3):
            items = [
                e for e in core_events if e["rank"] == rank and e["sample"] == sample
            ]
            all_items = [
                e for e in events if e["rank"] == rank and e["sample"] == sample
            ]
            layers = [
                r for r in per_layer if r["rank"] == rank and r["sample"] == sample
            ]
            assert len(layers) == 30
            per_request.append(
                {
                    "rank": rank,
                    "sample": sample,
                    "core_gpu_sum_ms": sum(e["duration_ms"] for e in items),
                    "core_gpu_span_sum_ms": sum(r["gpu_span_ms"] for r in layers),
                    "core_gpu_union_ms": union_ns((e["start"], e["end"]) for e in items)
                    / 1e6,
                    "model_gpu_span_ms": (
                        max(e["end"] for e in all_items)
                        - min(e["start"] for e in all_items)
                    )
                    / 1e6,
                    "model_gpu_union_ms": union_ns(
                        (e["start"], e["end"]) for e in all_items
                    )
                    / 1e6,
                    "stages_ms": {
                        stage: sum(
                            e["duration_ms"] for e in items if e["stage"] == stage
                        )
                        for stage in stage_names
                    },
                }
            )
    kernels = defaultdict(list)
    for event in core_events:
        grid = tuple(event.get("grid" + dim, 0) for dim in "XYZ")
        block = tuple(event.get("block" + dim, 0) for dim in "XYZ")
        kernels[event["stage"], event["kind"], event["name"], grid, block].append(event)
    kernel_rows = []
    for (stage, kind, name, grid, block), items in kernels.items():
        kernel_rows.append(
            {
                "stage": stage,
                "kind": kind,
                "name": name,
                "grid": list(grid),
                "block": list(block),
                "calls": len(items),
                "total_gpu_ms": sum(e["duration_ms"] for e in items),
                "mean_per_request_rank_ms": sum(e["duration_ms"] for e in items) / 12,
                "median_call_ms": statistics.median(e["duration_ms"] for e in items),
                "registers_per_thread": sorted(
                    {e.get("registersPerThread", 0) for e in items}
                ),
                "shared_memory_bytes": sorted(
                    {
                        e.get("staticSharedMemory", 0) + e.get("dynamicSharedMemory", 0)
                        for e in items
                    }
                ),
            }
        )
    kernel_rows.sort(key=lambda r: -r["total_gpu_ms"])
    summary = {
        "backend": backend,
        "core_ranges": nvtx_core_count,
        "annotated_gpu_core_groups": len(core_groups),
        "sm_counts": [r["sm_count"] for r in result["annotations"]],
        "stage_mean_ms": {
            stage: statistics.mean(r["stages_ms"][stage] for r in per_request)
            for stage in stage_names
        },
        "mean_core_gpu_ms": statistics.mean(r["core_gpu_sum_ms"] for r in per_request),
        "median_core_gpu_ms": statistics.median(
            r["core_gpu_sum_ms"] for r in per_request
        ),
        "mean_core_gpu_span_sum_ms": statistics.mean(
            r["core_gpu_span_sum_ms"] for r in per_request
        ),
        "mean_model_gpu_span_ms": statistics.mean(
            r["model_gpu_span_ms"] for r in per_request
        ),
        "mean_model_gpu_union_ms": statistics.mean(
            r["model_gpu_union_ms"] for r in per_request
        ),
        "core_memcpy_count": sum(e["kind"] == "memcpy" for e in core_events),
        "core_memcpy_bytes": sum(
            e.get("bytes", 0) for e in core_events if e["kind"] == "memcpy"
        ),
        "per_request_rank": per_request,
        "kernels": kernel_rows,
    }
    (folder / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in (("layers", per_layer), ("kernels", kernel_rows)):
        with (folder / (name + ".csv")).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main():
    summary = {backend: analyze(backend) for backend in ("flashinfer", "triton")}
    for backend, row in summary.items():
        manifest = json.loads((ROOT / backend / "launch.json").read_text())
        for name, digest in manifest["hashes"].items():
            assert (
                hashlib.sha256(
                    (ROOT / backend / "source" / name).read_bytes()
                ).hexdigest()
                == digest
            )
        print(
            backend,
            "core GPU ms",
            row["mean_core_gpu_ms"],
            "span sum",
            row["mean_core_gpu_span_sum_ms"],
        )
        print("stages", row["stage_mean_ms"])
        for kernel in row["kernels"][:10]:
            print(
                kernel["name"][:120],
                kernel["grid"],
                kernel["calls"],
                kernel["mean_per_request_rank_ms"],
            )
    summary["core_gpu_ms_saved"] = (
        summary["flashinfer"]["mean_core_gpu_ms"]
        - summary["triton"]["mean_core_gpu_ms"]
    )
    summary["core_gpu_span_ms_saved"] = (
        summary["flashinfer"]["mean_core_gpu_span_sum_ms"]
        - summary["triton"]["mean_core_gpu_span_sum_ms"]
    )
    (ROOT / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

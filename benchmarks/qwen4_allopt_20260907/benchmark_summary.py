#!/usr/bin/env python3
"""Validate vLLM bench JSON and retain every formal round's key statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
)


def one_round(path: Path, label: str, prompts: int, input_len: int, output_len: int) -> dict[str, Any]:
    record: dict[str, Any] = {"label": label, "path": str(path), "errors": []}
    errors: list[str] = record["errors"]
    status_path = path.with_name(f"{path.stem}.status.json")
    if status_path.is_file():
        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
            record["command_status"] = status_data
            if status_data.get("returncode") != 0:
                errors.append(f"benchmark command returned {status_data.get('returncode')}")
        except (OSError, ValueError):
            errors.append("benchmark command status is invalid")
    else:
        errors.append("benchmark command status is missing")
    if not path.is_file():
        errors.append("result JSON is missing")
        record["status"] = "fail"
        return record
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        errors.append("result JSON is invalid")
        record["status"] = "fail"
        return record
    if not isinstance(data, dict):
        errors.append("result JSON is not an object")
        record["status"] = "fail"
        return record

    expected = {
        "num_prompts": prompts,
        "completed": prompts,
        "failed": 0,
        "total_input_tokens": prompts * input_len,
        "total_output_tokens": prompts * output_len,
    }
    record["configuration"] = {
        "num_prompts": data.get("num_prompts"),
        "input_len": input_len,
        "output_len": output_len,
        "max_concurrency": data.get("max_concurrency"),
        "request_rate": data.get("request_rate"),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            errors.append(f"{key}={data.get(key)!r}, expected {value!r}")

    for key, expected_len in (("input_lens", input_len), ("output_lens", output_len)):
        values = data.get(key)
        if not isinstance(values, list) or len(values) != prompts or set(values) != {expected_len}:
            errors.append(f"{key} is not exactly {prompts} entries of {expected_len}")

    record["metrics"] = {key: data.get(key) for key in METRICS}
    for key in ("p99_ttft_ms", "p99_tpot_ms", "p99_itl_ms"):
        if not isinstance(data.get(key), (int, float)):
            errors.append(f"{key} is missing from vLLM result")
    record["duration_s"] = data.get("duration")
    record["status"] = "pass" if not errors else "fail"
    return record


def aggregate(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group_name, group in (("all_formal", rounds), ("formal2_plus3", rounds[1:])):
        out[group_name] = {"round_count": len(group), "metrics": {}, "range_drift": {}}
        for metric in METRICS:
            values = [r.get("metrics", {}).get(metric) for r in group]
            values = [float(v) for v in values if isinstance(v, (int, float))]
            out[group_name]["metrics"][metric] = mean(values) if values else None
            if values:
                low, high = min(values), max(values)
                center = abs(mean(values))
                out[group_name]["range_drift"][metric] = {
                    "min": low,
                    "max": high,
                    "range": high - low,
                    "relative_range_pct": ((high - low) / center * 100.0) if center else None,
                }
            else:
                out[group_name]["range_drift"][metric] = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--formal-rounds", type=int, default=3)
    parser.add_argument("--formal-prompts", type=int, default=128)
    parser.add_argument("--warmup-prompts", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rounds: list[dict[str, Any]] = []
    warmup = one_round(args.benchmark_dir / "warmup.json", "warmup", args.warmup_prompts, args.input_len, args.output_len)
    for index in range(1, args.formal_rounds + 1):
        rounds.append(one_round(args.benchmark_dir / f"formal{index}.json", f"formal{index}", args.formal_prompts, args.input_len, args.output_len))
    result: dict[str, Any] = {
        "status": "pass" if warmup["status"] == "pass" and all(r["status"] == "pass" for r in rounds) else "fail",
        "workload": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "warmup_prompts": args.warmup_prompts,
            "formal_prompts": args.formal_prompts,
            "formal_rounds": args.formal_rounds,
            "max_concurrency": 64,
            "seed": 12345,
            "temperature": 0,
            "ignore_eos": True,
        },
        "warmup": warmup,
        "formal_rounds": rounds,
        "aggregates": aggregate(rounds),
    }
    result["errors"] = []
    if warmup["status"] != "pass":
        result["errors"].append("warmup validation failed")
    for row in rounds:
        if row["status"] != "pass":
            result["errors"].append(f"{row['label']} validation failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"benchmark summary: {result['status']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

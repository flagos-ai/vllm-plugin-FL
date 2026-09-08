# Copyright (c) 2026 BAAI. All rights reserved.

"""Format vLLM serve benchmark JSON output for FlagCICD benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _num(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _metric(value: Any) -> dict[str, Any]:
    number = _num(value)
    return {"values": [number], "avg": number, "p50": number, "p99": number}


def _latency_metric(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    avg = _num(data.get(f"mean_{prefix}_ms"))
    return {
        "values": [avg],
        "avg": avg,
        "p50": _num(data.get(f"median_{prefix}_ms")),
        "p99": _num(data.get(f"p99_{prefix}_ms")),
    }


def build_report(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "ttft_ms": _latency_metric(data, "ttft"),
        "tpot_ms": _latency_metric(data, "tpot"),
        "itl_ms": _latency_metric(data, "itl"),
        "e2el_ms": _latency_metric(data, "e2el"),
        "request_throughput_req_s": _metric(data.get("request_throughput")),
        "request_goodput_req_s": _metric(data.get("request_goodput")),
        "output_throughput_tok_s": _metric(data.get("output_throughput")),
        "total_token_throughput_tok_s": _metric(data.get("total_token_throughput")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-type",
        required=True,
        choices=["serve"],
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = build_report(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

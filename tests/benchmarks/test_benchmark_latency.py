# Copyright (c) 2026 BAAI. All rights reserved.

"""Smoke test for vllm bench latency."""

import json

import pytest

from tests.benchmarks.utils import load_benchmark_case, run_command, to_cli_args


@pytest.mark.benchmark
def test_benchmark_latency(tmp_path):
    case = load_benchmark_case()
    params = case.get("parameters", {})

    output_json = tmp_path / "latency_result.json"
    command = ["vllm", "bench", "latency"]
    command.extend(to_cli_args(params))
    command.extend(["--output-json", str(output_json)])

    result = run_command(command)
    assert result.returncode == 0, result.stderr
    assert output_json.exists()

    data = json.loads(output_json.read_text())
    avg_latency = data.get("avg_latency", data.get("mean_latency", 0))
    assert avg_latency > 0
    latencies = data.get("latencies")
    num_iters = data.get("num_iters", 0)
    assert (latencies is not None and len(latencies) > 0) or num_iters > 0

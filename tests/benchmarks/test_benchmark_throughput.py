# Copyright (c) 2026 BAAI. All rights reserved.

"""Smoke test for vllm bench throughput."""

import json

import pytest

from tests.benchmarks.utils import load_benchmark_case, run_command, to_cli_args


@pytest.mark.benchmark
def test_benchmark_throughput(tmp_path):
    case = load_benchmark_case()
    params = case.get("parameters", {})

    output_json = tmp_path / "throughput_result.json"
    command = ["vllm", "bench", "throughput"]
    command.extend(to_cli_args(params))
    command.extend(["--output-json", str(output_json)])

    result = run_command(command)
    assert result.returncode == 0, result.stderr
    assert output_json.exists()

    data = json.loads(output_json.read_text())
    num_requests = data.get("num_requests", data.get("num_prompts", 0))
    tokens_per_second = data.get("tokens_per_second", data.get("output_throughput", 0))
    assert num_requests > 0
    assert tokens_per_second > 0

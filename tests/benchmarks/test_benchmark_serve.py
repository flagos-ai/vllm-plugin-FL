# Copyright (c) 2026 BAAI. All rights reserved.

"""Smoke test for vllm bench serve."""

import json

import pytest

from tests.benchmarks.utils import load_benchmark_case, run_command, to_cli_args
from tests.e2e_tests.serving.server_helper import VllmServer


@pytest.mark.benchmark
def test_benchmark_serve(tmp_path):
    case = load_benchmark_case()

    server_params = dict(case.get("server_parameters", {}))
    client_params = dict(case.get("client_parameters", {}))

    model = server_params.pop("model")
    served_model_name = server_params.pop("served_model_name", "")
    tp_size = int(server_params.pop("tensor_parallel_size", 1))
    server_extra_args = to_cli_args(server_params)

    result_json = tmp_path / "serve_result.json"

    with VllmServer(
        model=model,
        tp_size=tp_size,
        served_model_name=served_model_name,
        extra_args=server_extra_args,
    ) as server:
        command = [
            "vllm",
            "bench",
            "serve",
            "--host",
            server.host,
            "--port",
            str(server.port),
        ]
        command.extend(to_cli_args(client_params))
        command.extend(
            [
                "--save-result",
                "--result-dir",
                str(tmp_path),
                "--result-filename",
                result_json.name,
            ]
        )

        result = run_command(command)

    assert result.returncode == 0, result.stderr
    assert result_json.exists()

    data = json.loads(result_json.read_text())
    assert data.get("completed", 0) > 0
    assert data.get("failed", 0) == 0

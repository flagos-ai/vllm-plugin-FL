import subprocess

import pytest

MODEL_NAME = "/models/Qwen3-Next-80B-A3B-Instruct"


@pytest.mark.benchmark
def test_bench_throughput():
    command = [
        "vllm",
        "bench",
        "throughput",
        "--model",
        MODEL_NAME,
        "--tensor-parallel-size",
        "4",
        "--dataset-name",
        "random",
        "--input-len",
        "6144",
        "--output-len",
        "1024",
        "--num-prompts",
        "1000",
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        "2048",
        "--load-format",
        "dummy",
        "--gpu-memory-utilization",
        "0.85",
        "--compilation-config",
        '{"cudagraph_mode": "FULL_AND_PIECEWISE"}',
        # "--enforce-eager",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

"""Shared H20 runners must not execute overlapping E2E GPU workloads."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_shared_h20_gpu_jobs_are_serialized():
    workflow = yaml.safe_load((ROOT / ".github/workflows/_e2e_test.yml").read_text())
    job = workflow["jobs"]["e2e-test"]
    concurrency = job["concurrency"]
    expression = concurrency["group"]
    condition, choices = (
        expression.removeprefix("${{ ")
        .removesuffix(" }}")
        .split(" && 'nvidia-h20-e2e-gpus' || ")
    )
    assert condition == "inputs.platform == 'nvidia' && inputs.device == 'h20'"
    assert choices == (
        "format('e2e-{0}-{1}-{2}-{3}-{4}', github.run_id, github.run_attempt, "
        "inputs.platform, inputs.device, inputs.task)"
    )
    assert concurrency["cancel-in-progress"] is False
    assert concurrency["queue"] == "max"
    caller = yaml.safe_load((ROOT / ".github/configs/nvidia.yml").read_text())
    assert caller["device_runner_labels"]["h20"]
    commands = [step.get("run", "") for step in job["steps"]]
    assert any("python tests/run.py" in command for command in commands)
    assert not any("gpu-memory-utilization" in command for command in commands)

"""The graph launcher's memory budget must reach the actual serve command."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("launcher", ["graph", "eager"])
@pytest.mark.parametrize("budget", [None, "0.70", "0.83"])
def test_memory_budget_forwarded(tmp_path, budget, launcher):
    binary = tmp_path / "vllm"
    binary.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    binary.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        MODEL_PATH=str(model),
        PORT="8231",
        SERVER_PID_FILE=str(tmp_path / "server.pid"),
    )
    env.pop("GPU_MEMORY_UTILIZATION", None)
    if budget is not None:
        env["GPU_MEMORY_UTILIZATION"] = budget
    result = subprocess.run(
        ["bash", str(ROOT / f"tools/adaptation-gate-cases/run_serve_{launcher}.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.splitlines()
    assert args[args.index("--gpu-memory-utilization") + 1] == (budget or "0.92")
    assert ("--enforce-eager" in args) == (launcher == "eager")


@pytest.mark.parametrize(
    "config",
    [
        None,
        '{"mode":0,"cudagraph_mode":"FULL","custom_ops":["all"],"ir_enable_torch_wrap":false}',
    ],
)
def test_graph_compilation_config_reaches_cli_as_one_argument(tmp_path, config):
    import json

    binary = tmp_path / "vllm"
    binary.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    binary.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        MODEL_PATH=str(model),
        PORT="8231",
        SERVER_PID_FILE=str(tmp_path / "server.pid"),
    )
    env.pop("GRAPH_COMPILATION_CONFIG", None)
    if config is not None:
        env["GRAPH_COMPILATION_CONFIG"] = config
    result = subprocess.run(
        ["bash", str(ROOT / "tools/adaptation-gate-cases/run_serve_graph.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.splitlines()
    assert "--enforce-eager" not in args
    if config is None:
        assert "--compilation-config" not in args
    else:
        actual = args[args.index("--compilation-config") + 1]
        assert actual == config
        assert json.loads(actual)["cudagraph_mode"] == "FULL"

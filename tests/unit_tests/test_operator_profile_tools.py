# SPDX-License-Identifier: Apache-2.0

import csv
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_PATH = REPO_ROOT / "tools/operator_profile/extract_operator_shapes.py"
PROFILE_SCRIPT = REPO_ROOT / "tools/operator_profile/profile.sh"
SERVE_SCRIPT = REPO_ROOT / "tools/operator_profile/serve.sh"
sys.path.insert(0, str(EXTRACTOR_PATH.parent))


def load_extractor():
    spec = importlib.util.spec_from_file_location(
        "operator_profile_extractor", EXTRACTOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_iter_events_accepts_common_trace_formats(tmp_path, compact, compressed):
    extractor = load_extractor()
    events = [
        {"name": "aten::add", "cat": "cpu_op", "args": {}},
        {"name": "kernel", "cat": "kernel", "dur": 1.5, "args": {}},
    ]
    payload = {"metadata": "x" * 5000, "traceEvents": events}
    suffix = ".json.gz" if compressed else ".json"
    path = tmp_path / f"trace{suffix}"
    text = json.dumps(
        payload,
        separators=(",", ":") if compact else None,
        indent=None if compact else 4,
    )
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as target:
            target.write(text)
    else:
        path.write_text(text, encoding="utf-8")

    assert list(extractor.iter_events(path)) == events


def test_iter_events_rejects_missing_trace_events(tmp_path):
    extractor = load_extractor()
    path = tmp_path / "trace.json"
    path.write_text('{"metadata": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="traceEvents not found"):
        list(extractor.iter_events(path))


def test_profile_script_rejects_nested_stale_trace(tmp_path):
    run_dir = tmp_path / "run"
    nested = run_dir / "profile/rank0"
    nested.mkdir(parents=True)
    (nested / "stale.pt.trace.json.gz").touch()

    commands = [
        ["bash", str(PROFILE_SCRIPT), "model", str(run_dir)],
        ["bash", str(SERVE_SCRIPT), "/model", "model", str(run_dir)],
    ]
    for command in commands:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "profile directory already contains a trace" in result.stderr


def make_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_serve_script_cleans_configured_flaggems_oplist(tmp_path):
    run_dir = tmp_path / "run"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(fake_bin / "vllm", "#!/bin/sh\nexit 0\n")
    oplist = tmp_path / "evidence/custom-oplist.txt"
    oplist.parent.mkdir()
    oplist.write_text("stale", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(oplist),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        ["bash", str(SERVE_SCRIPT), "/model", "model", str(run_dir)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert not oplist.exists()


def test_profile_script_moves_configured_flaggems_oplist(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "profile").mkdir(parents=True)
    (run_dir / "results").mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(fake_bin / "curl", "#!/bin/sh\nexit 0\n")
    make_executable(fake_bin / "python3", "#!/bin/sh\nexit 0\n")
    oplist = tmp_path / "evidence/custom-oplist.txt"
    oplist.parent.mkdir()
    oplist.write_text("current", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(oplist),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        ["bash", str(PROFILE_SCRIPT), "model", str(run_dir)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    copied = run_dir / "results/flaggems_enable_oplist.txt"
    assert result.returncode == 0
    assert copied.read_text(encoding="utf-8") == "current"
    assert not oplist.exists()


def test_serve_script_uses_supported_vllm_profiler_config(tmp_path):
    profiler_config_type = pytest.importorskip("vllm.config").ProfilerConfig
    run_dir = tmp_path / "run"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation = tmp_path / "invocation.json"
    make_executable(
        fake_bin / "vllm",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['PROFILE_TEST_INVOCATION']).write_text(json.dumps({"
        "'argv': sys.argv[1:], "
        "'oplist': os.environ['FLAGGEMS_ENABLE_OPLIST_PATH']}))\n",
    )
    env = os.environ.copy()
    env.pop("FLAGGEMS_ENABLE_OPLIST_PATH", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROFILE_TEST_INVOCATION": str(invocation),
        }
    )
    subprocess.run(
        [
            "bash",
            str(SERVE_SCRIPT),
            "/model",
            "model",
            str(run_dir),
            "--tensor-parallel-size",
            "2",
            "--enforce-eager",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    recorded = json.loads(invocation.read_text())
    argv = recorded["argv"]
    assert argv[:4] == ["serve", "/model", "--served-model-name", "model"]
    assert argv[4] == "--profiler-config"
    assert argv[6:] == ["--tensor-parallel-size", "2", "--enforce-eager"]
    config = profiler_config_type(**json.loads(argv[5]))
    assert config.profiler == "torch"
    assert config.torch_profiler_dir == str(run_dir / "profile")
    assert config.torch_profiler_record_shapes
    assert config.ignore_frontend
    assert not config.torch_profiler_with_stack
    assert not config.torch_profiler_with_memory
    assert not config.torch_profiler_dump_cuda_time_total
    assert not config.capture_torch_profiler
    assert config.delay_iterations == config.max_iterations == 0
    assert config.warmup_iterations == 0
    assert recorded["oplist"] == str(run_dir / "flaggems_enable_oplist.txt")


@pytest.mark.parametrize("compressed", [False, True])
def test_extractor_selects_vllm_runtime_rank_zero(tmp_path, compressed):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    capture_dir = profile_dir / "capture_traces"
    capture_dir.mkdir()
    events = [
        {
            "cat": "cpu_op",
            "name": "aten::mul",
            "args": {
                "External id": 1,
                "Input Dims": [[4], [4]],
                "Input type": ["float", "float"],
            },
        },
        {
            "cat": "kernel",
            "name": "mul_kernel",
            "dur": 2,
            "args": {"External id": 1},
        },
    ]
    runtime_path = profile_dir / "dp0_pp0_tp0_dcp0_ep0_rank0.123.pt.trace.json"
    payload = json.dumps({"traceEvents": events})
    if compressed:
        runtime_path = runtime_path.with_suffix(".json.gz")
        with gzip.open(runtime_path, "wt", encoding="utf-8") as target:
            target.write(payload)
    else:
        runtime_path.write_text(payload, encoding="utf-8")
    for ignored_path in (
        profile_dir / "dp0_pp0_tp1_dcp0_ep1_rank1.123.pt.trace.json",
        capture_dir / "graph_capture_rank_0.123.pt.trace.json",
    ):
        ignored_path.write_text(
            json.dumps(
                {"traceEvents": [{"cat": "kernel", "name": "ignored", "dur": 99}]}
            ),
            encoding="utf-8",
        )
    output = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(EXTRACTOR_PATH),
            "--trace-dir",
            str(profile_dir),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output / "summary.json").read_text())
    assert summary["runtime_trace_files"] == [str(runtime_path)]
    assert summary["scope"]["rank"] == 0
    assert not summary["scope"]["graph_capture_included"]
    assert summary["kernel_event_count"] == 1
    assert summary["kernel_time_total_us"] == 2
    assert all(
        value for value in summary["conservation"].values() if isinstance(value, bool)
    )
    with (output / "kernel_shape_dtype.csv").open(newline="") as source:
        details = list(csv.DictReader(source))
    assert len(details) == 1
    assert details[0]["operator_name"] == "aten::mul"
    assert details[0]["kernel_name"] == "mul_kernel"
    assert details[0]["mapping_status"] == "operator_shape_matched"
    assert json.loads(details[0]["input_shapes"]) == [[4], [4]]
    assert json.loads(details[0]["input_dtypes"]) == ["float", "float"]

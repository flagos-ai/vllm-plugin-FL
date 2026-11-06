# SPDX-License-Identifier: Apache-2.0

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

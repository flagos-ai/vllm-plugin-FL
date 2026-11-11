"""CI wiring and fail-closed checks, not hardware acceptance evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_gate_workflow_is_required_for_iluvatar():
    platform = yaml.safe_load(
        (ROOT / ".github/workflows/_platform_test.yml").read_text()
    )
    jobs = platform["jobs"]
    gate = jobs["iluvatar-gates"]
    assert gate["uses"] == "./.github/workflows/_iluvatar_gate.yml"
    assert "inputs.platform == 'iluvatar'" in gate["if"]
    assert "iluvatar-gates" in jobs["notify"]["needs"]
    assert "functional" in jobs["notify"]["needs"]
    assert "unit" not in jobs["notify"]["needs"]
    assert gate["needs"] == ["setup", "benchmark"]
    assert "!cancelled()" in gate["if"]
    assert "needs.setup.result == 'success'" in gate["if"]
    assert jobs["functional"]["uses"] == "./.github/workflows/_functional_test.yml"
    status = jobs["notify"]["with"]["status"]
    assert "needs.iluvatar-gates.result == 'success'" in status
    assert "inputs.platform != 'iluvatar'" in status
    assert "needs.iluvatar-gates.result == 'cancelled'" in status
    workflow = yaml.safe_load((ROOT / gate["uses"][2:]).read_text())
    steps = workflow["jobs"]["gates"]["steps"]
    assert any(
        s.get("run") == "bash .github/scripts/iluvatar/run_gates.sh" for s in steps
    )
    assert any(
        s.get("if") == "always()" and s.get("uses") == "actions/upload-artifact@v4"
        for s in steps
    )


def test_gate_missing_model_fails_before_server_launch(tmp_path):
    env = dict(
        os.environ,
        ILUVATAR_MODEL_27B=str(tmp_path / "missing-27b"),
        ILUVATAR_MODEL_35B=str(tmp_path / "missing-35b"),
        FLAGCX_PATH=str(tmp_path / "missing-flagcx"),
        ILUVATAR_GATE_RESULTS=str(tmp_path / "results"),
    )
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/iluvatar/run_gates.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Missing model:" in result.stderr
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("missing", ["configuration", "library", "wrapper"])
def test_gate_missing_flagcx_fails_before_server_launch(tmp_path, missing):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    flagcx = tmp_path / "flagcx"
    library = flagcx / "build/lib/libflagcx.so"
    if missing == "wrapper":
        library.parent.mkdir(parents=True)
        library.touch()
    env = dict(
        os.environ,
        ILUVATAR_MODEL_27B=str(model),
        ILUVATAR_MODEL_35B=str(model),
        ILUVATAR_GATE_RESULTS=str(tmp_path / "results"),
    )
    env.pop("FLAGCX_PATH", None)
    if missing != "configuration":
        env["FLAGCX_PATH"] = str(flagcx)
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/iluvatar/run_gates.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    if missing == "configuration":
        assert "ILUVATAR_FLAGCX_PATH" in result.stderr
        assert "inside the container" in result.stderr
    else:
        expected = (
            library
            if missing == "library"
            else flagcx / "plugin/interservice/flagcx_wrapper.py"
        )
        assert f"Missing FlagCX file: {expected}" in result.stderr
    assert not (tmp_path / "results").exists()


def test_ci_leaves_memory_budget_at_vllm_default():
    source = (ROOT / ".github/scripts/iluvatar/run_gates.sh").read_text()
    assert "GPU_MEMORY_UTILIZATION" not in source
    assert "--gpu-memory-utilization" not in source


@pytest.mark.parametrize("override", [None, '{"mode":3,"cudagraph_mode":"FULL"}'])
def test_ci_graph_groups_keep_default_separate(override, tmp_path):
    source = (ROOT / ".github/scripts/iluvatar/run_gates.sh").read_text()
    assert "for mode in eager graph-workaround graph-default" in source
    assignment = next(
        line
        for line in source.splitlines()
        if line.startswith("WORKAROUND_COMPILATION_CONFIG=")
    )
    start = source.index("        unset GRAPH_COMPILATION_CONFIG")
    end = source.index("        printf", start)
    block = source[start:end]
    capture = tmp_path / "argv.json"
    launcher = tmp_path / "vllm"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"open({str(capture)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    launcher.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    for mode in ("eager", "graph-default", "graph-workaround"):
        env = dict(os.environ, mode=mode)
        env.pop("GRAPH_COMPILATION_CONFIG", None)
        if override is not None:
            env["GRAPH_COMPILATION_CONFIG"] = override
        result = subprocess.run(
            [
                "bash",
                "-c",
                assignment
                + "\n"
                + block
                + '\nprintf "%s\\n%s" "$serve_mode" "${GRAPH_COMPILATION_CONFIG:-unset}"',
            ],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        serve_mode, actual = result.stdout.split("\n", 1)
        assert serve_mode == ("eager" if mode == "eager" else "graph")
        if mode == "graph-workaround":
            assert json.loads(actual) == (
                json.loads(override)
                if override
                else {
                    "mode": 0,
                    "cudagraph_mode": "FULL",
                    "custom_ops": ["all"],
                    "ir_enable_torch_wrap": False,
                }
            )
        else:
            assert actual == "unset"
        subprocess.run(
            [
                "bash",
                "-c",
                assignment + "\n" + block + '\nbash "$LAUNCHER"',
            ],
            env=dict(
                env,
                PATH=f"{tmp_path}:{os.environ['PATH']}",
                MODEL_PATH=str(model),
                PORT="8123",
                SERVER_PID_FILE=str(tmp_path / "server.pid"),
                LAUNCHER=str(ROOT / "tools/adaptation-gate-cases/run_serve_graph.sh"),
            ),
            check=True,
        )
        argv = json.loads(capture.read_text())
        assert "--enforce-eager" not in argv
        assert "--no-async-scheduling" not in argv
        if mode == "graph-workaround":
            assert json.loads(argv[argv.index("--compilation-config") + 1]) == (
                json.loads(actual)
            )
        else:
            assert "--compilation-config" not in argv
    assert '"$group/config.txt"' in source
    assert '"$group/commit.txt"' in source


def test_iluvatar_is_discovered_and_uses_vendor_runners(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    result = subprocess.run(
        [sys.executable, str(ROOT / ".github/scripts/detect_platforms.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = next(
        line.removeprefix("platforms=")
        for line in result.stdout.splitlines()
        if line.startswith("platforms=")
    )
    platforms = json.loads(output)
    assert "iluvatar" in platforms
    assert "nvidia" in platforms
    assert "cuda" not in platforms
    config = yaml.safe_load((ROOT / ".github/configs/iluvatar.yml").read_text())
    assert config["runner_labels"] == ["iluvatar-cicd-vllm-plugin"]
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = ci["jobs"]["test-iluvatar"]
    assert "'iluvatar'" in job["if"]
    assert job["with"]["platform"] == "iluvatar"
    assert job["uses"] == "./.github/workflows/_platform_test.yml"
    platform = yaml.safe_load(
        (ROOT / ".github/workflows/_platform_test.yml").read_text()
    )
    unit_job = platform["jobs"]["unit"]
    assert (
        "needs.setup.outputs.config).runner_labels" in unit_job["with"]["runner_labels"]
    )
    unit = yaml.safe_load((ROOT / ".github/workflows/_unit_test.yml").read_text())
    assert (
        unit["jobs"]["unit-tests"]["runs-on"] == "${{ fromJson(inputs.runner_labels) }}"
    )
    gate = yaml.safe_load((ROOT / ".github/workflows/_iluvatar_gate.yml").read_text())
    assert (
        gate["jobs"]["gates"]["runs-on"]
        == "${{ fromJson(inputs.config).runner_labels }}"
    )


def test_gate_checks_out_requested_ref_and_records_commit():
    platform = yaml.safe_load(
        (ROOT / ".github/workflows/_platform_test.yml").read_text()
    )
    gate = platform["jobs"]["iluvatar-gates"]
    assert gate["with"]["branch"] == "${{ needs.setup.outputs.revision }}"
    setup = platform["jobs"]["setup"]
    assert setup["outputs"]["revision"] == "${{ steps.revision.outputs.sha }}"
    freeze = next(s for s in setup["steps"] if s.get("id") == "revision")
    result = subprocess.run(
        ["bash", "-c", freeze["run"]],
        cwd=ROOT,
        env=dict(os.environ, GITHUB_OUTPUT=os.devnull),
        capture_output=True,
    )
    assert result.returncode == 0
    for name in ("functional", "unit", "e2e", "benchmark", "iluvatar-gates"):
        assert platform["jobs"][name]["with"]["branch"] == (
            "${{ needs.setup.outputs.revision }}"
        )
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/_iluvatar_gate.yml").read_text()
    )
    steps = workflow["jobs"]["gates"]["steps"]
    checkout = next(s for s in steps if s.get("uses") == "actions/checkout@v4")
    assert checkout["with"]["ref"] == "${{ inputs.branch }}"
    assert any("git rev-parse HEAD" in s.get("run", "") for s in steps)

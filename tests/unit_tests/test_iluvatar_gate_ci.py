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


def test_ci_graph_config_default_and_override():
    import json

    source = (ROOT / ".github/scripts/iluvatar/run_gates.sh").read_text()
    start = source.index("if [[ -z ${GRAPH_COMPILATION_CONFIG:-} ]]")
    end = source.index("\nfi", start) + len("\nfi")
    block = source[start:end] + '\nprintf "%s" "$GRAPH_COMPILATION_CONFIG"'
    for override in (None, '{"mode":3,"cudagraph_mode":"FULL"}'):
        env = dict(os.environ)
        env.pop("GRAPH_COMPILATION_CONFIG", None)
        if override is not None:
            env["GRAPH_COMPILATION_CONFIG"] = override
        result = subprocess.run(
            ["bash", "-c", f"set -e\n{block}"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        actual = json.loads(result.stdout)
        if override is not None:
            assert actual == json.loads(override)
        else:
            assert actual == {
                "mode": 0,
                "cudagraph_mode": "FULL",
                "custom_ops": ["all"],
                "ir_enable_torch_wrap": False,
            }


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
    assert "cuda" in platforms
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

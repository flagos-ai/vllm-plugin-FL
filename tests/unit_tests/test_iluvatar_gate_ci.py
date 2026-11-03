"""CI wiring and fail-closed checks, not hardware acceptance evidence."""

import os
import subprocess
from pathlib import Path

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


def test_ci_graph_config_default_and_override():
    import json

    source = (ROOT / ".github/scripts/iluvatar/run_gates.sh").read_text()
    start = source.index("if [[ -z ${GRAPH_COMPILATION_CONFIG:-} ]]")
    end = source.index("export GRAPH_COMPILATION_CONFIG", start)
    block = source[start:end] + 'printf "%s" "$GRAPH_COMPILATION_CONFIG"'
    for override in (None, '{"mode":3,"cudagraph_mode":"FULL"}'):
        env = dict(os.environ)
        env.pop("GRAPH_COMPILATION_CONFIG", None)
        if override is not None:
            env["GRAPH_COMPILATION_CONFIG"] = override
        result = subprocess.run(
            ["bash", "-c", block], env=env, capture_output=True, text=True, check=True
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

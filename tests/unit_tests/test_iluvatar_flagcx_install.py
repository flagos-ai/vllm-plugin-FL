"""FlagCX provisioning contracts; model acceptance remains a hardware gate."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / ".github/scripts/iluvatar/install_flagcx.sh"


def test_workflow_provisions_flagcx_before_setup_and_gates():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/_iluvatar_gate.yml").read_text()
    )
    commands = [step.get("run") for step in workflow["jobs"]["gates"]["steps"]]
    assert (
        commands.index("bash .github/scripts/iluvatar/install_flagcx.sh")
        < commands.index("bash .github/scripts/iluvatar/setup.sh")
        < commands.index("bash .github/scripts/iluvatar/run_gates.sh")
    )
    subprocess.run(["bash", "-n", str(INSTALLER)], check=True)


@pytest.mark.parametrize("missing", ["library", "wrapper", "invalid_library"])
def test_explicit_invalid_flagcx_fails_without_export(tmp_path, missing):
    root = tmp_path / "FlagCX with spaces"
    library = root / "build/lib/libflagcx.so"
    wrapper = root / "plugin/interservice/flagcx_wrapper.py"
    root.mkdir()
    if missing != "library":
        library.parent.mkdir(parents=True)
        library.write_text("not a shared library")
    if missing != "wrapper":
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("")
    github_env = tmp_path / "github-env"
    env = dict(os.environ, FLAGCX_PATH=str(root), GITHUB_ENV=str(github_env))
    result = subprocess.run(
        ["bash", str(INSTALLER)], env=env, text=True, capture_output=True
    )
    assert result.returncode != 0
    assert not github_env.exists()
    if missing == "invalid_library":
        assert "OSError" in result.stderr
    else:
        assert "Missing FlagCX file:" in result.stderr


def test_default_build_is_pinned_to_corex():
    source = INSTALLER.read_text()
    assert 'if [[ -z "${FLAGCX_PATH:-}" ]]' in source
    assert "revision=6143384797219733aee89bd3118b2cb184ad4959" in source
    assert 'fetch --depth 1 origin "$revision"' in source
    assert "submodule update --init --depth 1 third-party/json" in source
    assert 'make -C "$FLAGCX_PATH" -j4 USE_ILUVATAR_COREX=1' in source
    assert source.index("ctypes.CDLL") < source.index(' >> "$GITHUB_ENV"')

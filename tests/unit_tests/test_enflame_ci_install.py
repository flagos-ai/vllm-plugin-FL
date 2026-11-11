"""Exercise the installer control flow without changing the test host's SDK."""

import os
import runpy
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/enflame/install_vllm.sh"
COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"


def test_runtime_gate_does_not_require_new_vllm():
    versions = {"torch": "2.11.0+cpu", "torch-gcu": "2.11.0+3.8.20260713"}
    with patch("importlib.metadata.version", side_effect=versions.__getitem__):
        runpy.run_path(str(ROOT / ".github/scripts/enflame/verify_stack.py"))[
            "verify_stack"
        ](True)


def run_installer(tmp_path, fail="", wrong_commit=False):
    log = tmp_path / "commands"
    python = tmp_path / "python"
    python.write_text("""#!/bin/bash
printf 'python %s\\n' "$*" >> "$COMMAND_LOG"
if [[ -n "$FAIL_ON" && "$*" == *"$FAIL_ON"* ]]; then exit 23; fi
""")
    git = tmp_path / "git"
    git.write_text("""#!/bin/bash
printf 'git %s\\n' "$*" >> "$COMMAND_LOG"
if [[ "$*" == *"rev-parse HEAD"* ]]; then printf '%s\\n' "$SOURCE_COMMIT"; fi
""")
    for p in (python, git):
        p.chmod(0o755)
    source = tmp_path / "source"
    source.mkdir()
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "COMMAND_LOG": str(log),
        "FAIL_ON": fail,
        "ENFLAME_VLLM_SOURCE_DIR": str(source),
        "SOURCE_COMMIT": "bad" if wrong_commit else COMMIT,
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True
    )
    return result, log.read_text().splitlines()


def test_install_precedes_strict_gate_and_preserves_vendor_torch(tmp_path):
    result, commands = run_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--runtime-only" in commands[0]
    build = next(
        i for i, c in enumerate(commands) if "--no-build-isolation --no-deps ." in c
    )
    assert any("verify_stack.py" in c for c in commands[build + 1 :])
    assert any("--require-hashes" in c for c in commands)
    assert any("--force-reinstall --no-deps" in c for c in commands)
    assert not any("uninstall" in c and ("torch" in c or "vllm" in c) for c in commands)


@pytest.mark.parametrize(
    "failure",
    ["--runtime-only", "pip download", "pip install -r", "--no-build-isolation"],
)
def test_installer_stops_on_failure(tmp_path, failure):
    result, commands = run_installer(tmp_path, failure)
    assert result.returncode == 23
    assert failure in commands[-1]


def test_wrong_source_commit_fails_before_install(tmp_path):
    result, commands = run_installer(tmp_path, wrong_commit=True)
    assert result.returncode != 0
    assert not any("pip install" in c for c in commands)


def test_setup_installs_before_full_stack_gate():
    setup = (ROOT / ".github/scripts/enflame/setup.sh").read_text()
    assert setup.index("bash .github/scripts/enflame/install_vllm.sh") < setup.index(
        "python .github/scripts/enflame/verify_stack.py"
    )


def test_installer_upgrades_legacy_cloudpickle(tmp_path):
    result, commands = run_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    assert any("cloudpickle>=3.1.0" in command for command in commands)

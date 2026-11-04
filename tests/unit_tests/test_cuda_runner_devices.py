# Copyright (c) 2026 BAAI. All rights reserved.
"""Run only the CI device-configuration helper in real, CPU-only Bash shells."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ".github/scripts/nvidia/runner_devices.sh"
_CONTROLLED = (
    "GITHUB_ACTIONS",
    "RUNNER_NAME",
    "GITHUB_ENV",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
    "BASH_ENV",
    "ENV",
)
_SNAPSHOT = r"""
for name in CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER NVIDIA_VISIBLE_DEVICES GITHUB_ENV; do
    if [[ -v "$name" ]]; then
        printf '__CUDA_RUNNER_@TAG@\t%s\tset\t%s\n' "$name" "${!name}"
    else
        printf '__CUDA_RUNNER_@TAG@\t%s\tunset\t\n' "$name"
    fi
done
"""
_CPU_ONLY = r"""
forbidden_device_or_install_command() { printf 'Unexpected hardware/install command\n' >&2; return 99; }
nvidia-smi() { forbidden_device_or_install_command; }
python() { forbidden_device_or_install_command; }
python3() { forbidden_device_or_install_command; }
pip() { forbidden_device_or_install_command; }
pip3() { forbidden_device_or_install_command; }
uv() { forbidden_device_or_install_command; }
mount() { forbidden_device_or_install_command; }
"""


@pytest.fixture(scope="module")
def bash():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = (
        str(git_bash)
        if os.name == "nt" and git_bash.is_file()
        else shutil.which("bash")
    )
    assert executable is not None, "The device helper contracts require Bash"
    return executable


def _environment(**values):
    env = os.environ.copy()
    for name in _CONTROLLED:
        env.pop(name, None)
    env["NVIDIA_VISIBLE_DEVICES"] = "GPU-untouched-container-visibility"
    env.update(
        {name: str(value) for name, value in values.items() if value is not None}
    )
    return env


def _snapshot(output, tag="CURRENT"):
    prefix = f"__CUDA_RUNNER_{tag}\t"
    values = {}
    for line in output.splitlines():
        if line.startswith(prefix):
            name, defined, value = line[len(prefix) :].split("\t", 2)
            values[name] = (defined, value)
    assert len(values) == 4, output
    return values


def _run(bash, env, *, configure=True, setup="", child=True):
    script = "\n".join(
        (
            "set -euo pipefail",
            _CPU_ONLY,
            setup,
            f"source {shlex.quote(HELPER)}",
            "configure_cuda_runner_devices" if configure else "",
            _SNAPSHOT.replace("@TAG@", "CURRENT"),
        )
    )
    if child:
        child_script = _SNAPSHOT.replace("@TAG@", "CHILD")
        script += '\n"$BASH" --noprofile --norc -c ' + shlex.quote(child_script)
    result = subprocess.run(
        [bash, "--noprofile", "--norc", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    child_state = _snapshot(result.stdout, "CHILD") if child else None
    return _snapshot(result.stdout), child_state


def _assert_visibility(state, mask, order):
    assert state["CUDA_VISIBLE_DEVICES"] == (
        ("unset", "") if mask is None else ("set", mask)
    )
    assert state["CUDA_DEVICE_ORDER"] == (
        ("unset", "") if order is None else ("set", order)
    )
    assert state["NVIDIA_VISIBLE_DEVICES"] == (
        "set",
        "GPU-untouched-container-visibility",
    )


def _github_updates(path):
    return dict(
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
    )


def test_sourcing_helper_has_no_device_configuration_or_environment_file_effect(
    bash, tmp_path
):
    github_env = tmp_path / "github.env"
    github_env.write_text("UNRELATED=keep\n", encoding="utf-8")
    state, child = _run(
        bash,
        _environment(
            GITHUB_ACTIONS="true",
            RUNNER_NAME="dsw-h20-03",
            GITHUB_ENV=github_env.as_posix(),
        ),
        configure=False,
    )
    _assert_visibility(state, None, None)
    _assert_visibility(child, None, None)
    assert github_env.read_text(encoding="utf-8") == "UNRELATED=keep\n"


@pytest.mark.parametrize(
    "runner,expected,existing_order",
    [("dsw-h20-03", "0,1,2,3", "FASTEST_FIRST"), ("dsw-h20-47", "4,5,6,7", None)],
)
def test_known_h20_runner_defaults_are_exported_and_persisted(
    bash, tmp_path, runner, expected, existing_order
):
    github_env = tmp_path / "github.env"
    github_env.write_text("UNRELATED=keep\n", encoding="utf-8")
    state, child = _run(
        bash,
        _environment(
            GITHUB_ACTIONS="true",
            RUNNER_NAME=runner,
            GITHUB_ENV=github_env.as_posix(),
            CUDA_DEVICE_ORDER=existing_order,
        ),
    )
    for actual in (state, child):
        _assert_visibility(actual, expected, "PCI_BUS_ID")
    assert _github_updates(github_env) == {
        "UNRELATED": "keep",
        "CUDA_VISIBLE_DEVICES": expected,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }


@pytest.mark.parametrize(
    "mask,order",
    [
        ("2,0", ""),
        ("", "FASTEST_FIRST"),
        ("GPU-uuid-123", None),
    ],
)
def test_explicit_mask_including_empty_preserves_existing_or_unset_order(
    bash, tmp_path, mask, order
):
    github_env = tmp_path / "github.env"
    state, child = _run(
        bash,
        _environment(
            GITHUB_ACTIONS="true",
            RUNNER_NAME="dsw-h20-47",
            GITHUB_ENV=github_env.as_posix(),
            CUDA_VISIBLE_DEVICES=mask,
            CUDA_DEVICE_ORDER=order,
        ),
    )
    for actual in (state, child):
        _assert_visibility(actual, mask, order)
    expected = {"CUDA_VISIBLE_DEVICES": mask}
    if order is not None:
        expected["CUDA_DEVICE_ORDER"] = order
    assert _github_updates(github_env) == expected


@pytest.mark.parametrize(
    "actions,runner",
    [
        (None, "dsw-h20-03"),
        ("false", "dsw-h20-47"),
        ("true", "dsw-h20-030"),
    ],
)
def test_non_ci_or_non_exact_runner_leaves_visibility_and_env_file_untouched(
    bash, tmp_path, actions, runner
):
    github_env = tmp_path / "github.env"
    github_env.write_text("UNRELATED=keep\n", encoding="utf-8")
    state, child = _run(
        bash,
        _environment(
            GITHUB_ACTIONS=actions,
            RUNNER_NAME=runner,
            GITHUB_ENV=github_env.as_posix(),
            CUDA_DEVICE_ORDER="FASTEST_FIRST",
        ),
    )
    for actual in (state, child):
        _assert_visibility(actual, None, "FASTEST_FIRST")
    assert github_env.read_text(encoding="utf-8") == "UNRELATED=keep\n"


def test_github_env_values_propagate_to_next_step_without_reselecting_runner(
    bash, tmp_path
):
    github_env = tmp_path / "github.env"
    initial = _environment(
        GITHUB_ACTIONS="true",
        RUNNER_NAME="dsw-h20-03",
        GITHUB_ENV=github_env.as_posix(),
    )
    _run(bash, initial)
    # GitHub Actions adds this file's assignments to the next step's process
    # environment. Consume the real file in a separate Bash subprocess.
    following = _environment(GITHUB_ACTIONS="true", RUNNER_NAME="dsw-h20-47")
    following.update(_github_updates(github_env))
    state, child = _run(bash, following)
    for actual in (state, child):
        _assert_visibility(actual, "0,1,2,3", "PCI_BUS_ID")


@pytest.mark.parametrize("github_env", [None, ""])
def test_shell_local_settings_are_exported_without_a_github_env_file(bash, github_env):
    state, child = _run(
        bash,
        _environment(
            GITHUB_ACTIONS="true", RUNNER_NAME="dsw-h20-03", GITHUB_ENV=github_env
        ),
        setup="CUDA_VISIBLE_DEVICES='GPU-shell-local'; CUDA_DEVICE_ORDER='FASTEST_FIRST'",
    )
    for actual in (state, child):
        _assert_visibility(actual, "GPU-shell-local", "FASTEST_FIRST")
        assert actual["GITHUB_ENV"] == (
            ("unset", "") if github_env is None else ("set", "")
        )

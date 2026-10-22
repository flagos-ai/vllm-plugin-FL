# Copyright 2026 FlagOS Contributors
"""CI installs the pinned source without replacing platform runtime packages."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def installer():
    script = (
        Path(__file__).resolve().parents[2] / ".github/scripts/install_flaggems_vllm.py"
    )
    spec = importlib.util.spec_from_file_location("ci_fused_install", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installs_pinned_source_without_runtime_dependencies(installer):
    with (
        patch.object(installer, "package_version", return_value="1.0"),
        patch.object(installer, "source_is_installed", side_effect=[False, True]),
        patch.object(installer, "pip_install") as install,
    ):
        installer.main()
    install.assert_called_once_with(
        "--no-deps",
        "--no-build-isolation",
        "--force-reinstall",
        f"flaggems-vllm @ git+{installer.REPOSITORY}@{installer.COMMIT}",
    )


def test_already_installed_source_is_not_reinstalled(installer):
    with (
        patch.object(installer, "package_version", return_value="1.0"),
        patch.object(installer, "source_is_installed", return_value=True),
        patch.object(installer, "pip_install") as install,
    ):
        installer.main()
    install.assert_not_called()


def test_failed_install_propagates(installer):
    error = subprocess.CalledProcessError(1, ["pip", "install"])
    with (
        patch.object(installer, "package_version", return_value="1.0"),
        patch.object(installer, "source_is_installed", return_value=False),
        patch.object(installer, "pip_install", side_effect=error),
        pytest.raises(subprocess.CalledProcessError) as caught,
    ):
        installer.main()
    assert caught.value is error


def test_rejects_wrong_source_after_install(installer):
    with (
        patch.object(installer, "package_version", return_value="1.0"),
        patch.object(installer, "source_is_installed", return_value=False),
        patch.object(installer, "pip_install"),
        pytest.raises(RuntimeError, match="must come from upstream commit"),
    ):
        installer.main()


def test_runtime_dependency_changes_are_rejected(installer):
    changed = False

    def install(*args):
        nonlocal changed
        changed = True

    with (
        patch.object(
            installer, "package_version", side_effect=lambda _: "2" if changed else "1"
        ),
        patch.object(installer, "source_is_installed", side_effect=[False, True]),
        patch.object(installer, "pip_install", side_effect=install),
        pytest.raises(RuntimeError, match="runtime dependencies changed"),
    ):
        installer.main()


@pytest.mark.parametrize(
    "metadata", [None, "not JSON", "[]", "null", "{}", '{"vcs_info": null}']
)
def test_invalid_metadata_is_not_an_installed_source(installer, metadata):
    with patch.object(
        installer,
        "distribution",
        return_value=SimpleNamespace(read_text=lambda _: metadata),
    ):
        assert not installer.source_is_installed()


def test_source_requires_matching_repository_and_commit(installer):
    source = {"url": installer.REPOSITORY, "vcs_info": {"commit_id": installer.COMMIT}}
    with patch.object(
        installer,
        "distribution",
        return_value=SimpleNamespace(read_text=lambda _: json.dumps(source)),
    ):
        assert installer.source_is_installed()
        source["vcs_info"]["commit_id"] = "bad"
        assert not installer.source_is_installed()

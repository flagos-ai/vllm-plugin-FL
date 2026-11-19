# Copyright (c) 2026 BAAI. All rights reserved.
"""Installer source selection without installing packages or using a device."""

import importlib.metadata as metadata
import importlib.util
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[2] / ".github/scripts/install_flaggems_vllm.py"


@pytest.fixture
def installer():
    spec = importlib.util.spec_from_file_location("fl_operator_installer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("vendor", ["nvidia", "thead", "THEAD"])
def test_cli_selects_source_with_empty_vllm_vendor(monkeypatch, installer, vendor):
    monkeypatch.setenv("GEMS_VENDOR", vendor)
    monkeypatch.setenv("VLLM_VENDOR", "")
    monkeypatch.setattr(metadata, "version", lambda name: "1.0")
    installed, calls = {}, []

    def distribution(name):
        if not installed:
            raise metadata.PackageNotFoundError(name)
        return SimpleNamespace(read_text=lambda filename: json.dumps(installed))

    def pip(command, *, env, check):
        calls.append(command)
        if vendor.lower() == "thead":
            installed.update(
                url=installer.THEAD_ARCHIVE,
                archive_info={"hashes": {"sha256": installer.THEAD_ARCHIVE_SHA256}},
            )
        else:
            installed.update(
                url=installer.REPOSITORY, vcs_info={"commit_id": installer.COMMIT}
            )

    monkeypatch.setattr(metadata, "distribution", distribution)
    monkeypatch.setattr(subprocess, "run", pip)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    expected = (
        f"{installer.THEAD_ARCHIVE}#sha256={installer.THEAD_ARCHIVE_SHA256}"
        if vendor.lower() == "thead"
        else f"git+{installer.REPOSITORY}@{installer.COMMIT}"
    )
    assert len(calls) == 1
    assert calls[0][-1] == f"flaggems-vllm @ {expected}"
    assert "--no-deps" in calls[0] and "--no-build-isolation" in calls[0]


@pytest.mark.parametrize("archive", [False, True])
def test_explicit_caller_parameters_override_vendor_defaults(
    monkeypatch, installer, archive
):
    monkeypatch.setenv("GEMS_VENDOR", "thead")
    monkeypatch.setattr(installer, "package_version", lambda name: "1.0")
    commit = "another-platform-fixed-commit"
    url, digest = "https://example.invalid/source.tar.gz", "0123456789abcdef"
    source = (
        {"url": url, "archive_info": {"hashes": {"sha256": digest}}}
        if archive
        else {"url": installer.REPOSITORY, "vcs_info": {"commit_id": commit}}
    )
    monkeypatch.setattr(
        installer,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda filename: json.dumps(source)),
    )
    monkeypatch.setattr(
        installer,
        "pip_install",
        lambda *args: pytest.fail("matching source reinstalled"),
    )
    installer.main(
        commit=commit,
        source_archive=url if archive else None,
        archive_sha256=digest if archive else None,
    )


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"url": "https://example.invalid/wrong"},
        {"url": "archive", "archive_info": {"hashes": {"sha256": "wrong"}}},
    ],
)
def test_archive_requires_both_url_and_digest(monkeypatch, installer, source):
    monkeypatch.setattr(
        installer,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda filename: json.dumps(source)),
    )
    assert not installer.release_is_installed("commit", "archive", "expected")

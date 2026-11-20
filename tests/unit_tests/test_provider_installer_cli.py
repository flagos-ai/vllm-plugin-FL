# Copyright 2026 FlagOS Contributors
"""Exercise the same script entry point used by vendor setup scripts."""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/install_flaggems_vllm.py"
)


@pytest.mark.parametrize("vendor", ["", "kunlunxin"])
@pytest.mark.parametrize("args", [[], ["--source-archive"], ["--platform", "enflame"]])
def test_installer_script_entrypoint(monkeypatch, vendor, args):
    import importlib.metadata as metadata
    import subprocess

    monkeypatch.setenv("CI_FLAGGEMS_VLLM_VENDOR", vendor)
    constants = runpy.run_path(str(SCRIPT))
    archive = "--source-archive" in args
    commit = constants["ENFLAME_COMMIT"] if "enflame" in args else constants["COMMIT"]
    source = (
        {
            "url": constants["SOURCE_ARCHIVE"],
            "archive_info": {"hashes": {"sha256": constants["SOURCE_ARCHIVE_SHA256"]}},
        }
        if archive
        else {"url": constants["REPOSITORY"], "vcs_info": {"commit_id": commit}}
    )
    distribution = Mock()
    distribution.read_text.return_value = json.dumps(source)
    monkeypatch.setattr(metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(metadata, "version", lambda name: "test-version")
    install = Mock()
    monkeypatch.setattr(subprocess, "run", install)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    install.assert_not_called()


def test_source_archive_installs_hash_pinned_artifact(monkeypatch):
    namespace = runpy.run_path(str(SCRIPT))
    main = namespace["main"]
    calls = []
    monkeypatch.setitem(main.__globals__, "package_version", lambda name: "test")
    monkeypatch.setitem(
        main.__globals__, "release_is_installed", lambda *args: bool(calls)
    )
    monkeypatch.setitem(
        main.__globals__, "pip_install", lambda *args: calls.append(args)
    )
    main(["--source-archive"])
    assert calls == [
        (
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            f"flaggems-vllm @ {namespace['SOURCE_ARCHIVE']}#sha256={namespace['SOURCE_ARCHIVE_SHA256']}",
        )
    ]


def test_enflame_archive_cannot_use_shared_pin():
    namespace = runpy.run_path(str(SCRIPT))
    with pytest.raises(SystemExit) as exc:
        namespace["main"](["--platform", "enflame", "--source-archive"])
    assert exc.value.code == 2


def test_archive_requires_digest(monkeypatch):
    namespace = runpy.run_path(str(SCRIPT))
    main = namespace["main"]
    monkeypatch.setitem(main.__globals__, "SOURCE_ARCHIVE_SHA256", None)
    with pytest.raises(SystemExit) as exc:
        main(["--source-archive"])
    assert exc.value.code == 2

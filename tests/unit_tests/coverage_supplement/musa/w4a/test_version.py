# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for vllm_fl.version fallbacks, without git or network."""

import sys
from types import ModuleType

import pytest

import vllm_fl
import vllm_fl.version as version


def _raise(exc):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


def _set_generated(monkeypatch, module):
    """Control what ``from . import _version`` resolves to.

    ``from . import _version`` first consults the ``vllm_fl`` package
    attribute, so overriding ``sys.modules`` alone is not enough when the
    real generated module was already imported.
    """
    if module is None:
        monkeypatch.delattr(vllm_fl, "_version", raising=False)
        monkeypatch.setitem(sys.modules, "vllm_fl._version", None)
    else:
        monkeypatch.setattr(vllm_fl, "_version", module, raising=False)
        monkeypatch.setitem(sys.modules, "vllm_fl._version", module)


def test_public_exports_shape():
    assert isinstance(version.__version__, str)
    assert isinstance(version.git_version, str)
    assert set(version.git_info) == {"id", "date"}
    assert version.git_info["id"] == version.git_version


def test_pkg_version_prefers_metadata(monkeypatch):
    monkeypatch.setattr(version.metadata, "version", lambda name: "1.2.3")
    assert version._pkg_version() == "1.2.3"


def test_pkg_version_falls_back_to_generated_file(monkeypatch):
    monkeypatch.setattr(
        version.metadata, "version", _raise(version.metadata.PackageNotFoundError)
    )
    fake = ModuleType("vllm_fl._version")
    fake.__version__ = "9.9.9"
    _set_generated(monkeypatch, fake)
    assert version._pkg_version() == "9.9.9"


def test_pkg_version_unknown_when_all_fail(monkeypatch):
    monkeypatch.setattr(
        version.metadata, "version", _raise(version.metadata.PackageNotFoundError)
    )
    _set_generated(monkeypatch, None)
    assert version._pkg_version() == "0.0.0+unknown"


def test_git_head_from_repo_success_and_empty(monkeypatch):
    monkeypatch.setattr(
        version.subprocess, "check_output", lambda *a, **k: "abc123\n"
    )
    assert version._git_head_from_repo() == "abc123"
    monkeypatch.setattr(version.subprocess, "check_output", lambda *a, **k: "  \n")
    assert version._git_head_from_repo() is None


def test_git_head_from_repo_failure(monkeypatch):
    monkeypatch.setattr(
        version.subprocess, "check_output", _raise(OSError("no git"))
    )
    assert version._git_head_from_repo() is None


def test_git_commit_date_from_repo_success(monkeypatch):
    monkeypatch.setattr(
        version.subprocess,
        "check_output",
        lambda *a, **k: "2026-01-02T03:04:05+00:00\n",
    )
    assert version._git_commit_date_from_repo() == "2026-01-02"
    monkeypatch.setattr(version.subprocess, "check_output", lambda *a, **k: "")
    assert version._git_commit_date_from_repo() is None


def test_git_commit_date_from_repo_failure(monkeypatch):
    monkeypatch.setattr(
        version.subprocess, "check_output", _raise(RuntimeError("boom"))
    )
    assert version._git_commit_date_from_repo() is None


@pytest.mark.parametrize(
    "git_version,git_date,expected",
    [
        ("deadbeef", "2026-02-03", ("deadbeef", "2026-02-03")),
        ("", "Unknown", (None, None)),
        ("Unknown", "", (None, None)),
        (None, None, (None, None)),
        (123, 456, (None, None)),
    ],
)
def test_load_scm_normalizes(monkeypatch, git_version, git_date, expected):
    fake = ModuleType("vllm_fl._version")
    fake.git_version = git_version
    fake.git_date = git_date
    _set_generated(monkeypatch, fake)
    assert version._load_scm() == expected


def test_load_scm_missing_generated_file(monkeypatch):
    _set_generated(monkeypatch, None)
    assert version._load_scm() == (None, None)

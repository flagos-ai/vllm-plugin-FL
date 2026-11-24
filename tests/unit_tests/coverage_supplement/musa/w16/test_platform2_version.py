# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual CPU contracts for vllm_fl.version fallbacks (no git deps asserted).

The installed package ships a real ``vllm_fl._version`` module, so the loader
``from . import _version`` resolves the *package attribute*, not
``sys.modules``. Patch that attribute directly and only assert the fallback
behaviour derived from the injected double -- never the real version string.
"""

from types import ModuleType

import vllm_fl
import vllm_fl.version as version


def _raise(exc):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


def _fake_scm(monkeypatch, **attrs):
    fake = ModuleType("vllm_fl._version")
    for key, value in attrs.items():
        setattr(fake, key, value)
    monkeypatch.setattr(vllm_fl, "_version", fake, raising=False)
    return fake


def test_pkg_version_falls_back_on_unexpected_metadata_error(monkeypatch):
    monkeypatch.setattr(version.metadata, "version", _raise(RuntimeError("broken")))
    _fake_scm(monkeypatch, __version__="0.0.0+probe")
    assert version._pkg_version() == "0.0.0+probe"


def test_load_scm_partial_and_invalid_values(monkeypatch):
    _fake_scm(monkeypatch, git_version="deadbeef", git_date="Unknown")
    assert version._load_scm() == ("deadbeef", None)

    _fake_scm(monkeypatch, git_version="", git_date="2026-02-03")
    assert version._load_scm() == (None, "2026-02-03")


def test_load_scm_missing_attributes(monkeypatch):
    _fake_scm(monkeypatch)
    assert version._load_scm() == (None, None)


def test_git_helpers_blank_output_returns_none(monkeypatch):
    monkeypatch.setattr(version.subprocess, "check_output", lambda *a, **k: "   \n")
    assert version._git_head_from_repo() is None
    monkeypatch.setattr(version.subprocess, "check_output", lambda *a, **k: "\n")
    assert version._git_commit_date_from_repo() is None

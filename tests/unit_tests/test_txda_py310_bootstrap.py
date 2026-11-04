# Copyright (c) 2026 BAAI. All rights reserved.
"""Execute the opt-in bootstrap without requiring a local Python 3.10 SDK."""

import datetime
import sys
from pathlib import Path
from types import ModuleType

import pytest
import typing_extensions

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "tools/tsingmicro_py310/sitecustomize.py"
TYPING_NAMES = ("Self", "Never", "NotRequired", "Required", "assert_never")


@pytest.fixture
def bootstrap(monkeypatch):
    fake_sys = ModuleType("sys")
    fake_sys.version_info = (3, 10, 20, "final", 0)
    fake_typing = ModuleType("typing")
    fake_datetime = ModuleType("datetime")
    fake_datetime.timezone = datetime.timezone
    monkeypatch.setenv("FL_TXDA_PY310_COMPAT", "1")
    monkeypatch.setenv("GEMS_VENDOR", "tsingmicro")
    source = compile(BOOTSTRAP.read_text(encoding="utf-8"), str(BOOTSTRAP), "exec")

    def run():
        namespace = {"__name__": "sitecustomize", "__file__": str(BOOTSTRAP)}
        # Keep stub stdlib modules confined to source execution so pytest's
        # own typing-dependent fixture machinery continues using real typing.
        with monkeypatch.context() as imports:
            imports.setitem(sys.modules, "sys", fake_sys)
            imports.setitem(sys.modules, "typing", fake_typing)
            imports.setitem(sys.modules, "datetime", fake_datetime)
            exec(source, namespace)
        return namespace

    return fake_sys, fake_typing, fake_datetime, run


def test_startup_exec_populates_missing_typing_and_utc(bootstrap):
    _, typing, dt, run = bootstrap
    run()
    for name in TYPING_NAMES:
        assert getattr(typing, name) is getattr(typing_extensions, name)
    assert dt.UTC is datetime.timezone.utc
    assert typing.NotRequired[int] == typing_extensions.NotRequired[int]
    with pytest.raises(AssertionError):
        typing.assert_never("unexpected value")


@pytest.mark.parametrize("compat", [None, "", "0", "true", "yes"])
def test_disabled_environment_does_not_touch_modules(bootstrap, monkeypatch, compat):
    _, typing, dt, run = bootstrap
    if compat is None:
        monkeypatch.delenv("FL_TXDA_PY310_COMPAT", raising=False)
    else:
        monkeypatch.setenv("FL_TXDA_PY310_COMPAT", compat)
    # Disabled paths must not even need the backport dependency.
    monkeypatch.setitem(sys.modules, "typing_extensions", None)
    before = vars(typing).copy(), vars(dt).copy()
    run()
    assert (vars(typing), vars(dt)) == before


@pytest.mark.parametrize("vendor", [None, "", "txda", "nvidia", "mthreads"])
def test_other_vendor_does_not_touch_modules(bootstrap, monkeypatch, vendor):
    _, typing, dt, run = bootstrap
    if vendor is None:
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
    else:
        monkeypatch.setenv("GEMS_VENDOR", vendor)
    monkeypatch.setitem(sys.modules, "typing_extensions", None)
    before = vars(typing).copy(), vars(dt).copy()
    run()
    assert (vars(typing), vars(dt)) == before


@pytest.mark.parametrize("version", [(3, 9, 25), (3, 11, 14), (3, 12, 12), (3, 13, 9)])
def test_other_python_versions_do_not_touch_modules(bootstrap, monkeypatch, version):
    fake_sys, typing, dt, run = bootstrap
    fake_sys.version_info = version
    monkeypatch.setitem(sys.modules, "typing_extensions", None)
    before = vars(typing).copy(), vars(dt).copy()
    run()
    assert (vars(typing), vars(dt)) == before


def test_existing_symbols_are_preserved_and_repeated_startup_is_idempotent(bootstrap):
    _, typing, dt, run = bootstrap
    originals = {name: object() for name in TYPING_NAMES}
    for name, value in originals.items():
        setattr(typing, name, value)
    dt.UTC = object()
    original_utc = dt.UTC

    run()
    run()

    for name, value in originals.items():
        assert getattr(typing, name) is value
    assert dt.UTC is original_utc


def test_only_missing_symbols_are_backfilled(bootstrap):
    _, typing, dt, run = bootstrap
    typing.Self = object()
    original_self = typing.Self
    run()
    assert typing.Self is original_self
    for name in TYPING_NAMES[1:]:
        assert getattr(typing, name) is getattr(typing_extensions, name)
    assert dt.UTC is datetime.timezone.utc

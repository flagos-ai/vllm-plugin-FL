# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU contracts for repacking the pinned Kunlunxin vLLM wheel."""

import importlib.util
import sys
from datetime import timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "docker/kunlunxin/python310_compat.py"
_SPEC = importlib.util.spec_from_file_location("kunlunxin_python310_compat", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_COMPAT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMPAT)
_FILENAME = "vllm/example.py"
_OUTLINES_FILENAME = "vllm/v1/structured_output/backend_outlines.py"


def test_mixed_typing_import_preserves_aliases_and_indentation():
    source = (
        "if True:\n"
        "    from typing import (\n"
        "        Any as Value, Required as Field, Self, TypedDict,\n"
        "    )\n"
    )

    assert _COMPAT.patch_python_source(source, _FILENAME) == (
        "if True:\n"
        "    from typing import Any as Value\n"
        "    from typing_extensions import Required as Field, Self, TypedDict\n"
    )


def test_datetime_utc_backports_import_and_attribute():
    source = (
        "import datetime\n"
        "from datetime import UTC as utc, timedelta\n"
        "values = ('\N{GREEK SMALL LETTER BETA}', utc, datetime.UTC, timedelta(days=1))\n"
    )
    namespace = {}

    exec(_COMPAT.patch_python_source(source, _FILENAME), namespace)

    _, imported_utc, attribute_utc, delta = namespace["values"]
    assert imported_utc is attribute_utc is timezone.utc
    assert delta.days == 1


def test_outlines_regex_internals_use_python310_modules():
    source = "from re import _constants, _parser  # type: ignore[attr-defined]\n"
    assert _COMPAT.patch_python_source(source, _OUTLINES_FILENAME) == (
        "import sre_constants as _constants\n"
        "import sre_parse as _parser  # type: ignore[attr-defined]\n"
    )


@pytest.mark.parametrize(
    ("source", "filename"),
    [
        ("from re import _constants, _parser", _FILENAME),
        ("from re import _parser", _OUTLINES_FILENAME),
        ("from re import _constants, _parser as parser", _OUTLINES_FILENAME),
        ("if True:\n    from re import _constants, _parser", _OUTLINES_FILENAME),
    ],
)
def test_regex_backport_rejects_unrecognized_imports(source, filename):
    with pytest.raises(ValueError, match="Unexpected regex internals import"):
        _COMPAT.patch_python_source(source, filename)


def test_exception_notes_keep_future_imports_and_work_without_add_note(monkeypatch):
    source = (
        '"""Module docstring."""\n'
        "from __future__ import annotations\n\n"
        "def record(error):\n"
        '    error.add_note("first")\n'
        '    error.add_note("second")\n'
    )
    helper_name = "vllm._flagos_python310"
    helper = ModuleType(helper_name)
    exec(_COMPAT.EXCEPTION_HELPER_SOURCE, helper.__dict__)
    monkeypatch.setitem(sys.modules, helper_name, helper)
    namespace = {}
    error = SimpleNamespace()

    exec(_COMPAT.patch_python_source(source, _FILENAME), namespace)
    namespace["record"](error)

    assert namespace["__doc__"] == "Module docstring."
    assert error.__notes__ == ["first", "second"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("from asyncio import TaskGroup", "Unbackported Python API"),
        ("import tomllib", "Unbackported tomllib"),
        ("asyncio.timeout(1)", r"Unbackported Python API asyncio\.timeout"),
        ("ExceptionGroup('group', [])", "Unbackported ExceptionGroup"),
        ("error.add_note(note='detail')", "Unexpected add_note signature"),
        (
            "try:\n    pass\nexcept* ValueError:\n    pass",
            r"Python 3\.10 syntax audit failed",
        ),
    ],
)
def test_unsupported_source_reports_filename(source, message):
    with pytest.raises(ValueError, match=rf"{message}.*{_FILENAME}:"):
        _COMPAT.patch_python_source(source, _FILENAME)


def test_repack_rejects_wheel_with_unexpected_checksum(tmp_path):
    wheel = tmp_path / "vllm-0.32.0+flagos-py3-none-any.whl"
    wheel.write_bytes(b"not the pinned wheel")

    with pytest.raises(
        ValueError, match=r"Expected the published vLLM 0\.32\.0\+flagos"
    ):
        _COMPAT.repack(wheel, tmp_path / "out")

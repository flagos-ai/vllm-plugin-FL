# Copyright (c) 2026 BAAI. All rights reserved.
"""Contracts for the explicitly downstream Sunrise Python 3.10 source patch."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def compat():
    path = Path(__file__).resolve().parents[2] / "docker/sunrise/python310_compat.py"
    spec = importlib.util.spec_from_file_location("_sunrise_python310_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_typing_backports_preserve_real_typed_dict_semantics(compat):
    source = """from typing import Any, NotRequired, Required, Self, TypedDict, assert_never
class Request(TypedDict, total=False):
    prompt: Required[str]
    options: NotRequired[Any]
class Builder:
    def finish(self) -> Self:
        return self
"""
    patched, names = compat.patch_python_source(source, "vllm/request.py")
    tree = ast.parse(patched, feature_version=(3, 10))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            assert not {alias.name for alias in node.names} & compat.TYPING_BACKPORTS
    namespace = {}
    exec(compile(patched, "vllm/request.py", "exec"), namespace)
    assert namespace["Request"].__required_keys__ == frozenset({"prompt"})
    assert namespace["Request"].__optional_keys__ == frozenset({"options"})
    assert namespace["Builder"]().finish().__class__ is namespace["Builder"]
    assert {"TypedDict", "NotRequired", "Required", "Self", "assert_never"} == set(
        names
    )


def test_multiline_aliases_and_nested_imports_remain_executable(compat):
    source = """# Non-ASCII header (€) should survive rewriting.
def helper():
    from typing import (
        Self as Result,
        Any as Payload,
    )
    return Result, Payload
"""
    patched, _ = compat.patch_python_source(source, "vllm/example.py")
    namespace = {}
    exec(patched, namespace)
    import typing_extensions

    assert namespace["helper"]()[0] is typing_extensions.Self
    assert patched.startswith("# Non-ASCII header (€)")


def test_ordinary_python310_types_are_unchanged(compat):
    source = "from typing import Any, TypeAlias\nItems: TypeAlias = list[Any] | None\n"
    assert compat.patch_python_source(source, "vllm/example.py") == (source, [])


@pytest.mark.parametrize(
    "source",
    [
        "from enum import StrEnum\n",
        "import tomllib\n",
        "from asyncio import TaskGroup\n",
        "import asyncio\nasyncio.timeout(1)\n",
        "import typing\nvalue = typing.Self\n",
        "raise ExceptionGroup('failure', [])\n",
        "try:\n    pass\nexcept* Exception:\n    pass\n",
        "type Items = list[str]\n",
    ],
)
def test_unbackported_apis_or_syntax_fail_explicitly(compat, source):
    with pytest.raises(ValueError, match="vllm/example.py"):
        compat.patch_python_source(source, "vllm/example.py")


def make_small_tree(tmp_path, compat, monkeypatch, *, bad_source=False):
    root = tmp_path / "upstream"
    (root / "requirements").mkdir(parents=True)
    (root / "vllm").mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "vllm"\nrequires-python = ">=3.11,<3.15"\n'
    )
    (root / "setup.py").write_text("# fixture packaging\n")
    (root / "requirements/common.txt").write_text("typing_extensions >= 4.10\n")
    (root / "vllm/config.py").write_text("from typing import Self\n")
    if bad_source:
        (root / "vllm/unsupported.py").write_text("from enum import StrEnum\n")
    monkeypatch.setattr(
        compat,
        "SOURCE_HASHES",
        {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in compat.SOURCE_HASHES
        },
    )
    return root


def test_source_guard_rejects_other_release_before_writing(tmp_path, compat):
    (tmp_path / "pyproject.toml").write_text('requires-python = ">=3.11"\n')
    with pytest.raises(ValueError, match="unmodified official v0.32.0"):
        compat.apply_compatibility(tmp_path, tmp_path / "manifest.json")
    assert not (tmp_path / "manifest.json").exists()
    assert (tmp_path / "pyproject.toml").read_text() == 'requires-python = ">=3.11"\n'


def test_failed_audit_does_not_leave_partially_patched_source(
    tmp_path, compat, monkeypatch
):
    root = make_small_tree(tmp_path, compat, monkeypatch, bad_source=True)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Unbackported"):
        compat.apply_compatibility(root, tmp_path / "manifest.json")
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_labels_downstream_patch_and_keeps_dependency_floors(
    tmp_path, compat, monkeypatch
):
    root = make_small_tree(tmp_path, compat, monkeypatch)
    requirements = (root / "requirements/common.txt").read_bytes()
    manifest_path = tmp_path / "manifest.json"
    compat.apply_compatibility(root, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["upstream_version"] == "0.32.0"
    assert "not upstream supported" in manifest["support_status"]
    assert "not an import" in manifest["validation_limits"]
    assert manifest["changes"]["vllm/config.py"]["typing_backports"] == ["Self"]
    assert (root / "requirements/common.txt").read_bytes() == requirements
    assert 'requires-python = ">=3.10,<3.15"' in (root / "pyproject.toml").read_text()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        compat.apply_compatibility(root, manifest_path)


def test_exception_note_backport_preserves_existing_notes(compat):
    namespace = {}
    exec(compat.EXCEPTION_HELPER_SOURCE, namespace)

    class LegacyError(Exception):
        add_note = None

    error = LegacyError("worker failure")
    add_note = namespace["add_exception_note"]
    add_note(error, "original shutdown failure")
    add_note(error, "second shutdown failure")
    assert error.__notes__ == ["original shutdown failure", "second shutdown failure"]
    with pytest.raises(TypeError, match="str"):
        add_note(error, 7)


def test_exception_note_call_patch_keeps_future_import_first(compat, monkeypatch):
    import sys
    from types import ModuleType

    helper = ModuleType("vllm._sunrise_python310")
    exec(compat.EXCEPTION_HELPER_SOURCE, helper.__dict__)
    monkeypatch.setitem(sys.modules, "vllm._sunrise_python310", helper)
    source = '''"""Worker documentation."""
from __future__ import annotations
from typing import Self
class LegacyError(Exception):
    add_note = None
error = LegacyError("worker failed")
error.add_note("shutdown failed")
'''
    patched, backports = compat.patch_python_source(source, "vllm/worker.py")
    namespace = {}
    exec(patched, namespace)
    assert namespace["error"].__notes__ == ["shutdown failed"]
    assert "BaseException.add_note" in backports
    assert ast.parse(patched).body[1].module == "__future__"


def test_python310_typed_dict_attribute_without_new_qualifiers_is_valid(compat):
    source = "import typing\nclass Modalities(typing.TypedDict, total=False):\n    image: str\n"
    assert compat.patch_python_source(source, "vllm/phi4mm.py") == (source, [])


def test_datetime_utc_uses_python310_equivalent(compat):
    import datetime

    source = (
        "import datetime\nvalue = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)\n"
    )
    patched, backports = compat.patch_python_source(source, "vllm/usage.py")
    namespace = {}
    exec(patched, namespace)
    assert namespace["value"].tzinfo is datetime.timezone.utc
    assert backports == ["datetime.UTC"]

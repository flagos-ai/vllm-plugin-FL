# Copyright (c) 2026 BAAI. All rights reserved.
"""Experimental Sunrise Python 3.10 source build of pinned vLLM 0.32.0.

This is a downstream compatibility patch, not upstream Python 3.10 support.
The Docker recipe verifies the upstream archive before invoking this script.
Only typing imports, exception-note backport calls and the Python floor change. Unsupported
syntax/APIs abort before any source is written; dependency resolution, actual
Python 3.10 imports and hardware acceptance remain separate required checks.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

UPSTREAM_VERSION = "0.32.0"
UPSTREAM_ARCHIVE_SHA256 = (
    "bf592c132f27bbb5f65cff0bbd2ee873eae722b79c8286934fd5b5cf88b43d61"
)
# Exact packaging files from the official v0.32.0 archive, before other patches.
SOURCE_HASHES = {
    "pyproject.toml": "eaa19cba70f13234be4acdc1eeb2126b937c6f0892fd1af29863db5a62bee95b",
    "setup.py": "5eb0e5cb1f5151276433446e6323d87dba141327567b1d326da89bfd919967f9",
    "requirements/common.txt": "366cb94fae6ea961da1e7d1c855f6316c3897b5ce9a839017db2bbebb668055a",
}
TYPING_BACKPORTS = frozenset(
    {
        "TypedDict",  # Python 3.10 stdlib TypedDict cannot interpret Required/NotRequired.
        "Self",
        "NotRequired",
        "Required",
        "LiteralString",
        "TypeVarTuple",
        "Unpack",
        "Never",
        "assert_never",
        "TypeAliasType",
        "override",
        "ReadOnly",
        "TypeIs",
        "NoDefault",
        "get_original_bases",
    }
)
# No ad-hoc replacements for structured concurrency or exception groups: these
# need semantic backports, and are absent from the pinned release's runtime.
UNSUPPORTED_IMPORTS = {
    "enum": {
        "StrEnum",
        "ReprEnum",
        "EnumCheck",
        "FlagBoundary",
        "verify",
        "member",
        "nonmember",
        "global_enum",
    },
    "asyncio": {"timeout", "timeout_at", "TaskGroup", "Runner"},
    "itertools": {"batched"},
    "collections.abc": {"Buffer"},
    "typing": {"get_protocol_members", "is_protocol", "evaluate_forward_ref"},
}
UNSUPPORTED_ATTRIBUTES = {
    "asyncio": {"timeout", "timeout_at", "TaskGroup", "Runner"},
    "enum": {"StrEnum"},
    "itertools": {"batched"},
    "math": {"sumprod"},
    "sys": {"exception"},
    "datetime": {"UTC"},
}


EXCEPTION_HELPER_PATH = "vllm/_sunrise_python310.py"
EXCEPTION_HELPER_SOURCE = """# Downstream Sunrise compatibility; upstream vLLM requires Python >= 3.11.
def add_exception_note(error, note):
    if not isinstance(note, str):
        raise TypeError("note must be a str")
    native = getattr(error, "add_note", None)
    if native is not None:
        return native(note)
    notes = getattr(error, "__notes__", None)
    if notes is None:
        error.__notes__ = [note]
    elif not isinstance(notes, list):
        raise TypeError("__notes__ must be a list")
    else:
        notes.append(note)
"""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_python_source(source: str, filename: str) -> tuple[str, list[str]]:
    """Move unsupported typing names to the maintained typing_extensions API."""
    try:
        tree = ast.parse(source, filename=filename, feature_version=(3, 10))
    except SyntaxError as exc:
        raise ValueError(
            f"Python 3.10 syntax audit failed: {filename}:{exc.lineno}: {exc.msg}"
        ) from exc
    data = source.encode("utf-8")
    line_offsets = [0]
    for line in data.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    edits = []
    imported = []
    needs_exception_helper = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {alias.name for alias in node.names}
            if node.module == "tomllib" or names & UNSUPPORTED_IMPORTS.get(
                node.module, set()
            ):
                raise ValueError(
                    f"Unbackported Python API in {filename}:{node.lineno}: {node.module}"
                )
            if node.module == "typing" and names & TYPING_BACKPORTS:
                old = [
                    alias for alias in node.names if alias.name not in TYPING_BACKPORTS
                ]
                new = [alias for alias in node.names if alias.name in TYPING_BACKPORTS]
                statements = []
                if old:
                    statements.append(
                        ast.unparse(ast.ImportFrom(module="typing", names=old, level=0))
                    )
                statements.append(
                    ast.unparse(
                        ast.ImportFrom(module="typing_extensions", names=new, level=0)
                    )
                )
                replacement = ("\n" + " " * node.col_offset).join(statements)
                start = line_offsets[node.lineno - 1] + node.col_offset
                end = line_offsets[node.end_lineno - 1] + node.end_col_offset
                edits.append((start, end, replacement.encode("utf-8")))
                imported.extend(alias.name for alias in new)
        elif isinstance(node, ast.Import):
            if any(alias.name == "tomllib" for alias in node.names):
                raise ValueError(f"Unbackported tomllib in {filename}:{node.lineno}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = node.value.id
            if module == "datetime" and node.attr == "UTC":
                start = line_offsets[node.lineno - 1] + node.col_offset
                end = line_offsets[node.end_lineno - 1] + node.end_col_offset
                edits.append((start, end, b"datetime.timezone.utc"))
                imported.append("datetime.UTC")
                continue
            if node.attr in UNSUPPORTED_ATTRIBUTES.get(module, set()) or (
                module == "typing" and node.attr in TYPING_BACKPORTS - {"TypedDict"}
            ):
                raise ValueError(
                    f"Unbackported Python API {module}.{node.attr} in {filename}:{node.lineno}"
                )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_note"
        ):
            if len(node.args) != 1 or node.keywords:
                raise ValueError(
                    f"Unexpected add_note signature in {filename}:{node.lineno}"
                )
            replacement = ast.Call(
                func=ast.Name(id="_sunrise_add_exception_note", ctx=ast.Load()),
                args=[node.func.value, node.args[0]],
                keywords=[],
            )
            start = line_offsets[node.lineno - 1] + node.col_offset
            end = line_offsets[node.end_lineno - 1] + node.end_col_offset
            edits.append((start, end, ast.unparse(replacement).encode("utf-8")))
            needs_exception_helper = True
        elif isinstance(node, ast.Name) and node.id in {
            "ExceptionGroup",
            "BaseExceptionGroup",
        }:
            raise ValueError(f"Unbackported {node.id} in {filename}:{node.lineno}")
    if needs_exception_helper:
        # Future imports must remain at the beginning, following any docstring.
        first_regular = next(
            node
            for node in tree.body
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                or isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        )
        offset = line_offsets[first_regular.lineno - 1]
        helper_import = (
            "from vllm._sunrise_python310 import "
            "add_exception_note as _sunrise_add_exception_note\n"
        )
        edits.append((offset, offset, helper_import.encode("utf-8")))
        imported.append("BaseException.add_note")
    for start, end, replacement in sorted(edits, reverse=True):
        data = data[:start] + replacement + data[end:]
    patched = data.decode("utf-8")
    ast.parse(patched, filename=filename, feature_version=(3, 10))
    return patched, sorted(set(imported))


def apply_compatibility(source_root: Path, manifest_path: Path) -> dict:
    """Validate the pinned tree, plan all edits, then emit source and provenance."""
    source_root = source_root.resolve(strict=True)
    manifest_path = manifest_path.resolve()
    if manifest_path.exists():
        raise ValueError(
            f"Refusing to overwrite compatibility manifest: {manifest_path}"
        )
    for relative, expected in SOURCE_HASHES.items():
        path = source_root / relative
        if path.is_symlink() or _digest(path.read_bytes()) != expected:
            raise ValueError(
                f"Expected unmodified official v{UPSTREAM_VERSION} source: {relative}"
            )
    package_root = source_root / "vllm"
    if not package_root.is_dir() or package_root.is_symlink():
        raise ValueError("Expected a regular vllm package directory")
    changes = {}
    audit_count = 0
    for path in [source_root / "setup.py", *sorted(package_root.rglob("*.py"))]:
        if path.is_symlink() or source_root not in path.resolve().parents:
            raise ValueError(f"Source path escapes the pinned tree: {path}")
        before = path.read_text(encoding="utf-8")
        relative = path.relative_to(source_root).as_posix()
        after, backports = patch_python_source(before, relative)
        audit_count += 1
        if after != before:
            changes[relative] = (before, after, backports)
    if any("BaseException.add_note" in names for _, _, names in changes.values()):
        if (source_root / EXCEPTION_HELPER_PATH).exists():
            raise ValueError("Unexpected existing downstream exception helper")
        ast.parse(EXCEPTION_HELPER_SOURCE, feature_version=(3, 10))
        changes[EXCEPTION_HELPER_PATH] = ("", EXCEPTION_HELPER_SOURCE, [])
        audit_count += 1
    project = source_root / "pyproject.toml"
    before = project.read_text(encoding="utf-8")
    old = 'requires-python = ">=3.11,<3.15"'
    if before.count(old) != 1:
        raise ValueError("Unexpected upstream Python requirement")
    after = before.replace(old, 'requires-python = ">=3.10,<3.15"', 1)
    changes["pyproject.toml"] = (before, after, [])
    manifest = {
        "upstream_version": UPSTREAM_VERSION,
        "upstream_archive_sha256": UPSTREAM_ARCHIVE_SHA256,
        "support_status": "experimental-downstream-python310; not upstream supported",
        "python_target": "3.10",
        "syntax_audited_files": audit_count,
        "dependency_policy": "Unchanged upstream requirements; resolver must succeed on Python 3.10",
        "validation_limits": "Syntax audit is not an import, inference or hardware acceptance result; Python 3.10 preserves exception __notes__ but its traceback printer does not render them",
        "changes": {
            relative: {
                "before_sha256": _digest(old.encode("utf-8")),
                "after_sha256": _digest(new.encode("utf-8")),
                "typing_backports": [name for name in backports if "." not in name],
                "runtime_backports": [name for name in backports if "." in name],
            }
            for relative, (old, new, backports) in sorted(changes.items())
        },
    }
    # Complete the audit before touching any source file.
    for relative, (_, patched, _) in changes.items():
        (source_root / relative).write_text(patched, encoding="utf-8", newline="\n")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = apply_compatibility(args.source_root, args.manifest)
    print(
        json.dumps(
            {
                "support_status": result["support_status"],
                "syntax_audited_files": result["syntax_audited_files"],
                "changed_files": len(result["changes"]),
            }
        )
    )


if __name__ == "__main__":
    main()

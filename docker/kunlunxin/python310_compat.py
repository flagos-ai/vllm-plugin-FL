# Copyright (c) 2026 BAAI. All rights reserved.
"""Repack the pinned FlagOS vLLM wheel for the Kunlunxin Python 3.10 stack.

The Python source rewrites come from the audited Sunrise 0.32 source build.
The vendor wheel already contains the empty-build dependency metadata and its
own torch.distributed compatibility fix. Preserve both when rebuilding RECORD.
"""

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

WHEEL_SHA256 = "2363ecc4c39173f075f4e431bab5d74eac8ea57fac5cbdac136f29bc5e6d8771"

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


EXCEPTION_HELPER_PATH = "vllm/_flagos_python310.py"
EXCEPTION_HELPER_SOURCE = """# Downstream FlagOS compatibility; upstream vLLM requires Python >= 3.11.
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


def patch_python_source(source: str, filename: str) -> str:
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
            if node.module == "datetime" and "UTC" in names:
                utc = next(alias for alias in node.names if alias.name == "UTC")
                others = [alias for alias in node.names if alias.name != "UTC"]
                others.append(ast.alias(name="timezone", asname="_flagos_timezone"))
                imports = ast.unparse(
                    ast.ImportFrom(module="datetime", names=others, level=0)
                )
                assignment = f"{utc.asname or utc.name} = _flagos_timezone.utc"
                replacement = imports + "\n" + " " * node.col_offset + assignment
                start = line_offsets[node.lineno - 1] + node.col_offset
                end = line_offsets[node.end_lineno - 1] + node.end_col_offset
                edits.append((start, end, replacement.encode("utf-8")))
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
        elif isinstance(node, ast.Import):
            if any(alias.name == "tomllib" for alias in node.names):
                raise ValueError(f"Unbackported tomllib in {filename}:{node.lineno}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = node.value.id
            if module == "datetime" and node.attr == "UTC":
                start = line_offsets[node.lineno - 1] + node.col_offset
                end = line_offsets[node.end_lineno - 1] + node.end_col_offset
                edits.append((start, end, b"datetime.timezone.utc"))
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
                func=ast.Name(id="_flagos_add_exception_note", ctx=ast.Load()),
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
            "from vllm._flagos_python310 import "
            "add_exception_note as _flagos_add_exception_note\n"
        )
        edits.append((offset, offset, helper_import.encode("utf-8")))
    for start, end, replacement in sorted(edits, reverse=True):
        data = data[:start] + replacement + data[end:]
    patched = data.decode("utf-8")
    ast.parse(patched, filename=filename, feature_version=(3, 10))
    return patched


def repack(wheel: Path, output_dir: Path) -> None:
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError("Expected the published vLLM 0.32.0+flagos wheel")
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        with ZipFile(wheel) as archive:
            archive.extractall(root)
        for path in sorted((root / "vllm").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            patched = patch_python_source(source, path.relative_to(root).as_posix())
            if patched != source:
                path.write_text(patched, encoding="utf-8", newline="\n")
        (root / EXCEPTION_HELPER_PATH).write_text(
            EXCEPTION_HELPER_SOURCE, encoding="utf-8", newline="\n"
        )
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(root), "-d", str(output_dir)],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    repack(args.wheel, args.output_dir)

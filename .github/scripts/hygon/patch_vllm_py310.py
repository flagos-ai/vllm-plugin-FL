#!/usr/bin/env python3
"""Adapt an upstream vLLM >= 0.32 checkout to Python 3.10 (build time only).

Upstream vLLM 0.32 declares ``requires-python = ">=3.11,<3.15"`` and imports
typing names that only exist from Python 3.11 (``Self``, ``override``,
``TypeIs``, ...), plus ``datetime.UTC``. The FlagOS 2.2 Hygon release image
ships Python 3.10, so this script rewrites those imports to their
``typing_extensions`` equivalents in place and relaxes ``requires-python``.

Monkeypatching ``typing`` globally is deliberately avoided: ``typing_extensions``
detects PEP 695 support with ``hasattr(typing, "TypeAliasType")`` and would then
call ``typing.ParamSpec(..., infer_variance=...)``, which Python 3.10 rejects.

The tree is compiled afterwards so syntax errors surface during the image build
instead of at runtime.
"""

import compileall
import re
import sys
from pathlib import Path

# Names that only exist in typing from Python 3.11 onwards.
BACKPORTED = frozenset(
    {
        "Self",
        "Never",
        "LiteralString",
        "assert_never",
        "NotRequired",
        "Required",
        "TypeVarTuple",
        "Unpack",
        "override",
        "CapsuleType",
        "TypeAliasType",
        "TypeIs",
        "deprecated",
        "reveal_type",
        "ReadOnly",
    }
)

TYPING_IMPORT_RE = re.compile(
    r"^(?P<indent>[ \t]*)from[ \t]+typing[ \t]+import[ \t]+"
    r"(?P<body>\([^)]*\)|[^\n]+)[ \t]*\n",
    re.MULTILINE,
)
DATETIME_IMPORT_RE = re.compile(
    r"^(?P<indent>[ \t]*)from[ \t]+datetime[ \t]+import[ \t]+(?P<names>[^\n(]+)\n",
    re.MULTILINE,
)
DATETIME_UTC_RE = re.compile(r"\bdatetime\.UTC\b")
REQUIRES_PYTHON_RE = re.compile(r'(requires-python\s*=\s*)"[^"]*"')


def split_names(body: str) -> list[str]:
    names = []
    for raw in body.replace("(", "").replace(")", "").split(","):
        name = raw.split("#", 1)[0].strip()
        if name:
            names.append(name)
    return names


def rewrite_typing_imports(source: str) -> tuple[str, int]:
    rewritten = 0

    def replace(match: re.Match) -> str:
        nonlocal rewritten
        names = split_names(match["body"])
        backported = [n for n in names if n.split()[0] in BACKPORTED]
        if not backported:
            return match.group(0)
        native = [n for n in names if n.split()[0] not in BACKPORTED]
        indent = match["indent"]
        lines = []
        if native:
            lines.append(f"{indent}from typing import {', '.join(native)}\n")
        lines.append(f"{indent}from typing_extensions import {', '.join(backported)}\n")
        rewritten += 1
        return "".join(lines)

    return TYPING_IMPORT_RE.sub(replace, source), rewritten


def rewrite_datetime_utc(source: str) -> tuple[str, int]:
    rewritten = 0

    def replace_import(match: re.Match) -> str:
        nonlocal rewritten
        names = split_names(match["names"])
        if "UTC" not in [n.split()[0] for n in names]:
            return match.group(0)
        remaining = [n for n in names if n.split()[0] != "UTC"]
        if "timezone" not in [n.split()[0] for n in remaining]:
            remaining.append("timezone")
        indent = match["indent"]
        rewritten += 1
        return (
            f"{indent}from datetime import {', '.join(remaining)}\n"
            f"\n{indent}UTC = timezone.utc  # Python 3.10 backport\n"
        )

    source = DATETIME_IMPORT_RE.sub(replace_import, source)
    source, count = DATETIME_UTC_RE.subn("datetime.timezone.utc", source)
    return source, rewritten + count


def patch_sources(root: Path) -> tuple[int, int]:
    typing_files = 0
    datetime_files = 0
    for path in (root / "vllm").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        patched, typing_changes = rewrite_typing_imports(source)
        patched, datetime_changes = rewrite_datetime_utc(patched)
        if patched == source:
            continue
        path.write_text(patched, encoding="utf-8")
        typing_files += bool(typing_changes)
        datetime_files += bool(datetime_changes)
    return typing_files, datetime_files


def relax_requires_python(root: Path) -> None:
    pyproject = root / "pyproject.toml"
    source = pyproject.read_text(encoding="utf-8")
    patched, count = REQUIRES_PYTHON_RE.subn(r'\1">=3.10,<3.15"', source, count=1)
    if count != 1:
        raise SystemExit("pyproject.toml: requires-python was not found exactly once")
    pyproject.write_text(patched, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <vllm-source-root>")
    root = Path(sys.argv[1]).resolve(strict=True)
    for required in (root / "vllm" / "__init__.py", root / "pyproject.toml"):
        if not required.is_file():
            raise SystemExit(f"not a vLLM source tree: missing {required}")

    typing_files, datetime_files = patch_sources(root)
    relax_requires_python(root)
    if not compileall.compile_dir(str(root / "vllm"), quiet=1, force=True):
        raise SystemExit("vllm package failed to compile for Python 3.10")
    print(
        "patched Python 3.10 compatibility: "
        f"{typing_files} files with backported typing imports, "
        f"{datetime_files} files with datetime.UTC"
    )


if __name__ == "__main__":
    main()

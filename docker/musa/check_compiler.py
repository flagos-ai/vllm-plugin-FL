#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Check that MUSA uses one consistent FlagTree / Triton 3.6 installation."""

import argparse
import ast
import importlib
from importlib import metadata
from pathlib import Path

TRITON_VERSION = "3.6.0"
REPAIR = (
    "Follow README.md 'Additional Steps for MUSA' to remove standalone Triton "
    "and install the FlagTree version pinned in requirements/musa.txt."
)


def check_compiler(metadata_only=False):
    try:
        provider = metadata.distribution("flagtree")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("FlagTree is not installed") from exc
    if "+mthreads" not in provider.version:
        raise RuntimeError(
            f"FlagTree {provider.version} is not a MThreads backend build"
        )
    try:
        standalone = metadata.distribution("triton")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            f"Standalone triton {standalone.version} is installed alongside FlagTree; "
            "both packages write the same triton namespace"
        )

    origin = Path(provider.locate_file("triton/__init__.py")).resolve()
    tree = ast.parse(origin.read_text(encoding="utf-8"))
    version = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            version = ast.literal_eval(node.value)
            break
    if version != TRITON_VERSION:
        raise RuntimeError(
            f"FlagTree provides Triton {version!r}, expected {TRITON_VERSION}"
        )
    backend = Path(provider.locate_file("triton/backends/mthreads/compiler.py"))
    if not backend.is_file():
        raise RuntimeError("FlagTree's triton.backends.mthreads compiler is missing")

    if not metadata_only:
        triton = importlib.import_module("triton")
        if Path(triton.__file__).resolve() != origin:
            raise RuntimeError(
                f"Active Triton comes from {triton.__file__}, expected {origin}"
            )
        if triton.__version__ != TRITON_VERSION:
            raise RuntimeError(
                f"Active Triton is {triton.__version__}, expected {TRITON_VERSION}"
            )
    print(f"FlagTree {provider.version} / Triton {version} / {origin}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Inspect installed files without importing accelerator runtimes",
    )
    args = parser.parse_args()
    try:
        check_compiler(metadata_only=args.metadata_only)
    except (RuntimeError, ImportError, OSError, SyntaxError, ValueError) as exc:
        parser.exit(1, f"MUSA compiler check failed: {exc}\n{REPAIR}\n")


if __name__ == "__main__":
    main()

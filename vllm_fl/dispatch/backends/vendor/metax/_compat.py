# Copyright (c) 2026 BAAI. All rights reserved.

"""Identify legacy MACA runtimes without importing their compiler modules."""

import importlib.metadata
import re
import sys


def _runtime_version(name: str) -> tuple[int, int] | None:
    module = sys.modules.get(name)
    value = getattr(module, "__version__", None)
    if value is None:
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    match = re.match(r"(\d+)\.(\d+)", str(value))
    return tuple(map(int, match.groups())) if match else None


def legacy_torch_runtime() -> bool:
    version = _runtime_version("torch")
    return version is not None and version < (2, 9)


def legacy_triton_runtime() -> bool:
    # The AST/constexpr workarounds were measured on MACA's Triton 3.0.
    # The vLLM 0.32 stack uses Triton 3.6 and must retain its native compiler.
    version = _runtime_version("triton")
    return version == (3, 0)

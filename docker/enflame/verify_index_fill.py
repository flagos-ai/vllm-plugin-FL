#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.

"""Verify the pinned FlagGems index_fill release bytes during Enflame setup.

This is an image/setup provenance check, not an engine startup requirement.
It uses package discovery and metadata without importing FlagGems or using a GPU.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

EXPECTED_VERSION = "5.4.0"
EXPECTED_BYTES = 11609
EXPECTED_SHA256 = "37aacdee85e497881f47a813b5489d26972b335ba89df312dbd74be5b1a2e07c"
SOURCE_PATH = "flag_gems/ops/index_fill.py"


def verify_index_fill() -> dict:
    distribution = importlib.metadata.distribution("flag-gems")
    if distribution.version != EXPECTED_VERSION:
        raise RuntimeError(f"Unsupported FlagGems release: {distribution.version}")
    root = Path(distribution.locate_file("")).resolve(strict=True)
    package = Path(distribution.locate_file("flag_gems/__init__.py")).resolve(
        strict=True
    )
    source = Path(distribution.locate_file(SOURCE_PATH)).resolve(strict=True)
    spec = importlib.util.find_spec("flag_gems")
    if (
        not package.is_relative_to(root)
        or not source.is_relative_to(package.parent)
        or spec is None
        or spec.origin is None
        or Path(spec.origin).resolve(strict=True) != package
    ):
        raise RuntimeError(
            "Active FlagGems package differs from the pinned distribution"
        )
    before = source.stat()
    with source.open("rb") as stream:
        raw = stream.read(EXPECTED_BYTES + 1)
    after = source.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise RuntimeError("FlagGems index_fill changed during release verification")
    digest = hashlib.sha256(raw).hexdigest()
    if (
        len(raw) != EXPECTED_BYTES
        or after.st_size != EXPECTED_BYTES
        or digest != EXPECTED_SHA256
    ):
        raise RuntimeError("FlagGems index_fill differs from the pinned release bytes")
    return {
        "passed": True,
        "scope": "pinned FlagGems index_fill release; no package import or GPU calls",
        "version": distribution.version,
        "path": str(source),
        "bytes": len(raw),
        "sha256": digest,
        "stable_during_read": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify_index_fill(), ensure_ascii=True))

# Copyright (c) 2026 BAAI. All rights reserved.
"""Install the FlagGems-vllm router dependency into the MetaX runtime."""

import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

COMMIT = "e00557103e284db5b57168386db60faa051989ef"
SOURCE = f"https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/{COMMIT}"
SHA256 = "2e19546b78a838c188b2e4e4f12b1a40b395a13fc75f24fbabb7229e63461656"
PACKAGE = "flaggems-vllm"


def installed_from_pin():
    try:
        record = distribution(PACKAGE).read_text("direct_url.json")
        source = json.loads(record or "{}")
    except (PackageNotFoundError, ValueError):
        return False
    archive = source.get("archive_info", {})
    digest = archive.get("hashes", {}).get("sha256")
    return source.get("url") == SOURCE and digest == SHA256


def verify_native_versions():
    manifest_path = os.environ.get("FL_METAX_STACK_MANIFEST")
    if not manifest_path:
        return
    locked_versions = json.loads(Path(manifest_path).read_text())
    expected = {
        name: value for name, value in locked_versions.items() if name != PACKAGE
    }
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"MetaX native runtime drifted: {expected} -> {actual}")


def main():
    verify_native_versions()
    if not installed_from_pin():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--force-reinstall",
                "--retries",
                "5",
                f"{PACKAGE} @ {SOURCE}#sha256={SHA256}",
            ],
            check=True,
        )
    if not installed_from_pin():
        raise RuntimeError(f"MetaX {PACKAGE} must come from commit {COMMIT}")
    verify_native_versions()
    print(f"[metax-ci] {PACKAGE} {version(PACKAGE)} from commit {COMMIT}")


if __name__ == "__main__":
    main()

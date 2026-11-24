# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Install FlashInfer 0.6.12 with the upstream CUDA runtime lookup fix."""

import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

from wheel.wheelfile import WheelFile

BASE_VERSION = "0.6.12"
FIXED_VERSION = "0.6.12+flagosci1"
WHEEL_SHA256 = "0c7a01e586b4796810d974cbf13a9c0eb2ade6a94d12e3220cf7782a1c09b8d3"
# Backport flashinfer-ai/flashinfer#5198, commit
# eabb5d3d34a32d60ede9048ab1856f76f959e04f. Keep the existing CUDA cubins.
MODULE_PATH = "flashinfer/comm/cuda_ipc.py"
FILENAME_CHECK = '''def _is_library_filename(maps_line: str, lib_name: str) -> bool:
    """Match the runtime filename, excluding similarly named stub libraries."""
    if "/" not in maps_line:
        return False
    filename = maps_line.strip().rsplit("/", 1)[-1]
    return re.fullmatch(
        rf"{re.escape(lib_name)}(?:-[0-9a-fA-F]+)?\\.so(?:\\.\\d+)*", filename
    ) is not None


'''


def build_fixed_wheel(original: Path, destination: Path) -> Path:
    """Rebuild the pinned wheel, updating its version and RECORD hashes."""
    if hashlib.sha256(original.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise RuntimeError("Unexpected FlashInfer wheel SHA-256")
    fixed = destination / original.name.replace(BASE_VERSION, FIXED_VERSION, 1)
    old_metadata = f"flashinfer_python-{BASE_VERSION}.dist-info/"
    new_metadata = f"flashinfer_python-{FIXED_VERSION}.dist-info/"
    with zipfile.ZipFile(original) as source:
        module = source.read(MODULE_PATH).decode("utf-8")
        condition = "            if lib_name in line:\n"
        if module.count(condition) != 1 or "def _is_library_filename(" in module:
            raise RuntimeError("Unexpected FlashInfer CUDA runtime lookup source")
        module = module.replace("import ctypes\n", "import ctypes\nimport re\n", 1)
        module = module.replace(
            "def find_loaded_library(", FILENAME_CHECK + "def find_loaded_library(", 1
        )
        module = module.replace(
            condition,
            "            if lib_name in line and _is_library_filename(line, lib_name):\n",
            1,
        )
        compile(module, MODULE_PATH, "exec")
        with WheelFile(fixed, "w") as target:
            for entry in source.infolist():
                if entry.filename == old_metadata + "RECORD":
                    continue
                content = source.read(entry)
                if entry.filename == MODULE_PATH:
                    content = module.encode("utf-8")
                if entry.filename == old_metadata + "METADATA":
                    previous = f"Version: {BASE_VERSION}\n".encode()
                    if content.count(previous) != 1:
                        raise RuntimeError("Unexpected FlashInfer wheel metadata")
                    content = content.replace(
                        previous, f"Version: {FIXED_VERSION}\n".encode(), 1
                    )
                entry.filename = entry.filename.replace(old_metadata, new_metadata, 1)
                target.writestr(entry, content)
    return fixed


def main() -> None:
    installed = version("flashinfer-python")
    if installed == FIXED_VERSION:
        print(f"FlashInfer CUDA runtime fix already installed: {installed}")
        return
    if installed != BASE_VERSION:
        raise RuntimeError(f"Expected FlashInfer {BASE_VERSION}, found {installed}")
    runtime = ("torch", "vllm", "numpy", "flashinfer-cubin")
    before = {name: version(name) for name in runtime}
    with tempfile.TemporaryDirectory(prefix="flashinfer-ci-") as directory:
        workspace = Path(directory)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-deps",
                "--only-binary=:all:",
                "--dest",
                str(workspace),
                f"flashinfer-python=={BASE_VERSION}",
            ],
            check=True,
        )
        original = next(workspace.glob("*.whl"))
        fixed = build_fixed_wheel(original, workspace)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(fixed),
            ],
            env=dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1"),
            check=True,
        )
    if version("flashinfer-python") != FIXED_VERSION:
        raise RuntimeError("FlashInfer backport installation failed")
    after = {name: version(name) for name in runtime}
    if after != before:
        raise RuntimeError(f"CI runtime dependencies changed: {before} -> {after}")
    print(f"FlashInfer {FIXED_VERSION}: backported upstream #5198")


if __name__ == "__main__":
    main()

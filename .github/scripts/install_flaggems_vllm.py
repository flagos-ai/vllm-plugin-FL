# Copyright (c) 2026 BAAI. All rights reserved.
"""Install FlagGems-vllm main in every platform's CI interpreter."""

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

REPOSITORY = "https://github.com/flagos-ai/FlagGems-vllm.git"
RUNTIME_PACKAGES = ("torch", "triton", "flag-gems", "vllm", "numpy")
PACKAGE_DEPENDENCIES = ("packaging", "pybind11", "PyYAML", "sqlalchemy")


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def pip_install(*args):
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1")
    subprocess.run([sys.executable, "-m", "pip", "install", *args], env=env, check=True)


def main():
    before = {name: package_version(name) for name in RUNTIME_PACKAGES}
    missing = [name for name in PACKAGE_DEPENDENCIES if package_version(name) is None]
    if missing:
        pip_install(*missing)
    pip_install(
        "--no-deps",
        "--no-build-isolation",
        "--force-reinstall",
        f"flaggems-vllm @ git+{REPOSITORY}@main",
    )
    after = {name: package_version(name) for name in RUNTIME_PACKAGES}
    if after != before:
        raise RuntimeError(f"CI runtime dependencies changed: {before} -> {after}")
    print(f"FlagGems-vllm main: {package_version('flaggems-vllm')}")


if __name__ == "__main__":
    main()

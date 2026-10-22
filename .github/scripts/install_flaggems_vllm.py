# Copyright (c) 2026 BAAI. All rights reserved.
"""Install the same pinned FlagGems-vllm source in every CI interpreter."""

import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version

# Public HY4/GLM APIs, portable math/constexpr hints, and XPU timer recovery.
# Includes the shared main pin plus flagos-ai/FlagGems-vllm#941.
COMMIT = "27f7b83a0c3bf05c288e513280c6f8337893b7a7"
REPOSITORY = "https://github.com/flagos-ai/FlagGems-vllm.git"
RUNTIME_PACKAGES = ("torch", "triton", "flag-gems", "vllm", "numpy")
PACKAGE_DEPENDENCIES = ("packaging", "pybind11", "PyYAML", "sqlalchemy")


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def source_is_installed():
    try:
        text = distribution("flaggems-vllm").read_text("direct_url.json")
        source = json.loads(text or "{}")
    except (PackageNotFoundError, ValueError):
        return False
    if not isinstance(source, dict) or not isinstance(source.get("vcs_info"), dict):
        return False
    return (
        source.get("url", "").removesuffix(".git") == REPOSITORY.removesuffix(".git")
        and source.get("vcs_info", {}).get("commit_id") == COMMIT
    )


def pip_install(*args):
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1")
    subprocess.run([sys.executable, "-m", "pip", "install", *args], env=env, check=True)


def main():
    before = {name: package_version(name) for name in RUNTIME_PACKAGES}
    missing = [name for name in PACKAGE_DEPENDENCIES if package_version(name) is None]
    if missing:
        pip_install(*missing)
    if not source_is_installed():
        # Replace another source even if its package version is identical.
        # Reuse the build tools already used by this platform's plugin setup.
        pip_install(
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            f"flaggems-vllm @ git+{REPOSITORY}@{COMMIT}",
        )
    if not source_is_installed():
        raise RuntimeError(f"FlagGems-vllm must come from upstream commit {COMMIT}")
    after = {name: package_version(name) for name in RUNTIME_PACKAGES}
    if after != before:
        raise RuntimeError(f"CI runtime dependencies changed: {before} -> {after}")
    print(
        f"FlagGems-vllm: upstream commit {COMMIT}, "
        f"package metadata {package_version('flaggems-vllm')}"
    )


if __name__ == "__main__":
    main()

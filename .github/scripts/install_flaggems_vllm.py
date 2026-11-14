# Copyright (c) 2026 BAAI. All rights reserved.
"""Install the pinned FlagGems-vllm source in the current CI interpreter."""

import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version

RELEASE = "v0.2.0"
COMMIT = "b5c2c2249319277ccda4d41aa0f65164bd9b2a19"
SUNRISE_COMMIT = "3b0f3880b2a781699462eacb7dd3e00cc24f2d43"
REPOSITORY = "https://github.com/flagos-ai/FlagGems-vllm.git"
RUNTIME_PACKAGES = ("torch", "flagtree", "triton", "flag-gems", "vllm", "numpy")
PACKAGE_DEPENDENCIES = ("packaging", "pybind11", "PyYAML", "sqlalchemy")


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def release_is_installed(commit=COMMIT):
    try:
        text = distribution("flaggems-vllm").read_text("direct_url.json")
        source = json.loads(text or "{}")
    except (PackageNotFoundError, ValueError):
        return False
    return (
        source.get("url", "").removesuffix(".git") == REPOSITORY.removesuffix(".git")
        and source.get("vcs_info", {}).get("commit_id") == commit
    )


def pip_install(*args):
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1")
    subprocess.run([sys.executable, "-m", "pip", "install", *args], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sunrise", action="store_true")
    parser.add_argument("--print-commit", action="store_true")
    args = parser.parse_args()
    commit = SUNRISE_COMMIT if args.sunrise else COMMIT
    if args.print_commit:
        print(commit)
        return

    before = {name: package_version(name) for name in RUNTIME_PACKAGES}
    missing = [name for name in PACKAGE_DEPENDENCIES if package_version(name) is None]
    if missing:
        pip_install(*missing)
    if not release_is_installed(commit):
        # Metadata still says 0.1.0; force replacement of another 0.1.0 source.
        # Reuse the build tools already used by this platform's plugin setup.
        pip_install(
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            f"flaggems-vllm @ git+{REPOSITORY}@{commit}",
        )
    if not release_is_installed(commit):
        raise RuntimeError(f"FlagGems-vllm must come from {REPOSITORY} at {commit}")
    after = {name: package_version(name) for name in RUNTIME_PACKAGES}
    if after != before:
        raise RuntimeError(f"CI runtime dependencies changed: {before} -> {after}")
    print(
        f"FlagGems-vllm: commit {commit}, "
        f"package metadata {package_version('flaggems-vllm')}"
    )


if __name__ == "__main__":
    main()

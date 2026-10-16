# Copyright (c) 2026 BAAI. All rights reserved.
"""Install FlagGems-vllm main in every platform's CI interpreter."""

import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version

SOURCE_REF = "main"
REPOSITORY = "https://github.com/flagos-ai/FlagGems-vllm.git"
RUNTIME_PACKAGES = ("torch", "triton", "flag-gems", "vllm", "numpy")
PACKAGE_DEPENDENCIES = ("packaging", "pybind11", "PyYAML", "sqlalchemy")


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def installed_source():
    try:
        text = distribution("flaggems-vllm").read_text("direct_url.json")
        source = json.loads(text or "{}")
    except (PackageNotFoundError, ValueError):
        return {}
    return source if isinstance(source, dict) else {}


def pip_install(*args):
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1")
    subprocess.run([sys.executable, "-m", "pip", "install", *args], env=env, check=True)


def main():
    before = {name: package_version(name) for name in RUNTIME_PACKAGES}
    missing = [name for name in PACKAGE_DEPENDENCIES if package_version(name) is None]
    if missing:
        pip_install(*missing)
    # Branch contents can change without a package-version change. Refresh main
    # even when the image already contains an installation from that branch.
    pip_install(
        "--no-deps",
        "--no-build-isolation",
        "--force-reinstall",
        f"flaggems-vllm @ git+{REPOSITORY}@{SOURCE_REF}",
    )
    source = installed_source()
    vcs = source.get("vcs_info", {})
    if (
        source.get("url", "").removesuffix(".git") != REPOSITORY.removesuffix(".git")
        or vcs.get("requested_revision") != SOURCE_REF
        or not vcs.get("commit_id")
    ):
        raise RuntimeError(f"FlagGems-vllm must come from {REPOSITORY}@{SOURCE_REF}")
    after = {name: package_version(name) for name in RUNTIME_PACKAGES}
    if after != before:
        raise RuntimeError(f"CI runtime dependencies changed: {before} -> {after}")
    print(
        f"FlagGems-vllm: {SOURCE_REF}, commit {vcs['commit_id']}, "
        f"package metadata {package_version('flaggems-vllm')}"
    )


if __name__ == "__main__":
    main()

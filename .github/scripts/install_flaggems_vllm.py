# Copyright (c) 2026 BAAI. All rights reserved.
"""Install the pinned FlagGems-vllm source in the platform's CI interpreter."""

import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version

RELEASE = "v0.2.0"
COMMIT = "b5c2c2249319277ccda4d41aa0f65164bd9b2a19"
# T-Head Qwen3.8 requires the HC, PLE and QSA operator exports.
THEAD_COMMIT = "80a8e9716abed1bac3fc0ba5fd4e9f650ae3f989"
THEAD_ARCHIVE = f"https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/{THEAD_COMMIT}"
THEAD_ARCHIVE_SHA256 = "1b51d13ac797e1396193e9841aa8af1fbf1478bad7d51466266803cd3343ba06"
REPOSITORY = "https://github.com/flagos-ai/FlagGems-vllm.git"
RUNTIME_PACKAGES = ("torch", "flagtree", "triton", "flag-gems", "vllm", "numpy")
PACKAGE_DEPENDENCIES = ("packaging", "pybind11", "PyYAML", "sqlalchemy")


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def release_is_installed(thead):
    try:
        text = distribution("flaggems-vllm").read_text("direct_url.json")
        source = json.loads(text or "{}")
    except (PackageNotFoundError, ValueError):
        return False
    if thead:
        return (
            source.get("url") == THEAD_ARCHIVE
            and source.get("archive_info", {}).get("hashes", {}).get("sha256")
            == THEAD_ARCHIVE_SHA256
        )
    return (
        source.get("url", "").removesuffix(".git") == REPOSITORY.removesuffix(".git")
        and source.get("vcs_info", {}).get("commit_id") == COMMIT
    )


def pip_install(*args):
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1")
    subprocess.run([sys.executable, "-m", "pip", "install", *args], env=env, check=True)


def main():
    thead = os.environ.get("GEMS_VENDOR", "").lower() == "thead"
    release, commit = RELEASE, COMMIT
    requirement_url = f"git+{REPOSITORY}@{commit}"
    if thead:
        release, commit = "0.3.0.dev1", THEAD_COMMIT
        # The source archive also works on PPU runners without Git HTTPS access.
        requirement_url = f"{THEAD_ARCHIVE}#sha256={THEAD_ARCHIVE_SHA256}"
    before = {name: package_version(name) for name in RUNTIME_PACKAGES}
    missing = [name for name in PACKAGE_DEPENDENCIES if package_version(name) is None]
    if missing:
        pip_install(*missing)
    if not release_is_installed(thead):
        # Replace a different source even if its metadata has the same version.
        # Reuse the build tools already used by this platform's plugin setup.
        pip_install(
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            f"flaggems-vllm @ {requirement_url}",
        )
    if not release_is_installed(thead):
        raise RuntimeError(f"FlagGems-vllm must come from {release} commit {commit}")
    after = {name: package_version(name) for name in RUNTIME_PACKAGES}
    if after != before:
        raise RuntimeError(f"CI runtime dependencies changed: {before} -> {after}")
    print(
        f"FlagGems-vllm: {release}, commit {commit}, "
        f"package metadata {package_version('flaggems-vllm')}"
    )


if __name__ == "__main__":
    main()

# Copyright (c) 2026 BAAI. All rights reserved.
"""Install a pinned FlagGems-vllm release without changing runtime dependencies."""

import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version

RELEASE = "v0.2.0"
COMMIT = "b5c2c2249319277ccda4d41aa0f65164bd9b2a19"
ENFLAME_COMMIT = "93fd1b0c1dc3b084dbf6872630fc9bb6113218ed"
REPOSITORY = "https://github.com/flagos-ai/FlagGems-vllm.git"
if os.environ.get("CI_FLAGGEMS_VLLM_VENDOR") == "iluvatar":
    COMMIT = "90dd13cd721f2ff139e6f42ef36854518c0b2b95"
    REPOSITORY = "https://github.com/flagos-ai/FlagGems-vllm.git"
RUNTIME_PACKAGES = ("torch", "flagtree", "triton", "flag-gems", "vllm", "numpy")
PACKAGE_DEPENDENCIES = ("packaging", "pybind11", "PyYAML", "sqlalchemy")


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def release_is_installed(commit=COMMIT, source_archive=None, archive_sha256=None):
    try:
        text = distribution("flaggems-vllm").read_text("direct_url.json")
        source = json.loads(text or "{}")
    except (PackageNotFoundError, ValueError):
        return False
    if source_archive is not None:
        digest = source.get("archive_info", {}).get("hashes", {}).get("sha256")
        return source.get("url") == source_archive and digest == archive_sha256
    return (
        source.get("url", "").removesuffix(".git") == REPOSITORY.removesuffix(".git")
        and source.get("vcs_info", {}).get("commit_id") == commit
    )


def pip_install(*args):
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1")
    subprocess.run([sys.executable, "-m", "pip", "install", *args], env=env, check=True)


def main(argv=None, *, commit=None, source_archive=None, archive_sha256=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform", default="", help="Use Enflame-specific source pin"
    )
    args = parser.parse_args(argv)
    if commit is None:
        commit = ENFLAME_COMMIT if args.platform == "enflame" else COMMIT
    release = "0.3.0.dev1" if args.platform == "enflame" else RELEASE
    source = (
        f"{source_archive}#sha256={archive_sha256}"
        if source_archive is not None
        else f"git+{REPOSITORY}@{commit}"
    )
    before = {name: package_version(name) for name in RUNTIME_PACKAGES}
    missing = [name for name in PACKAGE_DEPENDENCIES if package_version(name) is None]
    if missing:
        pip_install(*missing)
    if not release_is_installed(commit, source_archive, archive_sha256):
        # Metadata may conceal a different source commit or archive.
        # Reuse the build tools already used by this platform's plugin setup.
        pip_install(
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            f"flaggems-vllm @ {source}",
        )
    if not release_is_installed(commit, source_archive, archive_sha256):
        raise RuntimeError(f"FlagGems-vllm must come from source commit {commit}")
    after = {name: package_version(name) for name in RUNTIME_PACKAGES}
    if after != before:
        raise RuntimeError(f"CI runtime dependencies changed: {before} -> {after}")
    print(
        f"FlagGems-vllm: {release}, source commit {commit}, "
        f"package metadata {package_version('flaggems-vllm')}"
    )


if __name__ == "__main__":
    main()

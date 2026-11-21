# Copyright (c) 2026 BAAI. All rights reserved.
"""Install the FlagGems-vllm source pinned for the CI interpreter."""

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
SOURCE_ARCHIVE = f"https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/{COMMIT}"
SOURCE_ARCHIVE_SHA256 = (
    "1b47e1eb7982acc606e2fc920561d169e2a17fac7c92ff65c5f11cafa6b8b16e"
)
if os.environ.get("CI_FLAGGEMS_VLLM_VENDOR") == "kunlunxin":
    COMMIT = "8ae5c22f5af6d650216f881c329c4dff20c182e1"
    SOURCE_ARCHIVE = (
        f"https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/{COMMIT}"
    )
    SOURCE_ARCHIVE_SHA256 = (
        "eab3544a5a6a2efaf7b3465a04ea0f867f77cfbe144ed7f2b8ad20c7b28bb36c"
    )
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
    if (
        source_archive is None
        and commit == COMMIT
        and SOURCE_ARCHIVE is not None
        and source.get("url") == SOURCE_ARCHIVE
    ):
        source_archive, archive_sha256 = SOURCE_ARCHIVE, SOURCE_ARCHIVE_SHA256
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


def main(*, commit=COMMIT, source_archive=None, archive_sha256=None):
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
        # Metadata still says 0.1.0; force replacement of another 0.1.0 source.
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
        f"FlagGems-vllm: source commit {commit}, "
        f"package metadata {package_version('flaggems-vllm')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--sunrise", action="store_true")
    source_group.add_argument("--source-archive", action="store_true")
    parser.add_argument("--print-commit", action="store_true")
    args = parser.parse_args()
    commit = SUNRISE_COMMIT if args.sunrise else COMMIT
    if args.print_commit:
        print(commit)
    elif args.source_archive:
        if SOURCE_ARCHIVE is None:
            parser.error("--source-archive is unavailable for the selected provider")
        main(source_archive=SOURCE_ARCHIVE, archive_sha256=SOURCE_ARCHIVE_SHA256)
    else:
        main(commit=commit)

# Copyright (c) 2026 BAAI. All rights reserved.
"""Verify the NVIDIA CI FlagGems tag by its Git provenance."""

import json
from importlib.metadata import PackageNotFoundError, distribution, version

RELEASE = "v5.4.0"
COMMIT = "6ed2f39071ef85db26b7e64f0652a162f434959c"
REPOSITORY = "https://github.com/flagos-ai/FlagGems.git"


def release_is_installed():
    try:
        source = json.loads(
            distribution("flag-gems").read_text("direct_url.json") or "{}"
        )
    except (PackageNotFoundError, ValueError):
        return False
    return (
        source.get("url", "").removesuffix(".git") == REPOSITORY.removesuffix(".git")
        and source.get("vcs_info", {}).get("commit_id") == COMMIT
    )


if __name__ == "__main__":
    if not release_is_installed():
        raise SystemExit(f"NVIDIA CI requires FlagGems {RELEASE}, source {COMMIT}")
    print(f"FlagGems: {RELEASE}, commit {COMMIT}, metadata {version('flag-gems')}")

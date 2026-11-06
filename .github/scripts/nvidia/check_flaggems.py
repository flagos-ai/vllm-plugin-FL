# Copyright (c) 2026 BAAI. All rights reserved.
"""Check the NVIDIA CI FlagGems release without trusting placeholder metadata."""

import json
from importlib.metadata import PackageNotFoundError, distribution, version

RELEASE = "5.4.0"
COMMIT = "6ed2f39071ef85db26b7e64f0652a162f434959c"
REPOSITORY = "https://github.com/flagos-ai/FlagGems.git"


def release_is_installed():
    try:
        if version("flag-gems") == RELEASE:
            return True
        text = distribution("flag-gems").read_text("direct_url.json")
        source = json.loads(text or "{}")
    except (PackageNotFoundError, ValueError):
        return False
    return (
        source.get("url", "").removesuffix(".git") == REPOSITORY.removesuffix(".git")
        and source.get("vcs_info", {}).get("commit_id") == COMMIT
    )


def main():
    if not release_is_installed():
        raise SystemExit(f"NVIDIA CI requires FlagGems v{RELEASE}, source {COMMIT}")
    print(f"FlagGems: v{RELEASE}, package metadata {version('flag-gems')}")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 BAAI. All rights reserved.
"""Install the v0.2.0 backport that provides the Ascend V2 sampler operators."""

from install_flaggems_vllm import main

COMMIT = "622f5bf0e476dcdb861e50ea49cc94b0852c2619"
ARCHIVE = f"https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/{COMMIT}"
SHA256 = "9dc791f6c8ef66291bab751805c35a4bee7477f81cca90baf4fb6c2d3f00a7fe"

if __name__ == "__main__":
    main(commit=COMMIT, source_archive=ARCHIVE, archive_sha256=SHA256)

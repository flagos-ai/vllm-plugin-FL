#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
set -euo pipefail

repo=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
revision=$(git -C "$repo" rev-parse --verify "HEAD^{commit}")

git -C "$repo" archive "$revision" | docker build \
    --platform linux/amd64 \
    --file docker/thead/Dockerfile.v0.32.0 \
    --build-arg "PLUGIN_SOURCE_SHA=$revision" \
    --target ci \
    "$@" -

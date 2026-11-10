#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
set -euo pipefail

repo=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
revision=$(git -C "$repo" rev-parse --verify "HEAD^{commit}")
source=$(git -C "$repo" show "$revision:docker/thead/plugin-source.ref")
[[ "$source" =~ ^[0-9a-f]{40}$ ]]

if ! git -C "$repo" cat-file -e "$source^{commit}"; then
    git -C "$repo" fetch --no-tags https://github.com/BrianPei/vllm-plugin-FL.git "$source"
fi

context=$(mktemp -d)
trap 'rm -rf -- "$context"' EXIT
mkdir "$context/plugin"
git -C "$repo" archive "$revision" docker/thead .github/scripts/install_flaggems_vllm.py \
    | tar -x -C "$context"
git -C "$repo" archive "$source" pyproject.toml setup.py README.md LICENSE vllm_fl \
    | tar -x -C "$context/plugin"

docker build \
    --platform linux/amd64 \
    --file "$context/docker/thead/Dockerfile.v0.32.0" \
    --build-arg "BUILD_SOURCE_SHA=$revision" \
    --build-arg "PLUGIN_SOURCE_SHA=$source" \
    --target inference \
    "$@" "$context"

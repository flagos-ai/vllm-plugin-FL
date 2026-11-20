#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
set -euo pipefail

repo="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
image="${1:-vllm-plugin-fl-thead:0.32.0-runtime-2.3.0}"
read -r source_sha < "$repo/docker/thead/plugin-source.ref"
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]]
git -C "$repo" cat-file -e "$source_sha^{commit}"
epoch="$(git -C "$repo" show -s --format=%ct "$source_sha")"

context="$(mktemp -d)"
trap 'rm -rf -- "$context"' EXIT
mkdir "$context/plugin" "$context/recipe" "$context/empty-seed"
git -C "$repo" archive "$source_sha" | tar -xf - -C "$context/plugin"
cd "$repo"
install -D -m 0644 docker/thead/Dockerfile.v0.32.0 "$context/recipe/docker/thead/Dockerfile.v0.32.0"
for file in docker/thead/{constraints,requirements-ci}.txt docker/thead/fetch-ci-wheels.py .github/scripts/thead/{runtime_stack.py,runtime-stack.json}; do
    install -D -m 0644 "$file" "$context/recipe/$file"
done
docker build --platform linux/amd64 --target ci --progress=plain \
    --build-context "plugin_source=$context/plugin" \
    --build-context "wheel_seed=${THEAD_WHEEL_SEED:-$context/empty-seed}" \
    --build-arg "PLUGIN_SOURCE_SHA=$source_sha" --build-arg "SOURCE_DATE_EPOCH=$epoch" \
    -f "$context/recipe/docker/thead/Dockerfile.v0.32.0" -t "$image" "$context/recipe"
docker image inspect "$image" --format '{{.Id}}'

#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Prepare a private vLLM 0.32 environment over the pinned Enflame SDK image.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_OUTPUT=${1:?Usage: prepare_vllm032.sh ENV_OUTPUT_FILE}
UPSTREAM_COMMIT=5df63a7203cdb9328d824a27b682e77c7b9928be
UV_VERSION=0.13.6
BUILD_ROOT=$(mktemp -d "${VLLM_ENFLAME_BUILD_ROOT:-${RUNNER_TEMP:-/tmp}}/enflame-vllm032.XXXXXX")
VENV=${VLLM_ENFLAME_VENV:-"$BUILD_ROOT/.venv"}
SOURCE="$BUILD_ROOT/source"
INDEX_URL=${VLLM_PYPI_INDEX_URL:-https://pypi.org/simple}
[[ "$INDEX_URL" == https://* && "$INDEX_URL" != *'@'* && "$INDEX_URL" != *'?'* && "$INDEX_URL" != *'#'* ]]
[[ ! -e "$VENV" ]] || { echo "Refusing to overwrite existing environment: $VENV" >&2; exit 1; }

if [[ -n "${VLLM_ENFLAME_UV:-}" ]]; then
  UV=$VLLM_ENFLAME_UV
else
  python -m pip install --no-deps --target "$BUILD_ROOT/uv-runtime" \
    --index-url "$INDEX_URL" "uv==$UV_VERSION"
  UV="$BUILD_ROOT/uv-runtime/bin/uv"
fi
[[ "$("$UV" --version | awk '{print $2}')" == "$UV_VERSION" ]]

if [[ -n "${VLLM_SOURCE_REPOSITORY:-}" ]]; then
  # Read-only local baseline: clone Git objects, never build in the supplied tree.
  git clone --no-hardlinks "$VLLM_SOURCE_REPOSITORY" "$SOURCE"
else
  SOURCE_URL=${VLLM_SOURCE_URL:-https://github.com/vllm-project/vllm.git}
  [[ "$SOURCE_URL" == https://* && "$SOURCE_URL" != *'@'* && "$SOURCE_URL" != *'?'* && "$SOURCE_URL" != *'#'* ]]
  git clone --depth 1 --branch v0.32.0 "$SOURCE_URL" "$SOURCE"
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]]
"$UV" venv --system-site-packages --python "$(command -v python)" "$VENV"
PYTHON="$VENV/bin/python"
(
  cd "$SOURCE"
  VLLM_TARGET_DEVICE=empty "$UV" pip install --python "$PYTHON" \
    --no-build-isolation --no-deps .
)

WHEEL_ARGS=(--index-url "$INDEX_URL")
if [[ -n "${VLLM_ENFLAME_WHEELHOUSE:-}" ]]; then
  WHEEL_ARGS=(--no-index --find-links "$VLLM_ENFLAME_WHEELHOUSE")
fi
# Resolve no transitive packages: preserve vendor Torch and FlagTree. The
# verifier below independently checks the effective recursive dependency tree.
"$UV" pip install --python "$PYTHON" --no-deps --require-hashes \
  "${WHEEL_ARGS[@]}" -r "$SCRIPT_DIR/requirements-vllm032.txt"
"$PYTHON" "$SCRIPT_DIR/verify_compiler.py"
"$PYTHON" "$SCRIPT_DIR/verify_index_fill.py"
"$PYTHON" "$SCRIPT_DIR/verify_runtime.py" --source "$SOURCE" \
  --output "$VENV/enflame-vllm032-provenance.json"

# CI setup sources this file in its current shell, then exports the interpreter
# to later Actions steps. Keep the private source tree for reproducibility.
printf 'export PATH=%q\n' "$VENV/bin:$PATH" > "$ENV_OUTPUT"
printf 'export VLLM_ENFLAME_UV=%q\n' "$UV" >> "$ENV_OUTPUT"
printf 'export VLLM_ENFLAME_VENV=%q\n' "$VENV" >> "$ENV_OUTPUT"

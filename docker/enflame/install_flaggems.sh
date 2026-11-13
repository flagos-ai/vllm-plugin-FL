#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install the exact source release used by the S60 adaptation environment.
set -euo pipefail
COMMIT=2bcfe2656dad0b654077fcfe7d66e6a93542afab
WORK="$(mktemp -d "${TMPDIR:-/tmp}/enflame-flaggems.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SOURCE="${ENFLAME_FLAGGEMS_SOURCE_DIR:-$WORK/source}"
if [[ -z "${ENFLAME_FLAGGEMS_SOURCE_DIR:-}" ]]; then
  git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone --depth 1 \
    --branch v5.5.0.dev0 https://github.com/flagos-ai/FlagGems.git "$SOURCE"
fi
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
# Honor the source's setuptools<77 build requirements in a PEP 517 sandbox;
# vLLM needs newer build tools in the runtime interpreter.
python -m pip install --index-url "${PYPI_INDEX_URL:-https://pypi.org/simple}" \
  --no-deps --force-reinstall "$SOURCE"
python -c 'from importlib.metadata import version; assert version("flag-gems") == "5.5.0.dev0", version("flag-gems")'

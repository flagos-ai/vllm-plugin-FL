#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${FLAGCX_PATH:-}" ]]; then
    revision=6143384797219733aee89bd3118b2cb184ad4959
    FLAGCX_PATH=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/iluvatar-flagcx.XXXXXX")
    git -C "$FLAGCX_PATH" init
    git -C "$FLAGCX_PATH" remote add origin https://github.com/flagos-ai/FlagCX.git
    git -C "$FLAGCX_PATH" fetch --depth 1 origin "$revision"
    git -C "$FLAGCX_PATH" checkout --detach FETCH_HEAD
    test "$(git -C "$FLAGCX_PATH" rev-parse HEAD)" = "$revision"
    git -C "$FLAGCX_PATH" submodule update --init --depth 1 third-party/json
    make -C "$FLAGCX_PATH" -j4 USE_ILUVATAR_COREX=1
fi

FLAGCX_PATH=$(cd "$FLAGCX_PATH" && pwd)
export FLAGCX_PATH
for file in build/lib/libflagcx.so plugin/interservice/flagcx_wrapper.py; do
    test -f "$FLAGCX_PATH/$file" || { echo "Missing FlagCX file: $FLAGCX_PATH/$file" >&2; exit 2; }
done
python - <<'PY'
import ctypes
import os
from pathlib import Path

root = Path(os.environ["FLAGCX_PATH"])
ctypes.CDLL(str(root / "build/lib/libflagcx.so"))
print(f"Validated FlagCX: {root}")
PY
if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf 'FLAGCX_PATH=%s\n' "$FLAGCX_PATH" >> "$GITHUB_ENV"
fi

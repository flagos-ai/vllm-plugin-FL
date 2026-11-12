#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Upgrade an Enflame dependency image without replacing its vendor Torch SDK.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python "${ROOT}/.github/scripts/enflame/verify_stack.py" --runtime-only

# Keep pins and wheel hashes identical to docker/enflame/Dockerfile.
VLLM_COMMIT=5df63a7203cdb9328d824a27b682e77c7b9928be
INDEX="${FLAGOS_INDEX_URL:-https://resource.flagos.net/repository/flagos-pypi-enflame/simple}"
# The dependency image defaults to a vendor mirror that lacks newer common deps.
# Keep generic packages on PyPI and vendor wheels on their hash-checked index.
PYPI_INDEX="${PYPI_INDEX_URL:-https://pypi.org/simple}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/enflame-vllm032.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
printf '%s\n' \
  'torch==2.11.0+cpu' \
  'torch-gcu==2.11.0+3.8.20260713' \
  'triton==3.6.0' \
  'triton-gcu==3.6.0+1.0.20260722' \
  'flagtree==0.7.0+enflame3.6' \
  'flag-gems==5.4.0' > "$WORK/constraints.txt"
printf '%s\n' \
  'flag-gems==5.4.0 --hash=sha256:09cb175e29f4fdfc78b87df46c4500d5a9b451e12cd64dda9a7d669f522bf5e5' \
  'flagtree==0.7.0+enflame3.6 --hash=sha256:94b3aaf9f9aefa02f535ee5339f16c5304c52a1de5e7e749f742f7b4bf01332d' \
  'triton-gcu==3.6.0+1.0.20260722 --hash=sha256:59d7a450a48f8de2a965ea6e456ea7c13bf53cb83abd5df929e4e8558da5f8f9' > "$WORK/wheels.txt"
# Fetch inputs before modifying the installed environment. No package fallback.
python -m pip download --no-deps --only-binary=:all: --require-hashes \
  --index-url "$INDEX" --dest "$WORK/wheels" -r "$WORK/wheels.txt"
if [[ -n "${ENFLAME_VLLM_SOURCE_DIR:-}" ]]; then
  SOURCE="$ENFLAME_VLLM_SOURCE_DIR"
  # Refuse local modifications; building them would defeat the commit pin.
  test -z "$(git -C "$SOURCE" status --porcelain --untracked-files=normal)"
else
  SOURCE="$WORK/vllm"
  git clone --depth 1 --branch v0.32.0 \
    "${VLLM_SOURCE_URL:-https://github.com/vllm-project/vllm.git}" "$SOURCE"
fi
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$VLLM_COMMIT"
python -m pip install --index-url "$PYPI_INDEX" 'setuptools>=77.0.3,<81' 'setuptools-scm>=8' \
  'setuptools-rust>=1.9' wheel 'packaging>=26' jinja2 regex \
  'cloudpickle>=3.1.0' 'PyYAML==6.0.1' 'sqlalchemy>=1.4.31,<2.1' \
  'scikit-build-core==0.11' pybind11 ninja 'cmake>=3.26.1' \
  -c "$WORK/constraints.txt"
python -m pip install --no-deps --no-index --find-links "$WORK/wheels" \
  'triton-gcu==3.6.0+1.0.20260722' 'flagtree==0.7.0+enflame3.6' 'flag-gems==5.4.0'
python -m pip install --index-url "$PYPI_INDEX" \
  -r "$SOURCE/requirements/common.txt" -c "$WORK/constraints.txt"
python -m pip install --index-url "$PYPI_INDEX" \
  'openai==3.28.0' 'pandas==3.0.6' -c "$WORK/constraints.txt"
# A normal pip install replaces the old distribution; never manually delete it.
(cd "$SOURCE" && VLLM_TARGET_DEVICE=empty python -m pip install \
  --no-build-isolation --no-deps .)
# Common dependencies can overlay upstream Triton onto FlagTree's namespace.
python -m pip uninstall -y triton
python -m pip install --force-reinstall --no-deps --no-index \
  --find-links "$WORK/wheels" 'flagtree==0.7.0+enflame3.6'
python "${ROOT}/.github/scripts/enflame/verify_stack.py"
python "${ROOT}/docker/enflame/verify_compiler.py"

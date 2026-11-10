#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Prepare the vLLM 0.32 empty wheel in the FlagOS Kunlunxin Python environment.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
index_url="${INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
vendor_index=https://resource.flagos.net/repository/flagos-pypi-kunlunxin/simple
package_url=https://resource.flagos.net/repository/flagos-pypi-kunlunxin/packages
wheel_url="$package_url/vllm/0.32.0+flagos/vllm-0.32.0+flagos-py3-none-any.whl"
flagcx_url="$package_url/flagcx/0.14.1+xre5.37.1/flagcx-0.14.1+xre5.37.1-cp310-cp310-linux_x86_64.whl"
flagcx_sha=b49203511f7906aaa3cebdbf304aff521a86f0540f6b57c6af6154092f8a658e
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

uv pip install --python /flagos/bin/python --no-deps --index-url "$index_url" \
  pip wheel "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8"
mkdir -p "$work/original" "$work/patched"
/flagos/bin/python -m pip download --no-deps --dest "$work/original" "$wheel_url"
/flagos/bin/python "$here/python310_compat.py" "$work/original/"*.whl "$work/patched"

# This is the latest stable FlagCX wheel for XRE 5.37.1 / CPython 3.10.
/flagos/bin/python -m pip install --no-deps "$flagcx_url#sha256=$flagcx_sha"

# Preserve Torch, Triton, FlagGems and the pinned FlagCX wheel.
/flagos/bin/python -m pip freeze --all \
  | awk -F '==' 'tolower($1) ~ /^(torch|torchvision|torchaudio|flagtree|triton|flag-gems|flagcx)$/' \
  > "$work/runtime-constraints.txt"
indexes=(--index-url "$index_url" --extra-index-url "$vendor_index")
if [[ -n "${EXTRA_INDEX_URL:-}" ]]; then
  indexes+=(--extra-index-url "$EXTRA_INDEX_URL")
fi
/flagos/bin/python -m pip install --no-cache-dir "${indexes[@]}" \
  --constraint "$work/runtime-constraints.txt" "numpy<2" "$work/patched/"*.whl

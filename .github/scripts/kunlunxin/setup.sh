#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install the checkout into the Kunlunxin CI Runtime.
set -euo pipefail

if [[ -r /etc/bash_env.sh && ${BASH_ENV:-} != /etc/bash_env.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/bash_env.sh
fi

unset BASH_ENV VLLM_FL_FLAGOS_WHITELIST
export PATH="/flagos/bin:$PATH"
export PYTHONPATH=/opt/flagtree
export USE_FLAGGEMS="${USE_FLAGGEMS:-1}" VLLM_FL_PREFER="${VLLM_FL_PREFER:-flagos}"
: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

if [[ -n "${FLAGCX_PATH:-}" && -d "$FLAGCX_PATH/build/lib" ]]; then
  export LD_LIBRARY_PATH="$FLAGCX_PATH/build/lib:/opt/kunlun/lib:${LD_LIBRARY_PATH:-}"
fi

git config --global --add safe.directory "$PWD"
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check

python - <<'PYCODE'
from importlib.metadata import version
from pathlib import Path

import flag_gems  # noqa: F401
import torch
import triton
import vllm_fl
from vllm.platforms import current_platform

assert version("vllm") in {"0.32.0+flagos", "0.32.0+empty"}, version("vllm")
assert Path(vllm_fl.__file__).resolve().is_relative_to(Path.cwd()), vllm_fl.__file__
assert Path(triton.__file__).resolve().is_relative_to("/opt/flagtree"), triton.__file__
assert torch.cuda.is_available() and torch.cuda.device_count() >= 4
assert current_platform.device_type == "cuda", current_platform
assert current_platform.vendor_name == "kunlunxin", current_platform
print(f"Kunlunxin CI uses vLLM {version('vllm')} and {vllm_fl.__file__}")
PYCODE

if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf '%s\n' 'BASH_ENV=' 'VLLM_FL_FLAGOS_WHITELIST=' >> "$GITHUB_ENV"
  for name in PATH PYTHONPATH LD_LIBRARY_PATH FLAGCX_PATH USE_FLAGGEMS VLLM_FL_PREFER; do
    if [[ -n "${!name:-}" ]]; then
      printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
    fi
  done
fi

#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Enflame S60 CI.
set -euo pipefail

: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${TOPS_VISIBLE_DEVICES:?TOPS_VISIBLE_DEVICES is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    VLLM_PLUGINS \
    TOPS_VISIBLE_DEVICES \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS \
    TORCH_GCU_ENABLE_INT64_AND_UINT64 \
    ENABLE_I64_CHECK \
    TORCHGCU_INDUCTOR_ENABLE; do
    echo "${name}=${!name:-}" >> "${GITHUB_ENV}"
  done
fi

# Preserve the image's vendor SDK, vLLM and flaggems-vllm. Older CI image
# digests lack xdist, which tests/run.py and session-finish probes require.
# Resolve only this test tool (and execnet), never the project's runtime extras.
# The SDK's private pip index lacks xdist; override it for this command only.
python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple 'pytest-xdist>=3.0.0,<4'

# Reinstall only the checked-out plugin without resolving image dependencies.
unset PYTHONPATH
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo 'PYTHONPATH=' >> "${GITHUB_ENV}"
fi

python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import flag_gems
import torch
import torch_gcu  # noqa: F401
import pandas  # noqa: F401
import vllm
import vllm.tool_parsers.utils  # noqa: F401
import vllm_fl

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"FlagGems vendor: {getattr(flag_gems, 'vendor_name', 'auto-detected')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.gcu.is_available()}")
print(f"Accelerator count: {torch.gcu.device_count()}")
PY

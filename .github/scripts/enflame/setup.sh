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

# The image supplies the vendor SDK, not necessarily the target vLLM release.
# Install first, then enforce the same strict package gate as the Docker recipe.
bash .github/scripts/enflame/install_vllm.sh
python .github/scripts/enflame/verify_stack.py

# Preserve the installed runtime when installing the PR checkout.
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install --extra-index-url https://pypi.org/simple "pytest-xdist>=3.0.0"

python .github/scripts/install_flaggems_vllm.py

# Verify pinned release files after all package installation steps.
python docker/enflame/verify_compiler.py
python docker/enflame/verify_index_fill.py

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

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
    echo "${name}=${!name}" >> "${GITHUB_ENV}"
  done
fi

# Reject the old CI image before installing this PR checkout. Keep vendor
# torch and compiler packages pinned; editable plugin install uses --no-deps.
python - <<'PY'
from importlib.metadata import version

from packaging.version import Version

vllm_version = version("vllm")
parsed = Version(vllm_version)
if parsed.base_version != "0.28.0" or "empty" not in (parsed.local or "").split("."):
    raise RuntimeError(f"Expected vLLM 0.28.0 empty build, found {vllm_version}")

expected = {
    "torch": "2.11.0+cpu",
    "torch-gcu": "2.11.0+3.8.20260713",
    "flag-gems": "5.4.0",
    "flagtree": "0.7.0+enflame3.6",
    "triton-gcu": "3.6.0+1.0.20260722",
    "openai": "3.28.0",
    "pandas": "3.0.6",
}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise RuntimeError(f"Expected {package} {wanted}, found {actual}")
print("Enflame CI package versions match the vLLM 0.28 recipe")
PY

# The vendor runtime, vLLM, and FlagGems come from the pinned base image.
python -m pip install --no-build-isolation --no-deps -e .

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

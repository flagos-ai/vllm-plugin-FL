#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Setup script for Moore Threads MUSA CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${MTHREADS_VISIBLE_DEVICES:?MTHREADS_VISIBLE_DEVICES is not set}"

# Install a wheel so serving workers cannot prefer the image's old package.
env -u VLLM_VENDOR python -m pip install --no-build-isolation --no-deps .

# Install the shared runner's reporting, timeout and parallel-test plugins.
# Keep failure fatal after trying independent package indexes in order.
install_test_dependencies() {
    local index_url
    for index_url in \
        https://pypi.tuna.tsinghua.edu.cn/simple \
        https://pypi.org/simple \
        https://mirrors.aliyun.com/pypi/simple; do
        if python -m pip install --index-url "$index_url" \
            "pytest-cov==7.1.0" \
            "pytest-json-report==1.5.0" \
            "pytest-timeout==2.4.0" \
            "pytest-xdist>=3.0.0"; then
            return 0
        fi
    done
    return 1
}
install_test_dependencies

python .github/scripts/install_flaggems_vllm.py

python -I - <<'PY'
import flag_gems
import torch
import torch_musa
import vllm
import vllm_fl
from vllm.platforms import current_platform

assert torch.musa.is_available(), "MUSA accelerator is unavailable"
assert torch.musa.device_count() > 0, "No MUSA devices detected"
assert current_platform.device_type == "musa", current_platform.device_type

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"MUSA available: {torch.musa.is_available()}")
print(f"MUSA devices: {torch.musa.device_count()}")
print(f"Platform: {current_platform}")
PY

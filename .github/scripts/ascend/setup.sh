#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Use the dependency image built by docker/ascend/Dockerfile.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

git config --global --add safe.directory "$(pwd)"

VLLM_DIST_VERSION="$(python -c 'from importlib.metadata import version; print(version("vllm"))' 2>/dev/null || true)"

if [[ "${VLLM_DIST_VERSION}" == "0.32.0+flagos" ]]; then
    # FlagOS Runtime 2.3.0 (CANN 9.2.1) + vLLM 0.32.0+flagos stack
    # Preserve Runtime PYTHONPATH=/opt/flagtree compiler selection (do not pass -I).
    python -B "$HERE/stack.py" --runtime-only

    if [[ ! -e .git && -z "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL:-}" ]]; then
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="$(python -c 'from importlib.metadata import PackageNotFoundError, version
try:
    print(version("vllm-plugin-fl"))
except PackageNotFoundError:
    print("0.5.0.dev0")')"
        export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL
    fi

    env -u VLLM_VENDOR uv pip install \
        --python "$(python -c 'import sys; print(sys.executable)')" \
        --break-system-packages \
        --no-build-isolation \
        --no-deps \
        -e .

    python - <<'PY'
from importlib.metadata import version

for package in ("pytest", "pytest-xdist", "pytest-cov", "pytest-json-report", "pytest-timeout"):
    print(f"CI tool: {package}=={version(package)}")
PY

    python -B "$HERE/stack.py"
    exit 0
fi

export VLLM_USE_V2_MODEL_RUNNER=1
python docker/ascend/runtime_constraints.py verify /opt/ascend-stack \
    --vllm-version 0.32.0 --check-imports

python -m pip install --no-build-isolation --no-deps -e .
# Resolve test tools with the captured NPU stack overrides as well.
uv --config-file /opt/ascend-stack/uv.toml pip install \
    --python "$(command -v python)" 'pytest-xdist>=3.0.0'
python .github/scripts/install_ascend_flaggems_vllm.py
python docker/ascend/runtime_constraints.py verify /opt/ascend-stack \
    --vllm-version 0.32.0 --check-imports

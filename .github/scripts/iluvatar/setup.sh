#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install the checked-out plugin into the published Iluvatar CI image.
set -euo pipefail

export PATH="/opt/conda/bin:${PATH:-}"
# Do not let a baked-in checkout shadow the source under test.
unset PYTHONPATH

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo 'PYTHONPATH=' >> "${GITHUB_ENV}"
  for name in PATH VLLM_PLUGINS USE_FLAGGEMS GEMS_VENDOR; do
    if [[ -n "${!name:-}" ]]; then
      echo "${name}=${!name}" >> "${GITHUB_ENV}"
    fi
  done
fi

# FlagGems-vllm is supplied by the CI image. Never replace it during setup;
# missing packages indicate a broken image, not a request to install from Git.
python -c 'from importlib.metadata import version; print("Reusing image FlagGems-vllm", version("flaggems-vllm"))'
python -m pip install --no-build-isolation --no-deps -e .

# The shared unit and functional runners pass -n to pytest. Reuse the image's
# dependency when it satisfies CI; install only when it is absent or too old.
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
try:
    installed = Version(version("pytest-xdist"))
except (PackageNotFoundError, ValueError):
    installed = None
if installed is None or installed < Version("3.0.0"):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "pytest-xdist>=3.0.0"], check=True)
else:
    print(f"Reusing preinstalled pytest-xdist {installed}")
PY

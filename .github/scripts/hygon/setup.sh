#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Hygon DCU CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

export GEMS_VENDOR="${GEMS_VENDOR:-hygon}"
export DTK_HOME="${DTK_HOME:-/opt/dtk}"
export ROCM_PATH="${ROCM_PATH:-${DTK_HOME}}"
export HIP_PATH="${HIP_PATH:-${DTK_HOME}/hip}"
export HSA_PATH="${HSA_PATH:-${DTK_HOME}/hsa}"
export HIP_CLANG_PATH="${HIP_CLANG_PATH:-${DTK_HOME}/llvm/bin}"
export DEVICE_LIB_PATH="${DEVICE_LIB_PATH:-${DTK_HOME}/amdgcn/bitcode}"

DTK_PATH="${DTK_HOME}/bin:${HIP_PATH}/bin:${HIP_CLANG_PATH}"
DTK_LIBRARY_PATH="/opt/hyhal/lib/criu:/opt/hyhal/lib/rocprofiler:/opt/hyhal/lib:${HIP_PATH}/lib:${DTK_HOME}/lib:${DTK_HOME}/llvm/lib:${DTK_HOME}/dcc/lib:${DTK_HOME}/aillvm/lib:${HSA_PATH}/lib"
export PATH="${DTK_PATH}:${PATH}"
export LD_LIBRARY_PATH="${DTK_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    GEMS_VENDOR \
    DTK_HOME \
    ROCM_PATH \
    HIP_PATH \
    HSA_PATH \
    HIP_CLANG_PATH \
    DEVICE_LIB_PATH \
    PATH \
    LD_LIBRARY_PATH; do
    echo "${name}=${!name}" >> "${GITHUB_ENV}"
  done
fi

echo "DTK_HOME=${DTK_HOME}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
test -e "${HIP_PATH}/lib/libgalaxyhip.so.5"
test -e "${DTK_HOME}/llvm/lib/libomp.so"

TEST_DEPS=(
  pytest
  pytest-cov
  pytest-timeout
  pytest-json-report
  scikit-build-core==0.11
  pybind11
  ninja
  cmake
  numpy
  requests
  openai
  decorator
  pyyaml
  sqlalchemy
)

FLAGGEMS_REF="8d23621eb1381ae96b315a9287ce3cc555433824"
FLAGGEMS_SOURCE="${FLAGGEMS_PATH:-/workspace/FlagGems}"

if command -v uv >/dev/null 2>&1; then
  uv pip install --system --upgrade pip
  uv pip install --system --no-build-isolation -e . --no-deps
  uv pip install --system "${TEST_DEPS[@]}"
else
  python -m pip install --upgrade pip
  python -m pip install --no-build-isolation -e . --no-deps
  python -m pip install "${TEST_DEPS[@]}"
fi

test -d "${FLAGGEMS_SOURCE}/src/flag_gems"
git config --global --add safe.directory "${FLAGGEMS_SOURCE}"

FLAGGEMS_HEAD="$(git -C "${FLAGGEMS_SOURCE}" rev-parse HEAD)"
if [[ "${FLAGGEMS_HEAD}" != "${FLAGGEMS_REF}" ]]; then
  echo "Unexpected FlagGems commit: ${FLAGGEMS_HEAD}; expected ${FLAGGEMS_REF}."
  exit 1
fi

if ! grep -q 'kwargs.pop("num_ldmatrixes", None)' \
  "${FLAGGEMS_SOURCE}/src/flag_gems/runtime/configloader.py"; then
  echo "FlagGems num_ldmatrixes compatibility patch is missing."
  exit 1
fi

FLAGGEMS_DIR="$(mktemp -d)/FlagGems"
cp -a "${FLAGGEMS_SOURCE}" "${FLAGGEMS_DIR}"

if command -v uv >/dev/null 2>&1; then
  uv pip install --system --no-build-isolation -e "${FLAGGEMS_DIR}" --no-deps
else
  python -m pip install --no-build-isolation -e "${FLAGGEMS_DIR}" --no-deps
fi

python - <<'PY'
import flag_gems
import torch

print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.cuda.is_available()}")
print(f"Accelerator count: {torch.cuda.device_count()}")
PY

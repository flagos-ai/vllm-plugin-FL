#!/bin/bash
set -eo pipefail
phase=${1:?candidate or base}
source /work/source/.venv/bin/activate
set -u
export PYTHONPATH=/probe
export PYTHONDONTWRITEBYTECODE=1
export TRITON_F32_DEFAULT=ieee
export USE_FLAGCX=OFF TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLAGGEMS_VENV=/work/source/.venv FLAGGEMS_VENDOR=nvidia GITHUB_WORKSPACE=/probe
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CLANG=/toolchain/clang
export TRITON_CACHE_DIR="/probe/cache/$phase"
export XDG_CACHE_HOME="/probe/xdg/$phase"
export TMPDIR="/probe/tmp/$phase"
export FLAGGEMS_CACHE_DIR="/probe/gems/$phase" TORCH_EXTENSIONS_DIR="/probe/extensions/$phase"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p /probe/cache /probe/xdg /probe/tmp /probe/gems /probe/extensions
mkdir "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
mkdir "$FLAGGEMS_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
test ! -e /dev/nvidia0
test -z "${CUDA_VISIBLE_DEVICES}"
test "$NVIDIA_VISIBLE_DEVICES" == void
cd /probe
exec /work/source/.venv/bin/python /probe/compile-one.py "$phase"

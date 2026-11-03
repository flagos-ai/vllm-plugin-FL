#!/bin/bash
# Copyright 2026 FlagOS Contributors
set -euo pipefail

cd /opt/fl-build
fetch() {
    local url=$1 name=$2 digest=$3
    curl -fsSL --retry 5 "$url" -o "$name"
    printf '%s  %s\n' "$digest" "$name" | sha256sum -c -
}

fetch "$FLAGTREE_SOURCE_URL" flagtree.tar.gz \
    075eb5b382dc438e9a54b87d9de6895b5e22dd1c674c04379802cff6973501fb
fetch "$FLAGCX_SOURCE_URL" flagcx.tar.gz \
    b6af4fc90602af6adacf57cb3b7a7c752af8158d63b62adbe580b5b244fbf4ec
fetch "$FLAGTREE_LLVM_URL" llvm.tar.gz \
    7c1a2eb0a80c216f7e44e436e2293cd94bdaded31c1710013b8a4753995227f3
fetch "$GOOGLETEST_SOURCE_URL" googletest.tar.gz \
    65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
mkdir -p flagtree llvm googletest /opt/fl-wheels
tar -xzf flagtree.tar.gz --strip-components=1 -C flagtree
tar -xzf llvm.tar.gz --strip-components=1 -C llvm
tar -xzf googletest.tar.gz --strip-components=1 -C googletest
flagcx=/opt/fl-build/flagtree/third_party/tle/third_party/flagcx
mkdir -p "$flagcx"
tar -xzf flagcx.tar.gz --strip-components=1 -C "$flagcx"

# Build the exact FlagCX revision selected by FlagTree. The PyTorch NCCL wheel
# supplies libnccl.so.2; this temporary prefix supplies its build-time link name.
nccl_root=$(python -c 'import nvidia.nccl; print(next(iter(nvidia.nccl.__path__)))')
mkdir -p /opt/fl-build/nccl/lib
ln -s "$nccl_root/include" /opt/fl-build/nccl/include
ln -s "$nccl_root/lib/libnccl.so.2" /opt/fl-build/nccl/lib/libnccl.so
make -C "$flagcx/bindings/ir/nvidia"
# Image builds have no host driver mount. Link against the toolkit stub;
# its libcuda.so.1 dependency resolves to the real driver at container runtime.
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
make -C "$flagcx" -j"${FLAGTREE_MAX_JOBS}" USE_NVIDIA=1 CCL_HOME=/opt/fl-build/nccl
test -s "$flagcx/build/lib/libflagcx_device.bc"
test -s "$flagcx/build/lib/libflagcx.so"

cd /opt/fl-build/flagtree
# NVIDIA is the default backend. Preserve the upstream LLVM/CUDA/Proton/TLE
# build; only redirect its fixed SDK and GoogleTest inputs to verified caches.
unset FLAGTREE_BACKEND
export TRITON_HOME=/opt/fl-build/cache
export MAX_JOBS="$FLAGTREE_MAX_JOBS" TRITON_PARALLEL_LINK_JOBS=1
export LLVM_SYSPATH=/opt/fl-build/llvm JSON_SYSPATH=/usr
export TRITON_APPEND_CMAKE_ARGS=-DGOOGLETEST_DIR=/opt/fl-build/googletest
# Official archives lack .git. Use the upstream-supported version override;
# the complete source commit/archive/wheel hashes are also recorded in CI.
FLAGTREE_WHEEL_VERSION=0.7.0+git317a0249 \
    python -m pip wheel --no-deps --no-build-isolation -v \
        --wheel-dir /opt/fl-wheels .

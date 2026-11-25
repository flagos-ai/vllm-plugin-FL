#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install this checkout without changing the image's native vLLM CUDA/FL runtime.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

export PYTHONPATH=/opt/triton

# Pin flaggems-vllm to the validated v0.2.0 release. The base image ships
# 0.3.0.dev20261119 which triggers illegal-memory-access crashes during
# CUDA-graph profiling on H20 with Qwen3.6-35B-A3B MoE (TMA descriptor
# fault in mm, and a second fault in the fused_silu activation path).
FLAGGEMS_VLLM_COMMIT=b5c2c2249319277ccda4d41aa0f65164bd9b2a19
pip install --no-deps --no-build-isolation \
    "flaggems-vllm @ git+https://github.com/flagos-ai/FlagGems-vllm.git@${FLAGGEMS_VLLM_COMMIT}"

# The test extras include pytest, pytest-xdist, and the audio dependencies.
pip install --no-build-isolation -e ".[test]"

# The dsw-hosted runners cap /dev/shm at ~64MB regardless of --ipc=host /
# --shm-size (verified: both flags reach docker create yet df still shows
# 63M). tmpfs charges 4KB per POSIX semaphore file, and the unit pass's 16
# xdist workers x flag_gems' ~1k libentry locks exceed ~15.7K files. The
# job container is privileged, so remount a private 32g tmpfs — but ONLY
# when the inherited shm is too small for the burst; hosts with a real
# large shm (a100 pool) are left untouched.
shm_kb=$(df -k --output=avail /dev/shm | tail -1 | tr -d ' ')
if [ "${shm_kb:-0}" -lt 1048576 ] 2>/dev/null; then
    mount -t tmpfs -o size=32g,mode=1777 tmpfs /dev/shm
    echo "remounted /dev/shm at 32g (inherited avail was ${shm_kb}KB)"
    df -h /dev/shm | tail -1
fi

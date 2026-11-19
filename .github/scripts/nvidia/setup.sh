#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install this checkout without changing the image's native vLLM CUDA/FL runtime.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

export PYTHONPATH=/opt/triton

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

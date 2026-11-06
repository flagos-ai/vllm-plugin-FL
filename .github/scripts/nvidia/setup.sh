#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install this checkout without changing the image's native vLLM CUDA/FL runtime.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

git config --global --add safe.directory "$(pwd)"

# Refresh uv before use: published images may predate GHSA-hq85-5cpj-mr62.
# Install only this tool so the pinned vLLM/FlagGems/FlagTree stack is preserved.
UV_REQUIREMENT=$(grep '^uv==' "$HERE/../../../docker/cuda/requirements-ci.txt")
python -m pip install --break-system-packages --no-deps "$UV_REQUIREMENT"

# The shared installer pins its Git source and preserves the runtime packages.
python "$HERE/../install_flaggems_vllm.py"

python -I -B "$HERE/stack.py" --runtime-only

# Reuse the official CUDA wheel and install only the plugin checkout.
env -u VLLM_VENDOR python -m uv pip install \
    --python "$(python -c 'import sys; print(sys.executable)')" \
    --break-system-packages \
    --no-build-isolation \
    --no-deps \
    -e .

# The 0.32 base image predates the harness's xdist unit pass (-n N); the
# other pytest plugins (cov/json-report/timeout) are already present.
# pip, not uv: the image's pip index is a reachable mirror while uv ignores
# pip.conf and dies connecting to files.pythonhosted.org on some runners.
python -m pip install --break-system-packages "pytest-xdist>=3.0.0"

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

python -I -B "$HERE/stack.py"

python - <<'PY'
import av
import soundfile
import soxr
from vllm.assets.audio import AudioAsset
from xgboost import XGBRanker

XGBRanker()
for asset_name in ("mary_had_lamb", "winning_call"):
    audio, sample_rate = AudioAsset(asset_name).audio_and_sample_rate
    assert audio.size > 0, asset_name
    assert sample_rate > 0, asset_name

print(
    "Audio: "
    f"av={av.__version__}, soundfile={soundfile.__version__}, "
    f"soxr={soxr.__version__}"
)
PY

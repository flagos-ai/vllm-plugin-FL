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

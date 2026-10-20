#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install this checkout without changing the image's vLLM EMPTY/FL runtime.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

git config --global --add safe.directory "$(pwd)"

python -I -B "$HERE/stack.py" --runtime-only

# EMPTY vLLM ships no CUDA extension. Build the plugin-owned extension needed
# by its custom operators and graph paths instead.
VLLM_VENDOR=cuda MAX_JOBS="${MAX_JOBS:-4}" uv pip install \
    --python "$(python -c 'import sys; print(sys.executable)')" \
    --break-system-packages \
    --no-build-isolation \
    --no-deps \
    -e .

python .github/scripts/install_flaggems_vllm.py

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

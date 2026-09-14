#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

uv pip install \
    --system \
    --break-system-packages \
    --no-build-isolation \
    --no-deps \
    -e .

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

import av
import flag_gems
import soundfile
import soxr
import triton
import vllm
from vllm.assets.audio import AudioAsset
from xgboost import XGBRanker

assert vllm.__version__.startswith("0.28.0"), vllm.__version__
assert version("flagtree") == "0.6.2a1", version("flagtree")
assert triton.__version__.startswith("3.6.0"), triton.__version__
try:
    standalone_triton = version("triton")
except PackageNotFoundError:
    standalone_triton = None
assert standalone_triton is None, standalone_triton
assert version("xgboost") == "3.3.0", version("xgboost")

XGBRanker()
for asset_name in ("mary_had_lamb", "winning_call"):
    audio, sample_rate = AudioAsset(asset_name).audio_and_sample_rate
    assert audio.size > 0, asset_name
    assert sample_rate > 0, asset_name

print(f"vLLM: {vllm.__version__}")
print(f"FlagTree: {version('flagtree')}")
print(f"Triton: {triton.__version__} (provided by FlagTree)")
print(f"FlagGems: {flag_gems.__version__}")
print(f"XGBoost: {version('xgboost')}")
print(
    "Audio: "
    f"av={av.__version__}, soundfile={soundfile.__version__}, "
    f"soxr={soxr.__version__}"
)
PY

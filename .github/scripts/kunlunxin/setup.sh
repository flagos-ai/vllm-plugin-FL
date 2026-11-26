#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install the checkout into the Kunlunxin CI Runtime.
set -euo pipefail

export USE_FLAGGEMS="${USE_FLAGGEMS:-1}" VLLM_FL_PREFER="${VLLM_FL_PREFER:-flagos}"
: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

if [[ -n "${FLAGCX_PATH:-}" && -d "$FLAGCX_PATH/build/lib" ]]; then
  export LD_LIBRARY_PATH="$FLAGCX_PATH/build/lib:/opt/kunlun/lib:${LD_LIBRARY_PATH:-}"
fi

git config --global --add safe.directory "$PWD"
# Runner V2 needs the P800 sampling and grammar providers.
CI_FLAGGEMS_VLLM_VENDOR=kunlunxin python .github/scripts/install_flaggems_vllm.py --source-archive
# Rebuild the native extension for the checkout used by this test run.
torch_package="$(python -c 'from importlib.util import find_spec; print(find_spec("torch").submodule_search_locations[0])')"
torch_dir="$torch_package/share/cmake/Torch"
test -f "$torch_dir/TorchConfig.cmake"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
VLLM_VENDOR=cuda PATH="$cuda_home/bin:$PATH" CUDACXX="$cuda_home/bin/nvcc" \
  CMAKE_ARGS="-DTorch_DIR=$torch_dir" \
  python -m pip install --no-build-isolation --no-deps -e .
python -m pip check

python - <<'PYCODE'
from importlib.metadata import version
from pathlib import Path

import flag_gems  # noqa: F401
import torch
import triton
import vllm_fl
import vllm_fl._C  # noqa: F401
from vllm.platforms import current_platform

assert version("vllm") in {"0.32.0+flagos", "0.32.0+empty"}, version("vllm")
assert Path(vllm_fl.__file__).resolve().is_relative_to(Path.cwd()), vllm_fl.__file__
assert Path(triton.__file__).resolve().is_relative_to("/opt/flagtree"), triton.__file__
assert torch.cuda.is_available() and torch.cuda.device_count() >= 4
assert current_platform.device_type == "cuda", current_platform
assert current_platform.vendor_name == "kunlunxin", current_platform
print(f"Kunlunxin CI uses vLLM {version('vllm')} and {vllm_fl.__file__}")
PYCODE

# vllm_fl._C registers weak_ref_tensor, which suppresses the empty-wheel op
# schemas whenever the extension is importable. The extension is validated
# above; the test run uses the empty-wheel path, so drop the built .so.
python - <<'PYCODE'
import importlib.util
from pathlib import Path

for so in Path("vllm_fl").glob("_C*.so"):
    so.unlink()
assert importlib.util.find_spec("vllm_fl._C") is None, "vllm_fl._C is still importable"
PYCODE

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in LD_LIBRARY_PATH FLAGCX_PATH USE_FLAGGEMS VLLM_FL_PREFER; do
    if [[ -n "${!name:-}" ]]; then
      printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
    fi
  done
fi

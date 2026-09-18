#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for MetaX C550 CI environment.
set -euo pipefail

export PATH="/opt/conda/bin:${PATH}"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH \
    GEMS_VENDOR \
    VLLM_PLUGINS; do
    echo "${name}=${!name}" >> "${GITHUB_ENV}"
  done
fi

# vLLM, FlagGems, and test dependencies are provided by the CI image.
# Only install the checked-out plugin source for this workflow run.
python -m pip install --no-build-isolation --no-deps -e .

# Patch FlagGems LibTuner to fall back to best_config on cache KeyError.
# TODO: remove when FlagGems fixes kernel hash stability in LibTuner.
python - <<'PATCH'
import pathlib

libentry = pathlib.Path("/workspace/FlagGems/src/flag_gems/utils/libentry.py")
src = libentry.read_text()

old = (
    "                    **self.nargs,\n"
    "                    **kwargs,\n"
    "                    **self.cache[key].all_kwargs(),\n"
    "                }\n"
    "                self.pre_hook(full_nargs, reset_only=True)\n"
    "                self.configs_timings = timings\n"
    "            config = self.cache[key]"
)

new = (
    "                    **self.nargs,\n"
    "                    **kwargs,\n"
    "                    **best_config.all_kwargs(),\n"
    "                }\n"
    "                self.pre_hook(full_nargs, reset_only=True)\n"
    "                self.configs_timings = timings\n"
    "            # Fallback: if cache read-back fails due to kernel hash mismatch,\n"
    "            # use best_config directly.\n"
    "            # TODO: remove when FlagGems fixes kernel hash stability in LibTuner.\n"
    "            try:\n"
    "                config = self.cache[key]\n"
    "            except KeyError:\n"
    "                config = best_config"
)

if old in src:
    libentry.write_text(src.replace(old, new, 1))
    print("FlagGems libentry.py patched OK")
elif new in src:
    print("FlagGems libentry.py already patched, skipping")
else:
    raise RuntimeError("FlagGems libentry.py patch target not found — check if upstream changed")
PATCH

python - <<'PY'
import flag_gems
import torch
import vllm
import vllm_fl

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.cuda.is_available()}")
print(f"Accelerator count: {torch.cuda.device_count()}")
PY

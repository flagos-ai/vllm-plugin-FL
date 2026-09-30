#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Hygon DCU CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${ROCM_PATH:?ROCM_PATH is not set}"
: "${HIP_PATH:?HIP_PATH is not set}"
: "${LD_LIBRARY_PATH:?LD_LIBRARY_PATH is not set}"

# Optional vars - use if set, otherwise use defaults
DTK_HOME="${DTK_HOME:-${DTKROOT:-/opt/dtk}}"

unset VLLM_FL_IMAGE_PLUGIN_ROOT
unset HYGON_USE_IMAGE_PLUGIN

echo "DTK_HOME=${DTK_HOME}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
test -e "${HIP_PATH}/lib/libgalaxyhip.so.5"
test -e "${DTK_HOME}/llvm/lib/libomp.so"

python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
from importlib import metadata

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

import vllm

installed_version = Version(metadata.version("vllm"))
imported_version = Version(vllm.__version__)
print(f"vLLM installed version: {installed_version}")
print(f"vLLM imported version: {imported_version}, path: {vllm.__file__}")

vllm_requirements = []
for raw_requirement in metadata.requires("vllm-plugin-fl") or []:
    requirement = Requirement(raw_requirement)
    if canonicalize_name(requirement.name) != "vllm":
        continue
    if requirement.marker is None or requirement.marker.evaluate({"extra": "test"}):
        vllm_requirements.append(requirement)

if not vllm_requirements:
    raise SystemExit(
        "ERROR: vllm-plugin-fl metadata has no vllm requirement for the test extra"
    )

for requirement in vllm_requirements:
    specifier = SpecifierSet(str(requirement.specifier))
    if not str(specifier):
        raise SystemExit(f"ERROR: vLLM requirement has no version constraint: {requirement}")
    print(f"Required vLLM version: {specifier}")
    if installed_version not in specifier or imported_version not in specifier:
        raise SystemExit(
            f"ERROR: vLLM installed={installed_version}, imported={imported_version} "
            f"does not satisfy vllm-plugin-fl requirement {requirement}"
        )

import flag_gems
import torch
import vllm_fl

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.cuda.is_available()}")
print(f"Accelerator count: {torch.cuda.device_count()}")
PY

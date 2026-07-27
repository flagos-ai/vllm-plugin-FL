#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Hygon DCU CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${DTK_HOME:?DTK_HOME is not set}"
: "${ROCM_PATH:?ROCM_PATH is not set}"
: "${HIP_PATH:?HIP_PATH is not set}"
: "${HSA_PATH:?HSA_PATH is not set}"
: "${HIP_CLANG_PATH:?HIP_CLANG_PATH is not set}"
: "${DEVICE_LIB_PATH:?DEVICE_LIB_PATH is not set}"
: "${LD_LIBRARY_PATH:?LD_LIBRARY_PATH is not set}"

USE_IMAGE_PLUGIN="${HYGON_USE_IMAGE_PLUGIN:-1}"

if [[ "${USE_IMAGE_PLUGIN}" == "1" ]]; then
    IMAGE_PLUGIN_ROOT="${VLLM_FL_IMAGE_PLUGIN_ROOT:-/opt/vllm-src/vllm-plugin-FL}"
    test -d "${IMAGE_PLUGIN_ROOT}/vllm_fl"

    # The Hygon CI image already contains the validated plugin commit. Keep the
    # checkout available for tests and configs, but load vllm_fl from the image.
    HYGON_SITE_DIR="${RUNNER_TEMP:-/tmp}/hygon-python-site"
    mkdir -p "${HYGON_SITE_DIR}"
    cat > "${HYGON_SITE_DIR}/sitecustomize.py" <<'PY'
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path


class _ImageVllmFLFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.package_dir = Path(root) / "vllm_fl"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm_fl":
            return None
        init_file = self.package_dir / "__init__.py"
        if not init_file.exists():
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            init_file,
            submodule_search_locations=[str(self.package_dir)],
        )


_root = os.environ.get("VLLM_FL_IMAGE_PLUGIN_ROOT")
if _root:
    sys.meta_path.insert(0, _ImageVllmFLFinder(_root))
PY

    export VLLM_FL_IMAGE_PLUGIN_ROOT="${IMAGE_PLUGIN_ROOT}"
    export PYTHONPATH="${HYGON_SITE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

    if [[ -n "${GITHUB_ENV:-}" ]]; then
        {
            echo "VLLM_FL_IMAGE_PLUGIN_ROOT=${VLLM_FL_IMAGE_PLUGIN_ROOT}"
            echo "PYTHONPATH=${PYTHONPATH}"
        } >> "${GITHUB_ENV}"
    fi
else
    unset VLLM_FL_IMAGE_PLUGIN_ROOT
fi

export HYGON_USE_IMAGE_PLUGIN="${USE_IMAGE_PLUGIN}"

echo "DTK_HOME=${DTK_HOME}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "HYGON_USE_IMAGE_PLUGIN=${HYGON_USE_IMAGE_PLUGIN}"
if [[ "${USE_IMAGE_PLUGIN}" == "1" ]]; then
    echo "VLLM_FL_IMAGE_PLUGIN_ROOT=${VLLM_FL_IMAGE_PLUGIN_ROOT}"
    echo "PYTHONPATH=${PYTHONPATH}"
fi
test -e "${HIP_PATH}/lib/libgalaxyhip.so.5"
test -e "${DTK_HOME}/llvm/lib/libomp.so"

python - <<'PY'
import os
from pathlib import Path

import flag_gems
import torch
import vllm
import vllm_fl

if os.environ.get("HYGON_USE_IMAGE_PLUGIN", "1") == "1":
    image_plugin_root = Path(os.environ["VLLM_FL_IMAGE_PLUGIN_ROOT"]).resolve()
    expected_package = image_plugin_root / "vllm_fl"
    plugin_file = Path(vllm_fl.__file__).resolve()
    if (
        plugin_file != expected_package / "__init__.py"
        and expected_package not in plugin_file.parents
    ):
        raise RuntimeError(
            f"Unexpected vllm_fl path: {plugin_file}; expected under {expected_package}"
        )

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.cuda.is_available()}")
print(f"Accelerator count: {torch.cuda.device_count()}")
PY

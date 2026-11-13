#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install the checked-out Hygon plugin as an ordinary wheel.
set -euo pipefail

REPO_ROOT="$(pwd -P)"
VERIFY_SCRIPT="${REPO_ROOT}/.github/scripts/hygon/verify_install.py"
git config --global --add safe.directory "${REPO_ROOT}"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
DTK_HOME="${DTK_HOME:-${DTKROOT:-/opt/dtk}}"
export ROCM_PATH="${ROCM_PATH:-${DTK_HOME}}"
export HIP_PATH="${HIP_PATH:-${DTK_HOME}/hip}"
: "${LD_LIBRARY_PATH:?LD_LIBRARY_PATH is not set}"
[[ "${GEMS_VENDOR}" == hygon ]]
[[ "${VLLM_PLUGINS}" == fl ]]
test -e "${HIP_PATH}/lib/libgalaxyhip.so.5"
test -e "${DTK_HOME}/llvm/lib/libomp.so"

# The validated Hygon image supplies native libraries, Torch, Gems and Tree.
# Do not compile CUDA extensions or resolve/upgrade the vendor dependencies.
unset VLLM_VENDOR VLLM_FL_IMAGE_PLUGIN_ROOT HYGON_USE_IMAGE_PLUGIN
unset PYTHONPATH PYTHONHOME
unset SETUPTOOLS_SCM_PRETEND_VERSION SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL
export GEMS_VENDOR=hygon VLLM_PLUGINS=fl
PLUGIN_SOURCE_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
export PLUGIN_SOURCE_SHA

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vllm-fl-hygon.XXXXXXXX")"
trap 'rm -rf -- "${BUILD_DIR}"' EXIT
cd "${BUILD_DIR}"
python -I -B "${VERIFY_SCRIPT}" --write-vendor-snapshot "${BUILD_DIR}/vendor-before.json"
python -I -B -m pip wheel \
    --no-build-isolation --no-deps --no-cache-dir \
    --wheel-dir "${BUILD_DIR}/wheels" "${REPO_ROOT}"
shopt -s nullglob
WHEELS=("${BUILD_DIR}"/wheels/vllm_plugin_fl-*.whl)
[[ "${#WHEELS[@]}" -eq 1 ]]
python -I -B -m pip install \
    --force-reinstall --no-build-isolation --no-deps --no-cache-dir \
    "${WHEELS[0]}"
python -I -B -m pip install "pytest-xdist>=3.0.0"

# The shared installer compares direct_url.json's raw URL with the canonical
# repository and re-clones when they differ.  The Hygon CI image is built
# through a Git mirror, so the recorded URL keeps a mirror prefix even though
# the pinned commit is installed, and github.com is unreachable from the
# runner.  Accept the pinned commit here and only fall back to the shared
# installer when the release is genuinely absent.
FLAGGEMS_VLLM_INSTALLER="${REPO_ROOT}/.github/scripts/install_flaggems_vllm.py"
if ! python -I -B - "${FLAGGEMS_VLLM_INSTALLER}" <<'PY'
import ast
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution

# Read COMMIT / RELEASE as literals instead of importing the installer:
# exec_module would run its top-level code (network probes, logging setup,
# file writes) on every CI run even when no reinstall is needed.
consts = {}
with open(sys.argv[1], encoding="utf-8") as fh:
    tree = ast.parse(fh.read(), filename=sys.argv[1])
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in ("COMMIT", "RELEASE"):
            try:
                consts[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass

commit = consts.get("COMMIT")
release = consts.get("RELEASE")
if not commit or not release:
    raise SystemExit(1)

try:
    direct_url = distribution("flaggems-vllm").read_text("direct_url.json")
    source = json.loads(direct_url or "{}")
except (PackageNotFoundError, ValueError):
    raise SystemExit(1)

if (
    source.get("vcs_info", {}).get("commit_id") != commit
    or "flagos-ai/FlagGems-vllm" not in source.get("url", "")
):
    raise SystemExit(1)

print(f"FlagGems-vllm {release} (commit {commit}) already installed")
PY
then
    python -I -B "${FLAGGEMS_VLLM_INSTALLER}"
fi
python -I -B "${VERIFY_SCRIPT}" \
    --compare-vendor-snapshot "${BUILD_DIR}/vendor-before.json" \
    --min-devices "${HYGON_MIN_DEVICES:-2}"

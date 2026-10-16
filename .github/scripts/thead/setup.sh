#!/bin/bash
# Copyright 2026 FlagOS Contributors
# The managed image supplies the fixed SDK/Torch/empty-vLLM/Gems/Tree stack
# and runtime/test/build dependencies. Never resolve upstream Torch 2.13.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../../.." && pwd)"
BASE_PYTHON=python3
if [[ -n "${THEAD_BASE_PYTHON:-}" ]]; then BASE_PYTHON="$THEAD_BASE_PYTHON"; fi
if [[ -n "${THEAD_CI_WORK_DIR:-}" ]]; then
  test ! -e "$THEAD_CI_WORK_DIR" || { echo "Refusing to reuse CI environment" >&2; exit 1; }
  mkdir -p "$THEAD_CI_WORK_DIR"
  WORK="$(cd -- "$THEAD_CI_WORK_DIR" && pwd)"
else
  WORK="$(mktemp -d /tmp/thead-ci.XXXXXXXX)"
fi
export THEAD_CI_WORK_DIR="$WORK"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf 'THEAD_CI_WORK_DIR=%s\n' "$WORK" >> "$GITHUB_ENV"
fi
export PIP_CONFIG_FILE=/dev/null
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_VENDOR="" VLLM_TARGET_DEVICE=empty
export VLLM_PLUGINS=fl GEMS_VENDOR=thead USE_FLAGGEMS=1
unset FL_BACKEND PYTHONPATH PYTHONHOME
export HOME="$WORK/home" XDG_CACHE_HOME="$WORK/cache"
export FLAGGEMS_CACHE_DIR="$WORK/cache/flaggems" TRITON_CACHE_DIR="$WORK/cache/triton"
export HF_HOME="$WORK/cache/huggingface"
TMPDIR="$(mktemp -d /tmp/thead-build.XXXXXXXX)"
export TMPDIR
mkdir -p "$HOME" "$TMPDIR" "$FLAGGEMS_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME" "$WORK/wheels"
"$BASE_PYTHON" -I -B "$HERE/stack.py" snapshot > "$WORK/vendor-before.json"
# A child venv does not inherit its parent venv's packages. Bind only the
# image-owned ordinary package directory, after rejecting editable providers.
"$BASE_PYTHON" -I -B - "$WORK" <<'PY'
import json
import sys
import sysconfig
from pathlib import Path
site = Path(sysconfig.get_path("purelib")).resolve()
if sys.prefix == sys.base_prefix or not site.is_dir():
    raise RuntimeError("THEAD_BASE_PYTHON must select the image-owned ordinary venv")
if any(site.glob("__editable__*")) or any(site.glob("*.egg-link")):
    raise RuntimeError("Editable packages are not allowed in the CI image")
for direct in site.glob("*.dist-info/direct_url.json"):
    data = json.loads(direct.read_text())
    if data.get("dir_info", {}).get("editable"):
        raise RuntimeError("Editable distribution is not allowed in the CI image")
(Path(sys.argv[1]) / "base-site.json").write_text(json.dumps({"site": str(site)}))
PY
"$BASE_PYTHON" -I -B -m venv --system-site-packages "$WORK/env"
PY="$WORK/env/bin/python"
export THEAD_CI_PYTHON="$PY" VIRTUAL_ENV="$WORK/env"
export PATH="$VIRTUAL_ENV/bin:$PATH"
"$PY" -I -B - "$WORK/base-site.json" <<'PY'
import json
import sys
import sysconfig
from pathlib import Path
base = Path(json.loads(Path(sys.argv[1]).read_text())["site"])
site = Path(sysconfig.get_path("purelib"))
if site.resolve() == base:
    raise RuntimeError("CI child venv was not created")
(site / "00_thead_image_site.pth").write_text(str(base) + "\n")
PY

# Build and install an ordinary wheel of this checkout in the fresh venv only.
"$PY" -I -B -m pip --isolated --disable-pip-version-check --no-cache-dir wheel \
  --no-index --no-deps --no-build-isolation --wheel-dir "$WORK/wheels" "$REPO"
"$PY" -I -S -B - "$WORK/wheels" <<'PY' > "$WORK/plugin-wheel.txt"
import sys
from pathlib import Path
wheels = list(Path(sys.argv[1]).glob("vllm_plugin_fl-*.whl"))
if len(wheels) != 1:
    raise RuntimeError("Expected one newly built plugin wheel")
print(wheels[0])
PY
"$PY" -I -B -m pip --isolated --disable-pip-version-check --no-cache-dir install \
  --ignore-installed --no-index --no-deps --no-compile "$(cat "$WORK/plugin-wheel.txt")"
"$PY" -I -B "$HERE/stack.py" check --compare "$WORK/vendor-before.json" > "$WORK/stack.json"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in THEAD_CI_WORK_DIR THEAD_CI_PYTHON VIRTUAL_ENV PATH HOME XDG_CACHE_HOME \
    FLAGGEMS_CACHE_DIR TRITON_CACHE_DIR HF_HOME TMPDIR PIP_CONFIG_FILE \
    TORCH_DEVICE_BACKEND_AUTOLOAD VLLM_PLUGINS GEMS_VENDOR USE_FLAGGEMS \
    VLLM_VENDOR VLLM_TARGET_DEVICE VLLM_WORKER_MULTIPROC_METHOD; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
echo "THead CI Python: $PY"

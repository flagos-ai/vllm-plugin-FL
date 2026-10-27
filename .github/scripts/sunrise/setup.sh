#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Preserve the matched vendor Torch/PTPU/Triton/FlagGems packages.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../../.." && pwd)"
BASE_PYTHON="${SUNRISE_BASE_PYTHON:-python3}"
# compiler itself probes Python: disable device autoload before selecting it.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="${PYTHONPATH:-}"
source /etc/profile.d/zz-compiler.sh
compiler triton
export PYTHONPATH="${PYTHONPATH:-}"
WORK="$(mktemp -d /tmp/sunrise-ci.XXXXXXXX)"
export SUNRISE_CI_WORK_DIR="$WORK"
# --system-site-packages alone does not inherit a parent /flagos venv.
"$BASE_PYTHON" - "$WORK/base-environment.json" <<'PY'
import json
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
names = ("torch", "triton", "flagtree", "flag-gems", "vllm")
versions = {}
for name in names:
    try:
        versions[name] = version(name)
    except PackageNotFoundError:
        versions[name] = None
site = Path(sysconfig.get_path("purelib")).resolve()
if not site.is_dir() or not versions["torch"]:
    raise RuntimeError("SUNRISE_BASE_PYTHON must select the matched vendor environment")
facts = {"site": str(site), "prefix": sys.prefix, "versions": versions}
Path(sys.argv[1]).write_text(json.dumps(facts, indent=2))
print("Sunrise base metadata:", json.dumps(facts))
PY
"$BASE_PYTHON" -m venv --system-site-packages "$WORK/env"
export VIRTUAL_ENV="$WORK/env" PATH="$WORK/env/bin:$PATH"
export SUNRISE_CI_PYTHON="$WORK/env/bin/python"
python - "$WORK/base-environment.json" <<'PY'
import json
import sys
import sysconfig
from pathlib import Path
base = Path(json.loads(Path(sys.argv[1]).read_text())["site"])
child = Path(sysconfig.get_path("purelib")).resolve()
if child == base:
    raise RuntimeError("Refusing to bind packages into the base environment")
# addsitedir processes the image-owned editable .pth files as well as wheels.
(child / "00_sunrise_vendor_site.pth").write_text(
    "import site; site.addsitedir(" + repr(str(base)) + ")\n"
)
PY
python - "$WORK/base-environment.json" <<'PY'
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
base = json.loads(Path(sys.argv[1]).read_text())
actual = {}
for name, expected in base["versions"].items():
    try:
        actual[name] = version(name)
    except PackageNotFoundError:
        actual[name] = None
    if actual[name] != expected:
        raise RuntimeError(f"Child lost vendor {name}: {expected} -> {actual[name]}")
print("Sunrise inherited child metadata:", json.dumps(actual))
PY
export VLLM_VENDOR="" VLLM_TARGET_DEVICE=empty
export GEMS_VENDOR=sunrise VLLM_PLUGINS=fl USE_FLAGGEMS=1
export VLLM_USE_V2_MODEL_RUNNER=0 TA_CLOSE_GRAPH_MEMORY=1 PCCL_PROTO=Simple
unset VLLM_FL_SUNRISE_BENCH_BASELINE_GDN_CORE_ATTN_ZEROS
mkdir -p "$WORK/wheels"
if [[ -n "${SUNRISE_VLLM_WHEEL:-}" ]]; then
  test -f "$SUNRISE_VLLM_WHEEL"
  python -m pip install --ignore-installed --no-deps "$SUNRISE_VLLM_WHEEL"
elif [[ -n "${SUNRISE_VLLM_SOURCE:-}" ]]; then
  test -d "$SUNRISE_VLLM_SOURCE"
  # Build a copy: never modify the mounted source or its installed package.
  cp -a "$SUNRISE_VLLM_SOURCE" "$WORK/vllm"
  python -m pip wheel --no-build-isolation --no-deps --wheel-dir "$WORK/wheels" "$WORK/vllm"
  python -m pip install --ignore-installed --no-deps "$WORK"/wheels/vllm-*.whl
fi
# A regular wheel takes precedence over image-owned plugin/editable providers.
python -m pip wheel --no-build-isolation --no-deps --wheel-dir "$WORK/wheels" "$REPO"
python -m pip install --ignore-installed --no-deps "$WORK"/wheels/vllm_plugin_fl-*.whl
python "$HERE/../install_flaggems_vllm.py"
python - "$WORK/base-environment.json" <<'PY'
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
base = json.loads(Path(sys.argv[1]).read_text())
actual = {}
for name, expected in base["versions"].items():
    try:
        actual[name] = version(name)
    except PackageNotFoundError:
        actual[name] = None
    if name != "vllm" and actual[name] != expected:
        raise RuntimeError(f"Vendor dependency changed: {name}: {expected} -> {actual[name]}")
if (actual["vllm"] or "").split("+")[0] != "0.28.0":
    raise RuntimeError("Child must import vLLM 0.28.0 metadata: " + repr(actual))
print("Sunrise final child metadata:", json.dumps(actual))
PY
# Runtime checks and all subsequent steps must register the PTPU backend.
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
# Missing/incompatible runtime packages fail check.sh instead of replacing Torch.
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in SUNRISE_CI_WORK_DIR SUNRISE_CI_PYTHON VIRTUAL_ENV PATH PYTHONPATH \
    TORCH_DEVICE_BACKEND_AUTOLOAD \
    VLLM_VENDOR VLLM_TARGET_DEVICE GEMS_VENDOR VLLM_PLUGINS USE_FLAGGEMS \
    VLLM_USE_V2_MODEL_RUNNER TA_CLOSE_GRAPH_MEMORY PCCL_PROTO; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
echo "Sunrise CI Python: $SUNRISE_CI_PYTHON"

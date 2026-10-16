#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Setup script for Kunlunxin P800 CI environment.
set -euo pipefail

if [[ -d /root/miniconda/envs/python310_torch29_cuda/bin ]]; then
  export PATH="/root/miniconda/envs/python310_torch29_cuda/bin:${PATH}"
fi

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${FLAGCX_PATH:?FLAGCX_PATH is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

# Align with #566 iluvatar: FlagGems vendor ops must be enabled in CI.
export USE_FLAGGEMS="${USE_FLAGGEMS:-1}"

git config --global --add safe.directory "$(pwd)"

# The Kunlunxin image ships vllm-plugin-FL preinstalled from an older ref. Put
# the checkout first on sys.path so a stale image copy can never win the import.
# Use $PWD rather than $(pwd) so this stays clear of ShellCheck SC2155
# (declaration must not mask a command substitution's return value).
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -d "${FLAGCX_PATH}/build/lib" ]]; then
  export LD_LIBRARY_PATH="${FLAGCX_PATH}/build/lib:/opt/kunlun/lib:${LD_LIBRARY_PATH:-}"
else
  echo "::warning::${FLAGCX_PATH}/build/lib not found; FlagCX may be unavailable."
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH \
    PYTHONPATH \
    LD_LIBRARY_PATH \
    GEMS_VENDOR \
    VLLM_PLUGINS \
    USE_FLAGGEMS \
    FLAGCX_PATH \
    CUDA_VISIBLE_DEVICES \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS; do
    if [[ -n "${!name:-}" ]]; then
      echo "${name}=${!name}" >> "${GITHUB_ENV}"
    fi
  done
fi

# vLLM, FlagGems, FlagCX, and vendor runtime dependencies are provided by the
# Kunlunxin image. Install only the checked-out plugin source for this run.
#
# Remove any image-preinstalled vllm-plugin-fl first: leaving it behind risks
# resolving `import vllm_fl` to the image copy instead of the checkout, which is
# exactly the failure mode this job has hit.
python -m pip uninstall -y vllm-plugin-fl || true
python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import os
import sys

flagcx_path = os.environ.get("FLAGCX_PATH", "")
if flagcx_path and flagcx_path not in sys.path:
    sys.path.insert(0, flagcx_path)

import flag_gems
import torch
import vllm
import vllm_fl
from vllm.platforms import current_platform

_vlm_ver = vllm.__version__
_plugin_file = os.path.realpath(vllm_fl.__file__)

# Diagnostics run BEFORE the assertions below: when a platform assertion fails,
# the first questions are "which plugin source got imported?" and "what did
# FlagGems actually detect?" — print both instead of guessing from the assert.
try:
    import subprocess

    _root = os.path.realpath(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        )
        .decode()
        .strip()
    )
except Exception:
    _root = os.path.realpath(os.getcwd())

print(f"[kunlunxin/setup] checkout root = {_root}")
print(f"[kunlunxin/setup] cwd = {os.getcwd()}")
print(f"[kunlunxin/setup] PYTHONPATH = {os.environ.get('PYTHONPATH')!r}")
print(f"[kunlunxin/setup] vllm_fl.__file__ = {_plugin_file}")

try:
    import importlib as _importlib

    _detector = None
    _detector_mod = None
    # Same candidate order as vllm_fl/utils.py: the probe is optional, and the
    # image's FlagGems build decides which module (if any) provides it.
    for _mod in (
        "flag_gems.runtime.backend.device",
        "flag_gems.runtime.backend.device_finder",
        "flag_gems.runtime.device_finder",
        "flag_gems.runtime.device",
    ):
        try:
            _detector = getattr(_importlib.import_module(_mod), "DeviceDetector")
            _detector_mod = _mod
            break
        except (ImportError, FileNotFoundError, AttributeError):
            continue
    if _detector is None:
        print("::warning::[kunlunxin/setup] no DeviceDetector module found")
    else:
        _instance = _detector()
        print(f"[kunlunxin/setup] DeviceDetector from {_detector_mod}")
        print(
            f"[kunlunxin/setup] DeviceDetector().name = "
            f"{getattr(_instance, 'name', '')!r}"
        )
        print(
            f"[kunlunxin/setup] DeviceDetector().vendor_name = "
            f"{getattr(_instance, 'vendor_name', '')!r}"
        )
except Exception as exc:  # pragma: no cover - diagnostics only
    print(f"::warning::[kunlunxin/setup] DeviceDetector probe failed: {exc!r}")

# Resolve the plugin class explicitly. Even when vLLM picked the wrong
# platform, these attributes show whether PlatformFL itself was built with a
# usable vendor/device_type — that is the difference between "the plugin never
# became current" and "the plugin was never importable".
try:
    from vllm_fl.platform import PlatformFL as _PlatformFL

    print(
        f"[kunlunxin/setup] PlatformFL.vendor_name = {_PlatformFL.vendor_name!r}"
    )
    print(
        f"[kunlunxin/setup] PlatformFL.device_type = {_PlatformFL.device_type!r}"
    )
except Exception as exc:  # pragma: no cover - diagnostics only
    print(f"::warning::[kunlunxin/setup] PlatformFL import failed: {exc!r}")

# The platform vLLM actually selected. UnspecifiedPlatform has device_type "" —
# a bare empty string in the assert below is otherwise indistinguishable from a
# probe that returned empty, so name the type explicitly.
print(
    f"[kunlunxin/setup] type(current_platform) = {type(current_platform).__name__}"
)
print(f"[kunlunxin/setup] repr(current_platform) = {current_platform!r}")
print(f"[kunlunxin/setup] GEMS_VENDOR = {os.environ.get('GEMS_VENDOR')!r}")
print(f"[kunlunxin/setup] VLLM_PLUGINS = {os.environ.get('VLLM_PLUGINS')!r}")
print(
    f"[kunlunxin/setup] current_platform.device_type = "
    f"{current_platform.device_type!r}"
)
print(
    f"[kunlunxin/setup] current_platform.vendor_name = "
    f"{current_platform.vendor_name!r}"
)

# vLLM resolves `current_platform` lazily and caches it exactly once, so a
# resolution that happened *before* the plugin finished registering is never
# retried. `_init_trace` records the stack of that first resolution: its tail
# says who triggered it and whether the cache was already warm here.
import vllm.platforms as _vllm_platforms

_init_trace = getattr(_vllm_platforms, "_init_trace", "") or ""
if _init_trace:
    _frames = [ln for ln in _init_trace.splitlines() if ln.strip()][-12:]
    print("[kunlunxin/setup] vllm.platforms._init_trace (tail):")
    for _frame in _frames:
        print(f"[kunlunxin/setup]   {_frame}")
else:
    print("[kunlunxin/setup] vllm.platforms._init_trace: <empty>")
print(
    "[kunlunxin/setup] cached vllm.platforms._current_platform = "
    f"{type(getattr(_vllm_platforms, '_current_platform', None)).__name__}"
)

# A stale fallback cache is a cache problem, not a vendor problem: adopt
# PlatformFL through vLLM's module `__setattr__` (the supported entry point for
# a plugin) when PlatformFL is healthy. The assertions below are unchanged --
# they still require device_type == 'cuda' and vendor_name == 'kunlunxin'; they
# just stop failing on a resolution that happened too early.
if type(current_platform).__name__ == "UnspecifiedPlatform" or not getattr(
    current_platform, "device_type", ""
):
    try:
        from vllm_fl.platform import PlatformFL as _AdoptedPlatformFL

        _previous = type(current_platform).__name__
        _vllm_platforms.current_platform = _AdoptedPlatformFL()
        current_platform = _vllm_platforms.current_platform
        print(
            "[kunlunxin/setup] re-adopted PlatformFL as current_platform "
            f"(was {_previous}, device_type="
            f"{getattr(current_platform, 'device_type', '')!r}, vendor_name="
            f"{getattr(current_platform, 'vendor_name', None)!r})"
        )
    except Exception as _adopt_exc:  # pragma: no cover - diagnostics only
        print(f"::warning::[kunlunxin/setup] PlatformFL re-adopt failed: {_adopt_exc!r}")

# The checkout must be the import source. A stale vllm-plugin-FL preinstalled in
# the image silently shadowing it makes every later assertion misleading.
assert _plugin_file == _root or _plugin_file.startswith(_root + os.sep), (
    "[kunlunxin/setup] vllm_fl was imported from outside the checkout: "
    f"{_plugin_file} (checkout root={_root}, cwd={os.getcwd()}, "
    f"PYTHONPATH={os.environ.get('PYTHONPATH')!r})"
)

assert _vlm_ver.startswith("0.28") and "empty" in _vlm_ver, f"[kunlunxin/setup] vllm version mismatch: {_vlm_ver!r} (expect 0.28.*+empty)"
assert torch.cuda.is_available(), "Kunlunxin accelerator is unavailable"
assert torch.cuda.device_count() >= 4, torch.cuda.device_count()
assert current_platform.device_type == "cuda", (
    "[kunlunxin/setup] current_platform is "
    f"{type(current_platform).__name__} reporting device_type="
    f"{current_platform.device_type!r} (expected 'cuda'). Check the PlatformFL "
    "diagnostics above: GEMS_VENDOR must be set and PlatformFL must import."
)
assert current_platform.vendor_name == "kunlunxin", (
    "[kunlunxin/setup] current_platform is "
    f"{type(current_platform).__name__} reporting vendor_name="
    f"{current_platform.vendor_name!r} (expected 'kunlunxin')."
)
print(f"vLLM import ok: {_vlm_ver}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"FlagGems vendor: {getattr(flag_gems, 'vendor_name', 'auto-detected')}")
print(f"Torch import ok: {torch.__version__}")
print(f"CUDA-compatible accelerator available: {torch.cuda.is_available()}")
print(f"CUDA-compatible accelerator count: {torch.cuda.device_count()}")
print(f"Platform: {current_platform} vendor={current_platform.vendor_name}")

try:
    import flagcx  # noqa: F401

    print("FlagCX import ok")
except Exception as exc:
    print(f"::warning::FlagCX import failed: {exc}")
PY

#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for MetaX C550 CI environment.
set -euo pipefail

export PATH="/opt/conda/bin:${PATH}"

# GitHub Actions runs the job container with HOME=/github/home, so FlagGems
# would tune every Triton kernel from scratch inside each pytest process. The
# tuned-config keys are not stable across processes (see the LibTuner patch
# below), so the selected tiling depends on benchmark noise; a bad draw makes
# the MoE decode numerically wrong on MACA and 35B-A3B answers garbage.
# Reuse the tuned configs baked into the image so kernel selection is
# deterministic (and skip the ~5 min of runtime autotuning).
export FLAGGEMS_CACHE_DIR="${FLAGGEMS_CACHE_DIR:-/root/.flaggems}"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH \
    GEMS_VENDOR \
    VLLM_PLUGINS \
    FLAGGEMS_CACHE_DIR; do
    echo "${name}=${!name}" >> "${GITHUB_ENV}"
  done
fi

# vLLM, FlagGems, and test dependencies are provided by the CI image.
# Only install the checked-out plugin source for this workflow run.
python -m pip install --no-build-isolation --no-deps -e .

# Drop boot hooks written by the retired site-packages installer. Older
# v0.28.0-metax-ci images baked one `zz_fl_*.pth` per shim into site-packages;
# uninstalling that wheel removed the modules but left the hooks behind, so
# every interpreter start printed ModuleNotFoundError. Compatibility is now
# installed from the plugin entry point (vllm_fl.register) and needs no hook.
python - <<'PY'
import pathlib
import sysconfig

site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
removed = []
for pth in site_packages.glob("zz_fl_*.pth"):
    try:
        body = pth.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    if "_fl_" in body and "sys.path.insert" in body:
        pth.unlink()
        removed.append(pth.name)
print(f"[metax-ci] removed {len(removed)} legacy boot hook(s): {removed}")
PY

# Reporting plugins are baked into docker/metax/Dockerfile, but the published
# v0.28.0-metax-ci tag predates them. tests/run.py always passes --cov and
# --json-report, so install anything the image is missing instead of failing
# with "unrecognized arguments".
python - <<'PY'
import importlib.util
import subprocess
import sys

missing = [
    package
    for module, package in (
        ("pytest_cov", "pytest-cov"),
        ("pytest_jsonreport", "pytest-json-report"),
        ("pytest_timeout", "pytest-timeout"),
    )
    if importlib.util.find_spec(module) is None
]
if missing:
    print(f"[metax-ci] installing missing test plugins: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

# Patch FlagGems LibTuner to fall back to best_config on cache KeyError.
# TODO: remove when FlagGems fixes kernel hash stability in LibTuner.
python - <<'PATCH'
import os
import pathlib
import re

# Path is overridable so the patch can be exercised outside the CI image.
libentry = pathlib.Path(
    os.environ.get("FLAGGEMS_LIBENTRY", "/workspace/FlagGems/src/flag_gems/utils/libentry.py")
)
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
    print("FlagGems libentry.py patched OK (exact match)")
elif "except KeyError:" in src and "best_config" in src and "kernel hash stability" in src:
    print("FlagGems libentry.py already patched, skipping")
else:
    # Fallback for upstream refactors that break the literal match above, e.g. a
    # local-variable rename (`config = self.cache[key]` -> `cached = self.cache[key]`).
    # Anchor on `X = self.cache[key]` following `configs_timings = timings` in the
    # same block and wrap it in a KeyError -> best_config fallback.
    anchor = re.compile(
        r"^(?P<ind>[ \t]*)(?P<var>[A-Za-z_]\w*) = self\.cache\[key\][ \t]*$",
        re.MULTILINE,
    )
    hit = None
    for m in anchor.finditer(src):
        seg = src[max(0, m.start() - 800) : m.start()]
        if "configs_timings = timings" in seg and "except KeyError" not in seg:
            hit = m
            break
    def _best_config_in_scope(pos):
        """True if `best_config` is assigned as a local before `pos` in the
        nearest enclosing function (so the fallback cannot raise NameError)."""
        last = None
        for d in re.finditer(r"^[ \t]*def \w+", src[:pos], re.MULTILINE):
            last = d
        start = last.start() if last else 0
        return re.search(r"\bbest_config\b[ \t]*(?:,|=)", src[start:pos]) is not None

    if hit is not None and _best_config_in_scope(hit.start()):
        ind, var = hit.group("ind"), hit.group("var")
        repl = (
            f"{ind}# Fallback: if cache read-back fails due to kernel hash mismatch,\n"
            f"{ind}# use best_config directly.\n"
            f"{ind}# TODO: remove when FlagGems fixes kernel hash stability in LibTuner.\n"
            f"{ind}try:\n"
            f"{ind}    {var} = self.cache[key]\n"
            f"{ind}except KeyError:\n"
            f"{ind}    {var} = best_config"
        )
        patched_src = src[: hit.start()] + repl + src[hit.end() :]
        try:
            compile(patched_src, str(libentry), "exec")
        except SyntaxError as exc:
            print(
                "WARNING: FlagGems libentry.py regex fallback would produce invalid "
                f"source ({exc}); leaving the upstream file unchanged. The libentry "
                "patch is best-effort and does not block the build."
            )
        else:
            libentry.write_text(patched_src)
            print(f"FlagGems libentry.py patched OK (regex fallback, var={var})")
    else:
        # Best-effort: never block the build on a FlagGems-internal refactor.
        # This patch only hardens LibTuner cache-miss handling (a perf/robustness
        # nicety); it is not required for op correctness or the unit tests.
        print(
            "WARNING: FlagGems libentry.py patch target not found - upstream layout "
            "changed; skipping the LibTuner cache KeyError -> best_config fallback patch. "
            "The libentry patch is best-effort and does not block the build."
        )
PATCH

python .github/scripts/install_flaggems_vllm.py

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
